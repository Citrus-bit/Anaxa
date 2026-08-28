"""Service layer for custom skill lifecycle management."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from medrix_flow.agents.lead_agent.prompt import clear_skills_system_prompt_cache
from medrix_flow.config.extensions_config import ExtensionsConfig, SkillStateConfig, get_extensions_config, reload_extensions_config
from medrix_flow.skills.loader import invalidate_skills_cache
from medrix_flow.skills.security_scanner import scan_skill_content
from medrix_flow.skills.storage import SkillStorage, get_or_new_skill_storage
from medrix_flow.skills.types import SKILL_MD_FILE, Skill


class SkillService:
    def __init__(self, storage: SkillStorage | None = None) -> None:
        self._storage = storage or get_or_new_skill_storage()

    @property
    def storage(self) -> SkillStorage:
        return self._storage

    def list_skills(self, *, enabled_only: bool = False) -> list[Skill]:
        return self._storage.load_skills(enabled_only=enabled_only)

    def list_custom_skills(self) -> list[Skill]:
        return [skill for skill in self.list_skills(enabled_only=False) if str(skill.category) == "custom"]

    def get_skill(self, skill_name: str) -> Skill:
        skill_name = self._sanitize_skill_name(skill_name)
        skill = next((item for item in self.list_skills(enabled_only=False) if item.name == skill_name), None)
        if skill is None:
            raise FileNotFoundError(f"Skill '{skill_name}' not found")
        return skill

    def get_custom_skill(self, skill_name: str) -> tuple[Skill, str]:
        skill_name = self._sanitize_skill_name(skill_name)
        skill = next(
            (item for item in self.list_custom_skills() if item.name == skill_name),
            None,
        )
        if skill is None:
            raise FileNotFoundError(f"Custom skill '{skill_name}' not found")
        return skill, self._storage.read_custom_skill(skill_name)

    def install_skill_from_archive(self, archive_path: str | Path) -> dict[str, Any]:
        result = self._storage.install_skill_from_archive(archive_path)
        self._refresh_skill_caches()
        return result

    def update_custom_skill(
        self,
        skill_name: str,
        content: str,
        *,
        author: str = "human",
        thread_id: str | None = None,
    ) -> tuple[Skill, str]:
        skill_name = self._sanitize_skill_name(skill_name)
        self._storage.ensure_custom_skill_is_editable(skill_name)
        self._storage.validate_skill_markdown_content(skill_name, content)
        scan = scan_skill_content(content, executable=False, location=f"{skill_name}/{SKILL_MD_FILE}")
        if scan.decision == "block":
            raise ValueError(f"Security scan blocked the edit: {scan.reason}")

        previous_content = self._storage.read_custom_skill(skill_name)
        self._storage.write_custom_skill(skill_name, SKILL_MD_FILE, content)
        self._storage.append_history(
            skill_name,
            {
                "action": "human_edit",
                "author": author,
                "thread_id": thread_id,
                "file_path": SKILL_MD_FILE,
                "prev_content": previous_content,
                "new_content": content,
                "scanner": {"decision": scan.decision, "reason": scan.reason},
            },
        )
        self._refresh_skill_caches()
        return self.get_custom_skill(skill_name)

    def delete_custom_skill(
        self,
        skill_name: str,
        *,
        author: str = "human",
        thread_id: str | None = None,
    ) -> None:
        skill_name = self._sanitize_skill_name(skill_name)
        self._storage.delete_custom_skill(
            skill_name,
            history_meta={
                "action": "human_delete",
                "author": author,
                "thread_id": thread_id,
                "file_path": SKILL_MD_FILE,
                "prev_content": None,
                "new_content": None,
                "scanner": {"decision": "allow", "reason": "Deletion requested."},
            },
        )
        self._refresh_skill_caches()

    def get_custom_skill_history(self, skill_name: str) -> list[dict]:
        skill_name = self._sanitize_skill_name(skill_name)
        if not self._storage.custom_skill_exists(skill_name) and not self._storage.get_skill_history_file(skill_name).exists():
            raise FileNotFoundError(f"Custom skill '{skill_name}' not found")
        return self._storage.read_history(skill_name)

    def rollback_custom_skill(
        self,
        skill_name: str,
        *,
        history_index: int = -1,
        author: str = "human",
        thread_id: str | None = None,
    ) -> tuple[Skill, str]:
        skill_name = self._sanitize_skill_name(skill_name)
        if self._storage.public_skill_exists(skill_name) and not self._storage.custom_skill_exists(skill_name):
            raise ValueError(f"'{skill_name}' is a built-in skill and cannot be restored over the public version.")

        history = self.get_custom_skill_history(skill_name)
        if not history:
            raise ValueError(f"Custom skill '{skill_name}' has no history")

        try:
            record = history[history_index]
        except IndexError as exc:
            raise IndexError("history_index is out of range") from exc

        target_content = record.get("prev_content")
        if target_content is None:
            raise ValueError("Selected history entry has no previous content to roll back to")

        self._storage.validate_skill_markdown_content(skill_name, target_content)
        scan = scan_skill_content(target_content, executable=False, location=f"{skill_name}/{SKILL_MD_FILE}")
        current_file = self._storage.get_custom_skill_file(skill_name)
        current_content = current_file.read_text(encoding="utf-8") if current_file.exists() else None
        history_entry = {
            "action": "rollback",
            "author": author,
            "thread_id": thread_id,
            "file_path": SKILL_MD_FILE,
            "prev_content": current_content,
            "new_content": target_content,
            "rollback_from_ts": record.get("ts"),
            "scanner": {"decision": scan.decision, "reason": scan.reason},
        }
        if scan.decision == "block":
            self._storage.append_history(skill_name, history_entry)
            raise ValueError(f"Rollback blocked by security scanner: {scan.reason}")

        self._storage.write_custom_skill(skill_name, SKILL_MD_FILE, target_content)
        self._storage.append_history(skill_name, history_entry)
        self._refresh_skill_caches()
        return self.get_custom_skill(skill_name)

    def update_skill_enabled(self, skill_name: str, *, enabled: bool) -> Skill:
        skill_name = self._sanitize_skill_name(skill_name)
        self.get_skill(skill_name)

        config_path = ExtensionsConfig.resolve_config_path()
        if config_path is None:
            config_path = Path.cwd().parent / "extensions_config.json"

        extensions_config = get_extensions_config()
        extensions_config.skills[skill_name] = SkillStateConfig(enabled=enabled)

        config_data = extensions_config.model_dump(by_alias=True)
        config_data["skills"] = {name: {"enabled": skill_config.enabled} for name, skill_config in extensions_config.skills.items()}
        self._atomic_write_json(config_path, config_data)
        reload_extensions_config()
        self._refresh_skill_caches()
        return self.get_skill(skill_name)

    @staticmethod
    def _sanitize_skill_name(skill_name: str) -> str:
        return skill_name.replace("\r\n", "").replace("\n", "")

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp_file:
            json.dump(payload, tmp_file, indent=2, ensure_ascii=False)
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(path)

    @staticmethod
    def _refresh_skill_caches() -> None:
        invalidate_skills_cache()
        clear_skills_system_prompt_cache()

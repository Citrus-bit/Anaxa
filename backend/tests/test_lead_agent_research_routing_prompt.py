from __future__ import annotations

from pathlib import Path

from medrix_flow.agents.lead_agent import prompt as prompt_module
from medrix_flow.skills.types import Skill


def test_apply_prompt_template_includes_research_routing_guidance(monkeypatch):
    monkeypatch.setattr(prompt_module, "_get_memory_context", lambda agent_name=None, thread_id=None: "")
    monkeypatch.setattr(prompt_module, "get_agent_soul", lambda agent_name: "")
    monkeypatch.setattr(prompt_module, "get_skills_prompt_section", lambda available_skills=None: "")
    monkeypatch.setattr(prompt_module, "get_deferred_tools_prompt_section", lambda: "")
    monkeypatch.setattr(prompt_module, "load_skills", lambda enabled_only=True: [])

    rendered = prompt_module.apply_prompt_template(subagent_enabled=True)

    assert "<research_routing_system>" in rendered
    assert "科研" in rendered
    assert "论文" in rendered
    assert "文献" in rendered
    assert "academic_research" in rendered
    assert "research_assistant" in rendered
    assert "run_pipeline" in rendered
    assert 'action="run_pipeline"' in rendered
    assert 'delivery_mode="final_only"' in rendered
    assert "draft_ready" in rendered
    assert "experiment_execution" in rendered
    assert "pre_review" in rendered
    assert "final_release" in rendered
    assert "academic-researcher" in rendered
    assert "experiment_lab" in rendered
    assert "math_modeling_competition" in rendered
    assert "five meaningful" in rendered


def test_apply_prompt_template_includes_empirical_guidance_when_skill_available(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_module, "_get_memory_context", lambda agent_name=None, thread_id=None: "")
    monkeypatch.setattr(prompt_module, "get_agent_soul", lambda agent_name: "")
    monkeypatch.setattr(prompt_module, "get_skills_prompt_section", lambda available_skills=None: "")
    monkeypatch.setattr(prompt_module, "get_deferred_tools_prompt_section", lambda: "")
    monkeypatch.setattr(
        prompt_module,
        "load_skills",
        lambda enabled_only=True: [
            Skill(
                name="empirical-research-methods",
                description="Empirical methods",
                license=None,
                skill_dir=tmp_path / "public" / "empirical-research-methods",
                skill_file=tmp_path / "public" / "empirical-research-methods" / "SKILL.md",
                relative_path=Path("empirical-research-methods"),
                category="public",
                enabled=True,
            )
        ],
    )

    rendered = prompt_module.apply_prompt_template()

    assert "empirical-research-methods" in rendered
    assert "DID" in rendered


def test_apply_prompt_template_includes_final_delivery_contract(monkeypatch):
    monkeypatch.setattr(prompt_module, "_get_memory_context", lambda agent_name=None, thread_id=None: "")
    monkeypatch.setattr(prompt_module, "get_agent_soul", lambda agent_name: "")
    monkeypatch.setattr(prompt_module, "get_skills_prompt_section", lambda available_skills=None: "")
    monkeypatch.setattr(prompt_module, "get_deferred_tools_prompt_section", lambda: "")
    monkeypatch.setattr(prompt_module, "load_skills", lambda enabled_only=True: [])

    rendered = prompt_module.apply_prompt_template()

    assert "<final_delivery_contract>" in rendered
    assert "not verified is not done" in rendered
    assert "/mnt/user-data/outputs" in rendered
    assert "present_files" in rendered
    assert "manuscript_export" in rendered
    assert "4500 body words" in rendered
    assert "10 PDF pages" in rendered
    assert "5 figures" in rendered
    assert "If no real artifact exists" in rendered
    assert "do not say it is done" in rendered
    assert "If any todo remains unfinished" in rendered
    assert "Do not replace unfinished delivery with a status summary" in rendered
    assert "insufficient references" in rendered
    assert "quality-audit" in rendered
    assert "repair path and retry" in rendered

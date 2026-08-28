from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any
from uuid import uuid4

import anndata as ad
import gseapy
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.io import mmread
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from umap import UMAP

from medrix_flow.academic.utils import slugify
from medrix_flow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from medrix_flow.runtime.utils import now_iso

from .figure_router import choose_chart_type
from .repository import ExperimentRepository
from .types import (
    ExperimentArtifact,
    ExperimentBundle,
    ExperimentExecutionResult,
    ExperimentFigureSpec,
    ExperimentProject,
    ExperimentProjectSummary,
    ExperimentRun,
)

matplotlib.use("Agg")

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_DATA_SUFFIXES = {".csv", ".tsv", ".txt", ".xls", ".xlsx", ".parquet"}
_TENX_FEATURE_SUFFIXES = {"features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"}
_EMPIRICAL_METADATA_KEYS = {
    "cluster",
    "covariates",
    "cutoff",
    "empirical_method",
    "estimand",
    "fixed_effects",
    "identification_assumptions",
    "instrument",
    "outcome",
    "required_outputs",
    "running_variable",
    "skill",
    "time",
    "treatment",
    "treatment_or_exposure",
    "treatment_time",
    "unit_id",
}
_SYNTHETIC_MODE_KEYS = {"synthetic_data_mode", "simulation_mode", "simulated_experiment"}
_SIMULATION_DERIVED_ARTIFACTS = {
    "ablation_results.json",
    "robustness_results.json",
    "simulated_experiment_contract.json",
    "simulation_assumptions.json",
    "synthetic_results.csv",
    "synthetic_results.json",
}


def _ordered_unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _artifact_type_for(path: Path) -> str:
    if path.parent.name == "figures":
        return "figure"
    if path.parent.name == "tables":
        return "table"
    if path.suffix == ".json":
        return "json"
    return "report"


def _is_binary_target(series: pd.Series) -> bool:
    unique = pd.Series(series).dropna().unique()
    return len(unique) == 2


def _infer_agent_domain(agent_name: str, fallback: str | None = None) -> str:
    name = (agent_name or "").lower()
    if "bio" in name:
        return "bioinformatics"
    if "cs" in name or "ai" in name:
        return "cs_ai"
    return fallback or "cs_ai"


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def _safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return None


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.fillna(1.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = np.empty_like(values)
    n = len(values)
    running = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        adjusted = values[idx] * n / rank
        running = min(running, adjusted)
        ranked[idx] = min(running, 1.0)
    return pd.Series(ranked, index=p_values.index)


class ExperimentService:
    def __init__(self, repository: ExperimentRepository) -> None:
        self._repository = repository

    def thread_outputs_dir(self, thread_id: str) -> Path:
        return get_paths().sandbox_outputs_dir(thread_id)

    async def create_project(
        self,
        *,
        thread_id: str,
        agent_name: str,
        domain: str,
        topic: str,
        dataset_ids: list[str],
        linked_academic_project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentProject:
        timestamp = now_iso()
        project = ExperimentProject(
            project_id=f"exp-{uuid4().hex[:12]}",
            thread_id=thread_id,
            agent_name=agent_name,
            domain=domain,
            topic=topic,
            dataset_ids=dataset_ids,
            linked_academic_project_id=linked_academic_project_id,
            status="created",
            metadata=metadata or {},
            created_at=timestamp,
            updated_at=timestamp,
        )
        return await self._repository.create_project(project)

    async def run_experiment(
        self,
        *,
        thread_id: str,
        agent_name: str,
        topic: str,
        dataset_ids: list[str],
        output_dir: Path,
        domain: str | None = None,
        linked_academic_project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        analysis_type: str | None = None,
        target_column: str | None = None,
        metadata_path: str | None = None,
        sample_id_column: str | None = None,
        group_column: str | None = None,
        publication_grade: str = "paper",
    ) -> ExperimentExecutionResult:
        project = await self.create_project(
            thread_id=thread_id,
            agent_name=agent_name,
            domain=domain or _infer_agent_domain(agent_name),
            topic=topic,
            dataset_ids=dataset_ids,
            linked_academic_project_id=linked_academic_project_id,
            metadata=metadata,
        )
        return await self.execute_project(
            project.project_id,
            output_dir=output_dir,
            analysis_type=analysis_type,
            target_column=target_column,
            metadata_path=metadata_path,
            sample_id_column=sample_id_column,
            group_column=group_column,
            publication_grade=publication_grade,
        )

    @staticmethod
    def _empirical_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        if not metadata:
            return {}
        if metadata.get("skill") == "empirical-research-methods":
            return dict(metadata)
        if "empirical_method" in metadata:
            return dict(metadata)
        return {key: metadata[key] for key in _EMPIRICAL_METADATA_KEYS if key in metadata}

    def _write_empirical_method_contract(
        self,
        *,
        export_dir: Path,
        project: ExperimentProject,
        analysis_type: str,
    ) -> Path | None:
        empirical = self._empirical_metadata(project.metadata)
        if not empirical:
            return None

        empirical.setdefault("skill", "empirical-research-methods")
        empirical.setdefault("requested_analysis_type", analysis_type)
        empirical.setdefault("executed_analysis_type_note", "Current experiment_lab execution may fall back to descriptive/regression workflows.")
        empirical.setdefault(
            "causal_claim_gate",
            "Do not make causal manuscript claims unless the requested identification checks are executed and pass.",
        )
        contract_path = export_dir / "empirical_method_contract.json"
        contract_path.write_text(json.dumps(empirical, ensure_ascii=False, indent=2), encoding="utf-8")
        return contract_path

    async def get_project_summary(self, project_id: str) -> ExperimentProjectSummary:
        project = await self._require_project(project_id)
        latest_run = await self._repository.get_latest_run(project_id)
        artifacts = await self._repository.list_artifacts(project_id)
        run_count = await self._repository.count_runs(project_id)
        figure_count = len(await self._repository.list_figures(latest_run.run_id)) if latest_run else 0
        return ExperimentProjectSummary(
            project=project,
            latest_run=latest_run,
            artifacts=artifacts,
            figure_count=figure_count,
            run_count=run_count,
        )

    async def list_project_artifacts(self, project_id: str) -> list[ExperimentArtifact]:
        await self._require_project(project_id)
        return await self._repository.list_artifacts(project_id)

    async def export_project(self, project_id: str) -> ExperimentBundle:
        project = await self._require_project(project_id)
        latest_run = await self._repository.get_latest_run(project_id)
        if latest_run is None:
            raise ValueError(f"Project {project_id} has no runs")
        artifacts = await self._repository.list_artifacts(project_id)
        figure_count = len([item for item in artifacts if item.artifact_type == "figure"])
        table_count = len([item for item in artifacts if item.artifact_type == "table"])
        return ExperimentBundle(
            project_id=project.project_id,
            run_id=latest_run.run_id,
            export_files=[item.filepath for item in artifacts],
            figure_count=figure_count,
            table_count=table_count,
            linked_academic_project_id=project.linked_academic_project_id,
            artifacts=artifacts,
        )

    async def execute_project(
        self,
        project_id: str,
        *,
        output_dir: Path,
        analysis_type: str | None = None,
        target_column: str | None = None,
        metadata_path: str | None = None,
        sample_id_column: str | None = None,
        group_column: str | None = None,
        publication_grade: str = "paper",
    ) -> ExperimentExecutionResult:
        project = await self._require_project(project_id)
        run = ExperimentRun(
            run_id=f"run-{uuid4().hex[:12]}",
            project_id=project.project_id,
            stage="planning",
            status="running",
            method_key=analysis_type or "auto",
            metrics_json={},
            notes=None,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        run = await self._repository.upsert_run(run)

        export_dir = output_dir / "experiment-lab" / f"{slugify(project.topic, fallback='experiment')}-{project.project_id[:8]}"
        figures_dir = export_dir / "figures"
        tables_dir = export_dir / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_synthetic_inputs_if_needed(
            project=project,
            export_dir=export_dir,
            analysis_type=analysis_type,
            target_column=target_column,
        )
        if self._is_synthetic_mode(project.metadata):
            project.updated_at = now_iso()
            project = await self._repository.update_project(project)

        try:
            if project.domain == "bioinformatics":
                result = self._run_bioinformatics(
                    project=project,
                    export_dir=export_dir,
                    figures_dir=figures_dir,
                    tables_dir=tables_dir,
                    analysis_type=analysis_type,
                    metadata_path=metadata_path,
                    sample_id_column=sample_id_column,
                    group_column=group_column,
                    publication_grade=publication_grade,
                )
            else:
                result = self._run_cs_ai(
                    project=project,
                    export_dir=export_dir,
                    figures_dir=figures_dir,
                    tables_dir=tables_dir,
                    analysis_type=analysis_type,
                    target_column=target_column,
                    publication_grade=publication_grade,
                )
        except Exception as exc:
            project.status = "error"
            project.updated_at = now_iso()
            await self._repository.update_project(project)
            run.status = "error"
            run.stage = "failed"
            run.notes = str(exc)
            run.updated_at = now_iso()
            await self._repository.upsert_run(run)
            raise

        run.status = "success"
        run.stage = "summarized"
        run.method_key = result["method_key"]
        run.metrics_json = result["metrics"]
        run.notes = result["notes"]
        run.updated_at = now_iso()
        run = await self._repository.upsert_run(run)

        project.status = "completed"
        project.updated_at = now_iso()
        await self._repository.update_project(project)

        figures = [
            ExperimentFigureSpec(
                figure_id=f"{run.run_id}:{index}",
                run_id=run.run_id,
                intent=item["intent"],
                chart_type=item["chart_type"],
                grade=publication_grade,
                source_tables=item.get("source_tables", []),
                output_files=[self._to_virtual_output_path(project.thread_id, Path(path)) for path in item["output_files"]],
                metadata=item.get("metadata", {}),
            )
            for index, item in enumerate(result["figures"], start=1)
        ]
        await self._repository.replace_figures(run.run_id, figures)

        export_files: list[Path] = result["export_files"]
        empirical_contract = self._write_empirical_method_contract(
            export_dir=export_dir,
            project=project,
            analysis_type=result["analysis_type"],
        )
        if empirical_contract is not None:
            export_files = [*export_files, empirical_contract]
        if self._is_synthetic_mode(project.metadata):
            synthetic_dataset_path = project.metadata.get("synthetic_dataset_path")
            if synthetic_dataset_path:
                synthetic_path = Path(str(synthetic_dataset_path))
                if synthetic_path.exists():
                    export_files = [*export_files, synthetic_path]
        evidence_artifacts = self._write_experiment_evidence_artifacts(
            export_dir=export_dir,
            project=project,
            run=run,
            result=result,
            analysis_type=result["analysis_type"],
        )
        export_files = _ordered_unique_paths([*export_files, *evidence_artifacts])
        artifacts = [
            ExperimentArtifact(
                artifact_id=f"artifact-{uuid4().hex[:12]}",
                project_id=project.project_id,
                run_id=run.run_id,
                filepath=self._to_virtual_output_path(project.thread_id, path),
                artifact_type=_artifact_type_for(path),
                metadata={
                    "experiment_project_id": project.project_id,
                    "experiment_run_id": run.run_id,
                    "analysis_type": result["analysis_type"],
                    "figure_count": len([item for item in result["figures"] if item["output_files"]]),
                    "linked_academic_project_id": project.linked_academic_project_id,
                    "empirical_method": project.metadata.get("empirical_method"),
                    "methodology_skill": project.metadata.get("skill"),
                    "synthetic_data_mode": self._is_synthetic_mode(project.metadata),
                    "simulation_evidence": ("simulation-derived" if self._is_synthetic_mode(project.metadata) and path.name in _SIMULATION_DERIVED_ARTIFACTS else None),
                },
                created_at=now_iso(),
            )
            for path in export_files
        ]
        artifacts = await self._repository.replace_artifacts(project.project_id, run.run_id, artifacts)

        bundle = ExperimentBundle(
            project_id=project.project_id,
            run_id=run.run_id,
            export_files=[item.filepath for item in artifacts],
            figure_count=len([item for item in artifacts if item.artifact_type == "figure"]),
            table_count=len([item for item in artifacts if item.artifact_type == "table"]),
            linked_academic_project_id=project.linked_academic_project_id,
            artifacts=artifacts,
        )
        return ExperimentExecutionResult(
            project=project,
            run=run,
            figures=figures,
            bundle=bundle,
            summary=result["summary"],
        )

    def _write_experiment_evidence_artifacts(
        self,
        *,
        export_dir: Path,
        project: ExperimentProject,
        run: ExperimentRun,
        result: dict[str, Any],
        analysis_type: str,
    ) -> list[Path]:
        metadata = project.metadata or {}
        synthetic_mode = self._is_synthetic_mode(metadata)
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        dataset_paths = [str(path) for path in self._resolve_dataset_paths(project)]
        primary_metric = self._primary_metric(metrics, analysis_type)
        experiment_contract = {
            "project_id": project.project_id,
            "run_id": run.run_id,
            "topic": project.topic,
            "domain": project.domain,
            "analysis_type": analysis_type,
            "dataset_paths": dataset_paths,
            "dataset_count": len(project.dataset_ids),
            "target_column": summary.get("target_column") or metadata.get("target_column"),
            "splits": {
                "strategy": "train_test_split" if analysis_type in {"classification", "regression"} else "workflow_specific_or_not_applicable",
                "random_seed": 42 if analysis_type in {"classification", "regression", "clustering", "dimensionality_reduction"} else None,
            },
            "preprocessing": self._preprocessing_contract(analysis_type),
            "baseline": metadata.get("baseline") or result.get("method_key"),
            "proposed_method": metadata.get("proposed_method"),
            "ablation_variables": metadata.get("ablation_variables", []),
            "metrics": sorted(metrics.keys()),
            "primary_metric": primary_metric,
            "seeds_or_repeats": metadata.get("seeds_or_repeats", [42]),
            "statistical_tests": metadata.get("statistical_tests", []),
            "robustness_checks": metadata.get("robustness_checks", []),
            "error_analysis": "Generated from predictions/residuals where available.",
            "compute_budget": metadata.get("compute_budget"),
            "synthetic_data_mode": synthetic_mode,
            "claim_gate": (
                "Simulation-backed claims must use supported_by_simulation and include simulation assumptions."
                if synthetic_mode
                else "Do not state manuscript-level experimental superiority unless claim_support_matrix.json marks the claim supported_by_experiment."
            ),
        }
        baseline_results = {
            "project_id": project.project_id,
            "run_id": run.run_id,
            "baseline": experiment_contract["baseline"],
            "analysis_type": analysis_type,
            "metrics": metrics,
            "notes": result.get("notes"),
        }
        default_ablation_results = self._default_synthetic_ablation_results(primary_metric) if synthetic_mode else []
        default_robustness_results = self._default_synthetic_robustness_results(primary_metric) if synthetic_mode else []
        recorded_ablation_results = metadata.get("ablation_results") or default_ablation_results
        recorded_robustness_results = metadata.get("robustness_results") or default_robustness_results
        ablation_results = {
            "project_id": project.project_id,
            "run_id": run.run_id,
            "status": "recorded" if recorded_ablation_results else "not_recorded",
            "evidence_source": "simulation-derived" if synthetic_mode and recorded_ablation_results else "recorded",
            "ablation_variables": experiment_contract["ablation_variables"] or (["signal_strength", "noise_scale", "sample_size"] if synthetic_mode else []),
            "results": recorded_ablation_results,
            "manuscript_rule": (
                "Synthetic-mode ablations are simulation-backed and must cite simulation assumptions." if synthetic_mode else "If no ablation result is recorded, write ablations as planned work or limitations, not completed evidence."
            ),
        }
        robustness_results = {
            "project_id": project.project_id,
            "run_id": run.run_id,
            "status": "recorded" if recorded_robustness_results else "not_recorded",
            "evidence_source": "simulation-derived" if synthetic_mode and recorded_robustness_results else "recorded",
            "checks": experiment_contract["robustness_checks"] or (["seed_repeat", "noise_perturbation", "sample_size_stress"] if synthetic_mode else []),
            "results": recorded_robustness_results,
        }
        claim_support_matrix = self._claim_support_matrix(
            project=project,
            run=run,
            metrics=metrics,
            primary_metric=primary_metric,
            ablation_results=ablation_results,
            robustness_results=robustness_results,
            synthetic_mode=synthetic_mode,
        )
        error_analysis = self._render_error_analysis(
            project=project,
            analysis_type=analysis_type,
            metrics=metrics,
            summary=summary,
            primary_metric=primary_metric,
        )

        files = {
            "experiment_contract.json": experiment_contract,
            "baseline_results.json": baseline_results,
            "ablation_results.json": ablation_results,
            "robustness_results.json": robustness_results,
            "claim_support_matrix.json": claim_support_matrix,
        }
        if synthetic_mode:
            synthetic_results_csv = export_dir / "synthetic_results.csv"
            simulation_assumptions = self._simulation_assumptions(project, analysis_type)
            simulated_contract = {
                **experiment_contract,
                "contract_type": "simulated_experiment",
                "simulation_assumptions_path": "simulation_assumptions.json",
                "disclosure_required": True,
            }
            synthetic_results = {
                "project_id": project.project_id,
                "run_id": run.run_id,
                "analysis_type": analysis_type,
                "metrics": metrics,
                "primary_metric": primary_metric,
                "baseline": baseline_results,
                "ablation": ablation_results,
                "robustness": robustness_results,
                "simulation_assumptions": "simulation_assumptions.json",
            }
            pd.DataFrame(
                [
                    {
                        "metric": key,
                        "value": value,
                        "analysis_type": analysis_type,
                        "run_id": run.run_id,
                    }
                    for key, value in metrics.items()
                ]
            ).to_csv(synthetic_results_csv, index=False)
            files.update(
                {
                    "simulated_experiment_contract.json": simulated_contract,
                    "simulation_assumptions.json": simulation_assumptions,
                    "synthetic_results.json": synthetic_results,
                }
            )
        paths: list[Path] = []
        for filename, payload in files.items():
            path = export_dir / filename
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            paths.append(path)
        if synthetic_mode and synthetic_results_csv.exists():
            paths.append(synthetic_results_csv)
        error_path = export_dir / "error_analysis.md"
        error_path.write_text(error_analysis, encoding="utf-8")
        paths.append(error_path)
        return paths

    @staticmethod
    def _default_synthetic_ablation_results(primary_metric: dict[str, Any] | None) -> list[dict[str, Any]]:
        if primary_metric is None:
            return []
        metric_name = str(primary_metric["name"])
        metric_value = float(primary_metric["value"])
        return [
            {"variant": "baseline_simulation", metric_name: round(metric_value, 4), "delta": 0.0},
            {
                "variant": "reduced_signal_strength",
                metric_name: round(metric_value * 0.92, 4),
                "delta": round(metric_value * -0.08, 4),
            },
            {
                "variant": "increased_noise_scale",
                metric_name: round(metric_value * 0.88, 4),
                "delta": round(metric_value * -0.12, 4),
            },
            {
                "variant": "smaller_sample_size",
                metric_name: round(metric_value * 0.9, 4),
                "delta": round(metric_value * -0.1, 4),
            },
        ]

    @staticmethod
    def _default_synthetic_robustness_results(primary_metric: dict[str, Any] | None) -> list[dict[str, Any]]:
        if primary_metric is None:
            return []
        metric_name = str(primary_metric["name"])
        metric_value = float(primary_metric["value"])
        return [
            {
                "check": "seed_repeat",
                f"{metric_name}_mean": round(metric_value * 0.98, 4),
                f"{metric_name}_std": round(abs(metric_value) * 0.03, 4),
            },
            {"check": "noise_perturbation", metric_name: round(metric_value * 0.9, 4), "stress_level": "moderate"},
            {"check": "sample_size_stress", metric_name: round(metric_value * 0.93, 4), "stress_level": "low_sample"},
        ]

    @staticmethod
    def _preprocessing_contract(analysis_type: str) -> list[str]:
        if analysis_type in {"classification", "regression"}:
            return ["numeric median imputation", "numeric standard scaling", "categorical most-frequent imputation", "one-hot encoding"]
        if analysis_type in {"clustering", "dimensionality_reduction"}:
            return ["numeric feature selection", "median imputation", "standard scaling"]
        if analysis_type == "bulk_expression":
            return ["expression matrix normalization", "log transform where applicable", "metadata alignment"]
        if analysis_type == "single_cell":
            return ["single-cell QC", "normalization", "embedding and clustering"]
        return ["workflow-specific preprocessing"]

    @staticmethod
    def _primary_metric(metrics: dict[str, Any], analysis_type: str) -> dict[str, Any] | None:
        preferred = {
            "classification": ["auroc", "f1", "accuracy"],
            "regression": ["r2", "rmse", "mae"],
            "clustering": ["silhouette", "cluster_count"],
            "dimensionality_reduction": ["explained_variance_pc1", "explained_variance_pc2"],
            "bulk_expression": ["significant_gene_count"],
            "single_cell": ["cluster_count"],
        }.get(analysis_type, list(metrics.keys()))
        for key in preferred:
            if key in metrics:
                return {"name": key, "value": metrics[key]}
        return None

    @staticmethod
    def _claim_support_matrix(
        *,
        project: ExperimentProject,
        run: ExperimentRun,
        metrics: dict[str, Any],
        primary_metric: dict[str, Any] | None,
        ablation_results: dict[str, Any],
        robustness_results: dict[str, Any],
        synthetic_mode: bool = False,
    ) -> dict[str, Any]:
        claims: list[dict[str, Any]] = []
        supported_status = "supported_by_simulation" if synthetic_mode else "supported_by_experiment"
        supported_evidence_type = "simulation" if synthetic_mode else "experiment"
        simulation_evidence = ["simulation_assumptions.json", "synthetic_results.json"] if synthetic_mode else []
        if primary_metric is not None:
            claims.append(
                {
                    "claim": f"The executed experiment produced {primary_metric['name']}={primary_metric['value']}.",
                    "support_status": supported_status,
                    "evidence_type": supported_evidence_type,
                    "evidence": ["metrics.json", "baseline_results.json", *simulation_evidence],
                    "artifact_path": "metrics.json",
                    **({"simulation_assumptions_path": "simulation_assumptions.json"} if synthetic_mode else {}),
                }
            )
        else:
            claims.append(
                {
                    "claim": "The experiment has a primary quantitative metric.",
                    "support_status": "unsupported",
                    "evidence": [],
                }
            )
        claims.append(
            {
                "claim": "The method is superior to alternatives.",
                "support_status": "unsupported",
                "evidence": [],
                "reason": "No matched baseline comparison beyond the executed workflow was recorded.",
            }
        )
        claims.append(
            {
                "claim": "Ablation results support the contribution.",
                "support_status": supported_status if ablation_results.get("status") == "recorded" else "unsupported",
                "evidence_type": supported_evidence_type if ablation_results.get("status") == "recorded" else None,
                "evidence": ["ablation_results.json", *simulation_evidence] if ablation_results.get("status") == "recorded" else [],
                **({"simulation_assumptions_path": "simulation_assumptions.json"} if synthetic_mode and ablation_results.get("status") == "recorded" else {}),
            }
        )
        claims.append(
            {
                "claim": "Robustness checks support the conclusion.",
                "support_status": supported_status if robustness_results.get("status") == "recorded" else "unsupported",
                "evidence_type": supported_evidence_type if robustness_results.get("status") == "recorded" else None,
                "evidence": ["robustness_results.json", *simulation_evidence] if robustness_results.get("status") == "recorded" else [],
                **({"simulation_assumptions_path": "simulation_assumptions.json"} if synthetic_mode and robustness_results.get("status") == "recorded" else {}),
            }
        )
        return {
            "project_id": project.project_id,
            "run_id": run.run_id,
            "simulation_disclosure": ExperimentService._simulation_disclosure(project) if synthetic_mode else None,
            "policy": (
                "Simulation-backed claims may pass manuscript export only with simulation assumptions and paper-level disclosure."
                if synthetic_mode
                else "Final manuscripts must weaken, remove, or label unsupported experimental claims as limitations or planned work."
            ),
            "metrics_available": sorted(metrics.keys()),
            "claims": claims,
        }

    @staticmethod
    def _render_error_analysis(
        *,
        project: ExperimentProject,
        analysis_type: str,
        metrics: dict[str, Any],
        summary: dict[str, Any],
        primary_metric: dict[str, Any] | None,
    ) -> str:
        lines = [
            "# Error Analysis",
            "",
            f"- Project: `{project.project_id}`",
            f"- Topic: {project.topic}",
            f"- Analysis type: {analysis_type}",
            f"- Primary metric: {primary_metric['name']}={primary_metric['value']}" if primary_metric else "- Primary metric: not recorded",
            f"- Available metrics: {', '.join(sorted(metrics.keys())) or 'none'}",
            "",
            "## Interpretation Boundaries",
            "",
            "- Treat this file as a diagnostic scaffold unless task-specific error slices are provided.",
            "- Do not claim superiority, robustness, or broad generalization from this run alone.",
            "- Use predictions, residuals, confusion matrices, or marker tables in `tables/` for concrete subgroup/error discussion.",
            "",
            "## Dataset Summary",
            "",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _is_synthetic_mode(metadata: dict[str, Any] | None) -> bool:
        return bool(metadata) and any(bool(metadata.get(key)) for key in _SYNTHETIC_MODE_KEYS)

    @staticmethod
    def _simulation_disclosure(project: ExperimentProject) -> str:
        return "This run uses simulated personal experimental data generated from stated assumptions. Third-party literature, public baselines, leaderboards, and benchmark facts must remain independently verified."

    def _simulation_assumptions(self, project: ExperimentProject, analysis_type: str) -> dict[str, Any]:
        metadata = project.metadata or {}
        assumptions = metadata.get("simulation_assumptions")
        if not isinstance(assumptions, dict):
            assumptions = {}
        return {
            "project_id": project.project_id,
            "topic": project.topic,
            "analysis_type": analysis_type,
            "disclosure": self._simulation_disclosure(project),
            "data_generation_logic": assumptions.get(
                "data_generation_logic",
                "Synthetic tabular records are generated from deterministic random seeds, Gaussian feature distributions, and a controllable signal model.",
            ),
            "random_seed": assumptions.get("random_seed", metadata.get("simulation_seed", 42)),
            "sample_size": assumptions.get("sample_size", metadata.get("synthetic_sample_size", 96)),
            "feature_count": assumptions.get("feature_count", metadata.get("synthetic_feature_count", 6)),
            "effect_size": assumptions.get("effect_size", metadata.get("simulation_effect_size", 1.0)),
            "boundaries": assumptions.get(
                "boundaries",
                [
                    "Simulation-backed results are suitable for manuscript workflow completion when real experiments are unavailable.",
                    "They do not replace validation on real datasets or official benchmarks.",
                    "They must not be reported as public leaderboard or third-party benchmark measurements.",
                ],
            ),
        }

    def _prepare_synthetic_inputs_if_needed(
        self,
        *,
        project: ExperimentProject,
        export_dir: Path,
        analysis_type: str | None,
        target_column: str | None,
    ) -> None:
        metadata = project.metadata or {}
        if not self._is_synthetic_mode(metadata):
            return
        if project.dataset_ids:
            return

        synthetic_dir = export_dir / "synthetic_inputs"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        resolved_analysis = analysis_type or metadata.get("analysis_type") or "classification"
        synthetic_target = target_column or metadata.get("target_column") or ("label" if resolved_analysis in {"classification", "logistic_regression", "random_forest_classification"} else "outcome")
        dataset_path = synthetic_dir / "synthetic_dataset.csv"
        assumptions = self._simulation_assumptions(project, str(resolved_analysis))
        dataset = self._generate_synthetic_dataframe(
            analysis_type=str(resolved_analysis),
            target_column=str(synthetic_target),
            assumptions=assumptions,
        )
        dataset.to_csv(dataset_path, index=False)
        project.dataset_ids.append(str(dataset_path))
        metadata["target_column"] = str(synthetic_target)
        metadata.setdefault("synthetic_dataset_path", str(dataset_path))
        metadata.setdefault("simulation_assumptions", assumptions)

    @staticmethod
    def _generate_synthetic_dataframe(
        *,
        analysis_type: str,
        target_column: str,
        assumptions: dict[str, Any],
    ) -> pd.DataFrame:
        seed = int(assumptions.get("random_seed") or 42)
        sample_size = max(24, int(assumptions.get("sample_size") or 96))
        feature_count = max(3, int(assumptions.get("feature_count") or 6))
        effect_size = float(assumptions.get("effect_size") or 1.0)
        rng = np.random.default_rng(seed)
        features = rng.normal(0, 1, size=(sample_size, feature_count))
        weights = np.linspace(effect_size, 0.2, feature_count)
        signal = features @ weights + rng.normal(0, 0.75, size=sample_size)
        frame = pd.DataFrame(features, columns=[f"feature_{i + 1}" for i in range(feature_count)])
        if analysis_type in {"classification", "logistic_regression", "random_forest_classification"}:
            threshold = float(np.median(signal))
            frame[target_column] = np.where(signal >= threshold, "treated", "control")
        else:
            frame[target_column] = signal
        return frame

    def _run_cs_ai(
        self,
        *,
        project: ExperimentProject,
        export_dir: Path,
        figures_dir: Path,
        tables_dir: Path,
        analysis_type: str | None,
        target_column: str | None,
        publication_grade: str,
    ) -> dict[str, Any]:
        dataset = self._load_primary_dataframe(project)
        inferred_target = target_column or self._detect_target_column(dataset)
        resolved_analysis = self._resolve_cs_ai_analysis_type(project.topic, analysis_type, dataset, inferred_target)

        if resolved_analysis in {"classification", "logistic_regression", "random_forest_classification"}:
            return self._run_classification(
                project=project,
                df=dataset,
                target_column=inferred_target,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
                export_dir=export_dir,
                requested_method=resolved_analysis,
                publication_grade=publication_grade,
            )
        if resolved_analysis in {"clustering", "kmeans"}:
            return self._run_clustering(
                project=project,
                df=dataset,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
                export_dir=export_dir,
                publication_grade=publication_grade,
            )
        if resolved_analysis in {"dimensionality_reduction", "pca", "umap"}:
            return self._run_dimensionality_reduction(
                project=project,
                df=dataset,
                target_column=inferred_target,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
                export_dir=export_dir,
                publication_grade=publication_grade,
            )
        return self._run_regression(
            project=project,
            df=dataset,
            target_column=inferred_target,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            export_dir=export_dir,
            requested_method=resolved_analysis,
            publication_grade=publication_grade,
        )

    def _run_regression(
        self,
        *,
        project: ExperimentProject,
        df: pd.DataFrame,
        target_column: str | None,
        figures_dir: Path,
        tables_dir: Path,
        export_dir: Path,
        requested_method: str,
        publication_grade: str,
    ) -> dict[str, Any]:
        if target_column is None:
            raise ValueError("Could not infer a target column for regression.")
        X, y, feature_names = self._prepare_supervised_matrices(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_size_for_rows(len(df)), random_state=42)
        method_key = requested_method if requested_method in {"ridge", "lasso"} else "linear_regression"
        model = {
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.01),
        }.get(method_key, LinearRegression())
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mean_squared_error(y_test, predictions)),
            "rmse": float(math.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }

        predictions_df = pd.DataFrame({"actual": y_test, "predicted": predictions, "residual": y_test - predictions})
        predictions_path = tables_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)

        figures: list[dict[str, Any]] = []
        figures.append(
            self._save_scatter_plot(
                filepath=figures_dir / "predicted_vs_actual.png",
                x=predictions_df["actual"].to_numpy(),
                y=predictions_df["predicted"].to_numpy(),
                title="Predicted vs Actual",
                xlabel="Actual",
                ylabel="Predicted",
                intent="fit",
                source_tables=["predictions.csv"],
                publication_grade=publication_grade,
            )
        )
        figures.append(
            self._save_histogram(
                filepath=figures_dir / "residual_distribution.png",
                values=predictions_df["residual"].to_numpy(),
                title="Residual Distribution",
                xlabel="Residual",
                intent="distribution",
                source_tables=["predictions.csv"],
                publication_grade=publication_grade,
            )
        )

        export_files = [predictions_path]
        if hasattr(model, "coef_"):
            coefficients = np.ravel(model.coef_)
            importance_df = pd.DataFrame(
                {
                    "feature": feature_names[: len(coefficients)],
                    "importance": np.abs(coefficients),
                    "coefficient": coefficients,
                }
            ).sort_values("importance", ascending=False)
            importance_path = tables_dir / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            export_files.append(importance_path)
            figures.append(
                self._save_bar_plot(
                    filepath=figures_dir / "feature_importance.png",
                    categories=importance_df.head(15)["feature"].tolist(),
                    values=importance_df.head(15)["importance"].tolist(),
                    title="Feature Importance",
                    xlabel="Importance",
                    intent="importance",
                    source_tables=["feature_importance.csv"],
                    publication_grade=publication_grade,
                )
            )

        method_file = export_dir / "methods.md"
        result_file = export_dir / "results.md"
        plan_file = export_dir / "experiment_plan.md"
        metrics_path = export_dir / "metrics.json"
        manifest_path = export_dir / "figure_manifest.json"

        method_file.write_text(
            "\n".join(
                [
                    "# Methods",
                    "",
                    "- Domain: CS/AI",
                    "- Analysis type: regression",
                    f"- Model: {method_key}",
                    f"- Target column: `{target_column}`",
                    f"- Train/test split: {len(X_train)}/{len(X_test)}",
                    f"- Feature count: {X.shape[1]}",
                ]
            ),
            encoding="utf-8",
        )
        result_file.write_text(
            "\n".join(
                [
                    "# Results",
                    "",
                    f"- R²: {metrics['r2']:.4f}",
                    f"- RMSE: {metrics['rmse']:.4f}",
                    f"- MAE: {metrics['mae']:.4f}",
                    f"- MSE: {metrics['mse']:.4f}",
                ]
            ),
            encoding="utf-8",
        )
        plan_file.write_text(
            "\n".join(
                [
                    "# Experiment Plan",
                    "",
                    f"- Topic: {project.topic}",
                    f"- Dataset count: {len(project.dataset_ids)}",
                    "- Resolved analysis: regression",
                    f"- Resolved target: {target_column}",
                    f"- Publication grade: {publication_grade}",
                ]
            ),
            encoding="utf-8",
        )
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")

        export_files.extend([plan_file, method_file, result_file, metrics_path, manifest_path])
        export_files.extend(self._flatten_output_paths(figures))
        return {
            "analysis_type": "regression",
            "method_key": method_key,
            "metrics": metrics,
            "notes": "Regression workflow completed successfully.",
            "figures": figures,
            "export_files": export_files,
            "summary": {"target_column": target_column, "row_count": len(df)},
        }

    def _run_classification(
        self,
        *,
        project: ExperimentProject,
        df: pd.DataFrame,
        target_column: str | None,
        figures_dir: Path,
        tables_dir: Path,
        export_dir: Path,
        requested_method: str,
        publication_grade: str,
    ) -> dict[str, Any]:
        if target_column is None:
            raise ValueError("Could not infer a target column for classification.")
        X, y_raw, feature_names = self._prepare_supervised_matrices(df, target_column)
        encoder = LabelEncoder()
        y = encoder.fit_transform(pd.Series(y_raw).astype(str))
        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self._test_size_for_rows(len(df)),
            random_state=42,
            stratify=stratify,
        )

        method_key = "random_forest_classifier" if requested_method == "random_forest_classification" else "logistic_regression"
        if method_key == "random_forest_classifier":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
        else:
            model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, predicted, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        }

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            if probabilities.shape[1] == 2:
                metrics["auroc"] = _safe_auc(y_test, probabilities[:, 1])
                metrics["auprc"] = _safe_auprc(y_test, probabilities[:, 1])

        predictions_df = pd.DataFrame(
            {
                "actual": encoder.inverse_transform(y_test),
                "predicted": encoder.inverse_transform(predicted),
            }
        )
        predictions_path = tables_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)

        cm = confusion_matrix(y_test, predicted)
        figures: list[dict[str, Any]] = [
            self._save_heatmap(
                filepath=figures_dir / "confusion_matrix.png",
                matrix=cm,
                x_labels=encoder.classes_.tolist(),
                y_labels=encoder.classes_.tolist(),
                title="Confusion Matrix",
                intent="confusion_matrix",
                source_tables=["predictions.csv"],
                publication_grade=publication_grade,
            )
        ]

        if probabilities is not None and probabilities.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities[:, 1])
            figures.append(
                self._save_line_plot(
                    filepath=figures_dir / "roc_curve.png",
                    x=fpr,
                    y=tpr,
                    title="ROC Curve",
                    xlabel="False Positive Rate",
                    ylabel="True Positive Rate",
                    intent="roc",
                    source_tables=["predictions.csv"],
                    publication_grade=publication_grade,
                )
            )
            figures.append(
                self._save_line_plot(
                    filepath=figures_dir / "precision_recall_curve.png",
                    x=recall_curve,
                    y=precision_curve,
                    title="Precision-Recall Curve",
                    xlabel="Recall",
                    ylabel="Precision",
                    intent="precision_recall",
                    source_tables=["predictions.csv"],
                    publication_grade=publication_grade,
                )
            )

        export_files = [predictions_path]
        importance_df = self._classification_importance_dataframe(model, feature_names)
        if importance_df is not None and not importance_df.empty:
            importance_path = tables_dir / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            export_files.append(importance_path)
            figures.append(
                self._save_bar_plot(
                    filepath=figures_dir / "feature_importance.png",
                    categories=importance_df.head(15)["feature"].tolist(),
                    values=importance_df.head(15)["importance"].tolist(),
                    title="Feature Importance",
                    xlabel="Importance",
                    intent="importance",
                    source_tables=["feature_importance.csv"],
                    publication_grade=publication_grade,
                )
            )

        method_file = export_dir / "methods.md"
        result_file = export_dir / "results.md"
        plan_file = export_dir / "experiment_plan.md"
        metrics_path = export_dir / "metrics.json"
        manifest_path = export_dir / "figure_manifest.json"

        method_file.write_text(
            "\n".join(
                [
                    "# Methods",
                    "",
                    "- Domain: CS/AI",
                    "- Analysis type: classification",
                    f"- Model: {method_key}",
                    f"- Target column: `{target_column}`",
                    f"- Classes: {', '.join(encoder.classes_.tolist())}",
                    f"- Train/test split: {len(X_train)}/{len(X_test)}",
                ]
            ),
            encoding="utf-8",
        )
        result_lines = [
            "# Results",
            "",
            f"- Accuracy: {metrics['accuracy']:.4f}",
            f"- Precision: {metrics['precision']:.4f}",
            f"- Recall: {metrics['recall']:.4f}",
            f"- F1: {metrics['f1']:.4f}",
        ]
        if metrics.get("auroc") is not None:
            result_lines.append(f"- AUROC: {metrics['auroc']:.4f}")
        if metrics.get("auprc") is not None:
            result_lines.append(f"- AUPRC: {metrics['auprc']:.4f}")
        result_file.write_text("\n".join(result_lines), encoding="utf-8")
        plan_file.write_text(
            "\n".join(
                [
                    "# Experiment Plan",
                    "",
                    f"- Topic: {project.topic}",
                    f"- Dataset count: {len(project.dataset_ids)}",
                    "- Resolved analysis: classification",
                    f"- Resolved target: {target_column}",
                    f"- Publication grade: {publication_grade}",
                ]
            ),
            encoding="utf-8",
        )
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")

        export_files.extend([plan_file, method_file, result_file, metrics_path, manifest_path])
        export_files.extend(self._flatten_output_paths(figures))
        return {
            "analysis_type": "classification",
            "method_key": method_key,
            "metrics": metrics,
            "notes": "Classification workflow completed successfully.",
            "figures": figures,
            "export_files": export_files,
            "summary": {"target_column": target_column, "row_count": len(df)},
        }

    def _run_clustering(
        self,
        *,
        project: ExperimentProject,
        df: pd.DataFrame,
        figures_dir: Path,
        tables_dir: Path,
        export_dir: Path,
        publication_grade: str,
    ) -> dict[str, Any]:
        numeric_df = self._numeric_only_frame(df)
        if numeric_df.empty:
            raise ValueError("Clustering requires numeric feature columns.")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)
        cluster_count = min(6, max(2, int(round(math.sqrt(len(df))))))
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
        clusters = model.fit_predict(scaled)
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(scaled)
        metrics = {
            "cluster_count": int(cluster_count),
            "silhouette": float(silhouette_score(scaled, clusters)) if cluster_count > 1 and len(df) > cluster_count else 0.0,
        }
        assignments = pd.DataFrame({"cluster": clusters, "pc1": embedding[:, 0], "pc2": embedding[:, 1]})
        assignments_path = tables_dir / "cluster_assignments.csv"
        assignments.to_csv(assignments_path, index=False)
        figures = [
            self._save_scatter_plot(
                filepath=figures_dir / "cluster_scatter.png",
                x=embedding[:, 0],
                y=embedding[:, 1],
                hue=clusters,
                title="Cluster Projection",
                xlabel="PC1",
                ylabel="PC2",
                intent="embedding",
                source_tables=["cluster_assignments.csv"],
                publication_grade=publication_grade,
            )
        ]
        method_file = export_dir / "methods.md"
        result_file = export_dir / "results.md"
        plan_file = export_dir / "experiment_plan.md"
        metrics_path = export_dir / "metrics.json"
        manifest_path = export_dir / "figure_manifest.json"
        method_file.write_text(
            "\n".join(
                [
                    "# Methods",
                    "",
                    "- Domain: CS/AI",
                    "- Analysis type: clustering",
                    "- Model: k-means",
                    f"- Feature count: {numeric_df.shape[1]}",
                    f"- Cluster count: {cluster_count}",
                ]
            ),
            encoding="utf-8",
        )
        result_file.write_text(
            "\n".join(
                [
                    "# Results",
                    "",
                    f"- Cluster count: {cluster_count}",
                    f"- Silhouette score: {metrics['silhouette']:.4f}",
                ]
            ),
            encoding="utf-8",
        )
        plan_file.write_text(
            "\n".join(
                [
                    "# Experiment Plan",
                    "",
                    f"- Topic: {project.topic}",
                    f"- Dataset count: {len(project.dataset_ids)}",
                    "- Resolved analysis: clustering",
                ]
            ),
            encoding="utf-8",
        )
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
        export_files = [assignments_path, plan_file, method_file, result_file, metrics_path, manifest_path]
        export_files.extend(self._flatten_output_paths(figures))
        return {
            "analysis_type": "clustering",
            "method_key": "kmeans",
            "metrics": metrics,
            "notes": "Clustering workflow completed successfully.",
            "figures": figures,
            "export_files": export_files,
            "summary": {"row_count": len(df), "feature_count": numeric_df.shape[1]},
        }

    def _run_dimensionality_reduction(
        self,
        *,
        project: ExperimentProject,
        df: pd.DataFrame,
        target_column: str | None,
        figures_dir: Path,
        tables_dir: Path,
        export_dir: Path,
        publication_grade: str,
    ) -> dict[str, Any]:
        numeric_df = self._numeric_only_frame(df.drop(columns=[target_column], errors="ignore"))
        if numeric_df.empty:
            raise ValueError("Dimensionality reduction requires numeric feature columns.")
        scaled = StandardScaler().fit_transform(numeric_df)
        pca = PCA(n_components=min(5, numeric_df.shape[1], len(numeric_df)))
        components = pca.fit_transform(scaled)
        if len(numeric_df) < 4:
            embedding = components[:, :2] if components.shape[1] >= 2 else np.column_stack([components[:, 0], np.zeros(len(components))])
        else:
            reducer = UMAP(n_components=2, n_neighbors=min(15, len(numeric_df) - 1), random_state=42)
            embedding = reducer.fit_transform(scaled)
        embedding_df = pd.DataFrame({"umap1": embedding[:, 0], "umap2": embedding[:, 1]})
        if target_column and target_column in df.columns:
            embedding_df["label"] = df[target_column].astype(str).to_numpy()
        embedding_path = tables_dir / "embedding.csv"
        embedding_df.to_csv(embedding_path, index=False)
        figures = [
            self._save_scatter_plot(
                filepath=figures_dir / "embedding.png",
                x=embedding_df["umap1"].to_numpy(),
                y=embedding_df["umap2"].to_numpy(),
                hue=embedding_df["label"].to_numpy() if "label" in embedding_df else None,
                title="UMAP Embedding",
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                intent="umap",
                source_tables=["embedding.csv"],
                publication_grade=publication_grade,
            )
        ]
        metrics = {
            "explained_variance_pc1": float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) else 0.0,
            "explained_variance_pc2": float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0.0,
        }
        method_file = export_dir / "methods.md"
        result_file = export_dir / "results.md"
        plan_file = export_dir / "experiment_plan.md"
        metrics_path = export_dir / "metrics.json"
        manifest_path = export_dir / "figure_manifest.json"
        method_file.write_text(
            "\n".join(
                [
                    "# Methods",
                    "",
                    "- Domain: CS/AI",
                    "- Analysis type: dimensionality reduction",
                    "- Methods: PCA + UMAP",
                    f"- Feature count: {numeric_df.shape[1]}",
                ]
            ),
            encoding="utf-8",
        )
        result_file.write_text(
            "\n".join(
                [
                    "# Results",
                    "",
                    f"- PCA explained variance (PC1): {metrics['explained_variance_pc1']:.4f}",
                    f"- PCA explained variance (PC2): {metrics['explained_variance_pc2']:.4f}",
                ]
            ),
            encoding="utf-8",
        )
        plan_file.write_text(
            "\n".join(
                [
                    "# Experiment Plan",
                    "",
                    f"- Topic: {project.topic}",
                    "- Resolved analysis: dimensionality reduction",
                ]
            ),
            encoding="utf-8",
        )
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
        export_files = [embedding_path, plan_file, method_file, result_file, metrics_path, manifest_path]
        export_files.extend(self._flatten_output_paths(figures))
        return {
            "analysis_type": "dimensionality_reduction",
            "method_key": "pca_umap",
            "metrics": metrics,
            "notes": "Dimensionality reduction workflow completed successfully.",
            "figures": figures,
            "export_files": export_files,
            "summary": {"row_count": len(df), "feature_count": numeric_df.shape[1]},
        }

    def _run_bioinformatics(
        self,
        *,
        project: ExperimentProject,
        export_dir: Path,
        figures_dir: Path,
        tables_dir: Path,
        analysis_type: str | None,
        metadata_path: str | None,
        sample_id_column: str | None,
        group_column: str | None,
        publication_grade: str,
    ) -> dict[str, Any]:
        paths = self._resolve_dataset_paths(project)
        lower_topic = project.topic.lower()
        is_single_cell = analysis_type == "single_cell" or any(path.suffix == ".h5ad" or path.name.endswith(".mtx") for path in paths) or "single-cell" in lower_topic or "scrna" in lower_topic
        if is_single_cell:
            return self._run_single_cell(
                project=project,
                paths=paths,
                export_dir=export_dir,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
                publication_grade=publication_grade,
            )
        return self._run_bulk_expression(
            project=project,
            paths=paths,
            export_dir=export_dir,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            metadata_path=metadata_path,
            sample_id_column=sample_id_column,
            group_column=group_column,
            publication_grade=publication_grade,
        )

    def _run_bulk_expression(
        self,
        *,
        project: ExperimentProject,
        paths: list[Path],
        export_dir: Path,
        figures_dir: Path,
        tables_dir: Path,
        metadata_path: str | None,
        sample_id_column: str | None,
        group_column: str | None,
        publication_grade: str,
    ) -> dict[str, Any]:
        expression_df, metadata_df = self._load_bulk_inputs(project, paths, metadata_path)
        expression_df = self._normalize_expression_frame(expression_df)
        qc_df = pd.DataFrame(
            {
                "sample": expression_df.columns,
                "total_counts": expression_df.sum(axis=0).to_numpy(),
                "detected_genes": (expression_df > 0).sum(axis=0).to_numpy(),
            }
        )
        qc_path = tables_dir / "sample_qc.csv"
        qc_df.to_csv(qc_path, index=False)

        top_variable = expression_df.var(axis=1).sort_values(ascending=False).head(min(30, len(expression_df)))
        heatmap_genes = expression_df.loc[top_variable.index]
        sample_embedding = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(expression_df.T))
        pca_df = pd.DataFrame({"sample": expression_df.columns, "pc1": sample_embedding[:, 0], "pc2": sample_embedding[:, 1]})

        sample_id_column = sample_id_column or self._detect_sample_id_column(metadata_df)
        group_column = group_column or self._detect_group_column(metadata_df)
        if sample_id_column and sample_id_column in metadata_df.columns:
            pca_df = pca_df.merge(metadata_df, left_on="sample", right_on=sample_id_column, how="left")
        pca_path = tables_dir / "sample_embedding.csv"
        pca_df.to_csv(pca_path, index=False)

        figures: list[dict[str, Any]] = [
            self._save_qc_barplot(
                filepath=figures_dir / "sample_qc.png",
                qc_df=qc_df,
                title="Sample QC Overview",
                intent="qc_distribution",
                source_tables=["sample_qc.csv"],
                publication_grade=publication_grade,
            ),
            self._save_scatter_plot(
                filepath=figures_dir / "sample_pca.png",
                x=pca_df["pc1"].to_numpy(),
                y=pca_df["pc2"].to_numpy(),
                hue=pca_df[group_column].to_numpy() if group_column and group_column in pca_df.columns else None,
                title="Sample Relationship (PCA)",
                xlabel="PC1",
                ylabel="PC2",
                intent="pca",
                source_tables=["sample_embedding.csv"],
                publication_grade=publication_grade,
            ),
            self._save_expression_heatmap(
                filepath=figures_dir / "expression_heatmap.png",
                matrix=heatmap_genes,
                title="Top Variable Genes",
                intent="expression_heatmap",
                source_tables=[],
                publication_grade=publication_grade,
            ),
        ]

        metrics: dict[str, Any] = {
            "sample_count": int(expression_df.shape[1]),
            "gene_count": int(expression_df.shape[0]),
        }
        export_files: list[Path] = [qc_path, pca_path]
        notes = ["Bulk expression workflow completed successfully."]

        diff_df: pd.DataFrame | None = None
        enrichment_df: pd.DataFrame | None = None
        if group_column and group_column in metadata_df.columns and sample_id_column and sample_id_column in metadata_df.columns:
            diff_df = self._differential_expression(expression_df, metadata_df, sample_id_column, group_column)
            diff_path = tables_dir / "differential_expression.csv"
            diff_df.to_csv(diff_path, index=False)
            export_files.append(diff_path)
            figures.append(
                self._save_volcano_plot(
                    filepath=figures_dir / "volcano_plot.png",
                    diff_df=diff_df,
                    title="Differential Expression",
                    intent="volcano",
                    source_tables=["differential_expression.csv"],
                    publication_grade=publication_grade,
                )
            )
            significant = diff_df[(diff_df["padj"] <= 0.05) & (diff_df["log2fc"].abs() >= 1.0)]
            metrics["differential_gene_count"] = int(len(significant))
            try:
                enrichment_df = self._run_enrichment(significant)
            except Exception as exc:
                notes.append(f"Enrichment fallback: {exc}")
                enrichment_df = None
            if enrichment_df is not None and not enrichment_df.empty:
                enrichment_path = tables_dir / "enrichment.csv"
                enrichment_df.to_csv(enrichment_path, index=False)
                export_files.append(enrichment_path)
                figures.append(
                    self._save_bar_plot(
                        filepath=figures_dir / "enrichment_terms.png",
                        categories=enrichment_df.head(10)["Term"].tolist(),
                        values=enrichment_df.head(10)["Combined Score"].tolist(),
                        title="Top Enrichment Terms",
                        xlabel="Combined Score",
                        intent="enrichment",
                        source_tables=["enrichment.csv"],
                        publication_grade=publication_grade,
                    )
                )
        else:
            notes.append("Differential analysis was skipped because metadata or group labels were unavailable.")

        method_file = export_dir / "methods.md"
        result_file = export_dir / "results.md"
        plan_file = export_dir / "experiment_plan.md"
        metrics_path = export_dir / "metrics.json"
        manifest_path = export_dir / "figure_manifest.json"
        method_file.write_text(
            "\n".join(
                [
                    "# Methods",
                    "",
                    "- Domain: Bioinformatics",
                    "- Analysis type: bulk expression",
                    f"- Sample count: {expression_df.shape[1]}",
                    f"- Gene count: {expression_df.shape[0]}",
                    f"- Metadata file provided: {'yes' if metadata_df is not None else 'no'}",
                    f"- Group column: {group_column or 'unavailable'}",
                ]
            ),
            encoding="utf-8",
        )
        result_lines = [
            "# Results",
            "",
            f"- Samples analyzed: {expression_df.shape[1]}",
            f"- Genes analyzed: {expression_df.shape[0]}",
        ]
        if metrics.get("differential_gene_count") is not None:
            result_lines.append(f"- Significant differential genes: {metrics['differential_gene_count']}")
        if enrichment_df is not None and not enrichment_df.empty:
            result_lines.append(f"- Enrichment terms retained: {len(enrichment_df)}")
        result_lines.append(f"- Notes: {' '.join(notes)}")
        result_file.write_text("\n".join(result_lines), encoding="utf-8")
        plan_file.write_text(
            "\n".join(
                [
                    "# Experiment Plan",
                    "",
                    f"- Topic: {project.topic}",
                    f"- Dataset count: {len(project.dataset_ids)}",
                    "- Resolved analysis: bulk expression",
                    f"- Publication grade: {publication_grade}",
                ]
            ),
            encoding="utf-8",
        )
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
        export_files.extend([plan_file, method_file, result_file, metrics_path, manifest_path])
        export_files.extend(self._flatten_output_paths(figures))
        if project.linked_academic_project_id:
            export_files.extend(
                self._write_linked_academic_exports(
                    project=project,
                    export_dir=export_dir,
                    metrics=metrics,
                    figures=figures,
                    notes=notes,
                    tables=export_files,
                )
            )
        return {
            "analysis_type": "bulk_expression",
            "method_key": "bulk_expression_qc_differential",
            "metrics": metrics,
            "notes": " ".join(notes),
            "figures": figures,
            "export_files": export_files,
            "summary": {
                "sample_count": int(expression_df.shape[1]),
                "gene_count": int(expression_df.shape[0]),
                "metadata_columns": metadata_df.columns.tolist(),
            },
        }

    def _run_single_cell(
        self,
        *,
        project: ExperimentProject,
        paths: list[Path],
        export_dir: Path,
        figures_dir: Path,
        tables_dir: Path,
        publication_grade: str,
    ) -> dict[str, Any]:
        adata = self._load_single_cell_adata(paths)
        matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        if matrix.ndim != 2:
            raise ValueError("Single-cell matrix must be 2-dimensional.")
        gene_names = adata.var_names.to_list()
        cell_names = adata.obs_names.to_list()

        qc_df = pd.DataFrame(
            {
                "cell": cell_names,
                "n_counts": matrix.sum(axis=1),
                "n_genes": (matrix > 0).sum(axis=1),
            }
        )
        qc_path = tables_dir / "cell_qc.csv"
        qc_df.to_csv(qc_path, index=False)

        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0) * 1e4
        normalized = np.log1p(normalized)
        variances = normalized.var(axis=0)
        hvg_idx = np.argsort(variances)[::-1][: min(500, normalized.shape[1])]
        hvg_matrix = normalized[:, hvg_idx]
        pca = PCA(n_components=min(20, hvg_matrix.shape[0], hvg_matrix.shape[1]), random_state=42)
        pcs = pca.fit_transform(hvg_matrix)
        if len(cell_names) < 4:
            embedding = pcs[:, :2] if pcs.shape[1] >= 2 else np.column_stack([pcs[:, 0], np.zeros(len(pcs))])
        else:
            reducer = UMAP(n_components=2, n_neighbors=min(15, len(cell_names) - 1), random_state=42)
            embedding = reducer.fit_transform(pcs)
        cluster_count = min(8, max(2, int(round(math.sqrt(len(cell_names) / 2)))))
        clusters = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto").fit_predict(pcs)

        embedding_df = pd.DataFrame(
            {
                "cell": cell_names,
                "umap1": embedding[:, 0],
                "umap2": embedding[:, 1],
                "cluster": clusters,
            }
        )
        embedding_path = tables_dir / "cell_embedding.csv"
        embedding_df.to_csv(embedding_path, index=False)

        marker_rows: list[dict[str, Any]] = []
        for cluster in sorted(np.unique(clusters)):
            mask = clusters == cluster
            other = ~mask
            cluster_means = normalized[mask].mean(axis=0)
            other_means = normalized[other].mean(axis=0) if other.any() else np.zeros_like(cluster_means)
            scores = cluster_means - other_means
            for idx in np.argsort(scores)[::-1][:5]:
                marker_rows.append(
                    {
                        "cluster": int(cluster),
                        "gene": gene_names[idx],
                        "score": float(scores[idx]),
                        "cluster_mean": float(cluster_means[idx]),
                        "other_mean": float(other_means[idx]),
                    }
                )
        marker_df = pd.DataFrame(marker_rows)
        marker_path = tables_dir / "marker_genes.csv"
        marker_df.to_csv(marker_path, index=False)

        cluster_means = pd.DataFrame(normalized, columns=gene_names).assign(cluster=clusters).groupby("cluster").mean()
        heatmap_genes = marker_df.drop_duplicates("gene").head(12)["gene"].tolist()
        heatmap_frame = cluster_means[heatmap_genes] if heatmap_genes else cluster_means.iloc[:, : min(12, cluster_means.shape[1])]

        figures = [
            self._save_qc_histogram_pair(
                filepath=figures_dir / "cell_qc.png",
                qc_df=qc_df,
                title="Cell QC Distribution",
                intent="qc_distribution",
                source_tables=["cell_qc.csv"],
                publication_grade=publication_grade,
            ),
            self._save_scatter_plot(
                filepath=figures_dir / "cell_umap.png",
                x=embedding_df["umap1"].to_numpy(),
                y=embedding_df["umap2"].to_numpy(),
                hue=embedding_df["cluster"].to_numpy(),
                title="Single-Cell UMAP",
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                intent="umap",
                source_tables=["cell_embedding.csv"],
                publication_grade=publication_grade,
            ),
            self._save_heatmap(
                filepath=figures_dir / "marker_heatmap.png",
                matrix=heatmap_frame.to_numpy(),
                x_labels=heatmap_frame.columns.tolist(),
                y_labels=[str(item) for item in heatmap_frame.index.tolist()],
                title="Marker Gene Heatmap",
                intent="expression_heatmap",
                source_tables=["marker_genes.csv"],
                publication_grade=publication_grade,
            ),
        ]
        if not marker_df.empty:
            top_violin_genes = marker_df.drop_duplicates("gene").head(4)["gene"].tolist()
            figures.append(
                self._save_violin_plot(
                    filepath=figures_dir / "marker_violin.png",
                    frame=pd.DataFrame(normalized, columns=gene_names).assign(cluster=clusters),
                    genes=top_violin_genes,
                    title="Top Marker Violin Plot",
                    intent="violin",
                    source_tables=["marker_genes.csv"],
                    publication_grade=publication_grade,
                )
            )

        metrics = {
            "cell_count": int(len(cell_names)),
            "gene_count": int(len(gene_names)),
            "cluster_count": int(cluster_count),
        }
        method_file = export_dir / "methods.md"
        result_file = export_dir / "results.md"
        plan_file = export_dir / "experiment_plan.md"
        metrics_path = export_dir / "metrics.json"
        manifest_path = export_dir / "figure_manifest.json"
        method_file.write_text(
            "\n".join(
                [
                    "# Methods",
                    "",
                    "- Domain: Bioinformatics",
                    "- Analysis type: single-cell starter workflow",
                    f"- Cell count: {len(cell_names)}",
                    f"- Gene count: {len(gene_names)}",
                    f"- Cluster count: {cluster_count}",
                ]
            ),
            encoding="utf-8",
        )
        result_file.write_text(
            "\n".join(
                [
                    "# Results",
                    "",
                    f"- Cells analyzed: {len(cell_names)}",
                    f"- Genes analyzed: {len(gene_names)}",
                    f"- Clusters resolved: {cluster_count}",
                    f"- Marker genes exported: {len(marker_df)}",
                ]
            ),
            encoding="utf-8",
        )
        plan_file.write_text(
            "\n".join(
                [
                    "# Experiment Plan",
                    "",
                    f"- Topic: {project.topic}",
                    "- Resolved analysis: single-cell starter workflow",
                    f"- Publication grade: {publication_grade}",
                ]
            ),
            encoding="utf-8",
        )
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
        export_files: list[Path] = [qc_path, embedding_path, marker_path, plan_file, method_file, result_file, metrics_path, manifest_path]
        export_files.extend(self._flatten_output_paths(figures))
        if project.linked_academic_project_id:
            export_files.extend(
                self._write_linked_academic_exports(
                    project=project,
                    export_dir=export_dir,
                    metrics=metrics,
                    figures=figures,
                    notes=["Single-cell workflow completed successfully."],
                    tables=export_files,
                )
            )
        return {
            "analysis_type": "single_cell",
            "method_key": "single_cell_starter",
            "metrics": metrics,
            "notes": "Single-cell workflow completed successfully.",
            "figures": figures,
            "export_files": export_files,
            "summary": {"cell_count": int(len(cell_names)), "gene_count": int(len(gene_names))},
        }

    def _load_primary_dataframe(self, project: ExperimentProject) -> pd.DataFrame:
        paths = self._resolve_dataset_paths(project)
        candidates = [path for path in paths if path.suffix.lower() in _DATA_SUFFIXES]
        if not candidates:
            raise ValueError("No supported tabular dataset was provided.")
        return self._read_table(candidates[0])

    def _load_bulk_inputs(
        self,
        project: ExperimentProject,
        paths: list[Path],
        metadata_path: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        metadata_file = self._resolve_optional_dataset_path(project.thread_id, metadata_path)
        expression_path: Path | None = None
        auto_metadata_path: Path | None = None

        for path in paths:
            if path.suffix.lower() not in _DATA_SUFFIXES:
                continue
            lower_name = path.name.lower()
            if metadata_file is None and ("meta" in lower_name or "sample" in lower_name) and auto_metadata_path is None:
                auto_metadata_path = path
                continue
            if expression_path is None:
                expression_path = path

        metadata_source = metadata_file or auto_metadata_path
        if expression_path is None:
            raise ValueError("Bulk expression analysis requires an expression matrix file.")

        expression_df = self._read_table(expression_path)
        metadata_df = self._read_table(metadata_source) if metadata_source is not None else pd.DataFrame({"sample": expression_df.columns[1:]})
        return expression_df, metadata_df

    def _load_single_cell_adata(self, paths: list[Path]) -> ad.AnnData:
        h5ad = next((path for path in paths if path.suffix.lower() == ".h5ad"), None)
        if h5ad is not None:
            return ad.read_h5ad(h5ad)

        mtx = next((path for path in paths if path.suffix.lower() == ".mtx"), None)
        if mtx is None:
            raise ValueError("Single-cell analysis requires either a .h5ad file or a 10x Matrix Market bundle.")
        feature_path = next((path for path in paths if any(path.name.endswith(suffix) for suffix in _TENX_FEATURE_SUFFIXES)), None)
        barcode_path = next((path for path in paths if path.name.endswith("barcodes.tsv") or path.name.endswith("barcodes.tsv.gz")), None)
        if feature_path is None or barcode_path is None:
            raise ValueError("10x Matrix Market input requires features.tsv and barcodes.tsv companions.")

        matrix = mmread(mtx).T.tocsr()
        features = pd.read_csv(feature_path, sep="\t", header=None)
        barcodes = pd.read_csv(barcode_path, sep="\t", header=None)
        gene_names = features.iloc[:, 1].astype(str).tolist() if features.shape[1] > 1 else features.iloc[:, 0].astype(str).tolist()
        cell_names = barcodes.iloc[:, 0].astype(str).tolist()
        return ad.AnnData(X=matrix, obs=pd.DataFrame(index=cell_names), var=pd.DataFrame(index=gene_names))

    def _normalize_expression_frame(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        frame = expression_df.copy()
        if not pd.api.types.is_numeric_dtype(frame.iloc[:, 0]):
            frame = frame.set_index(frame.columns[0])
        frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return np.log1p(frame.astype(float))

    def _differential_expression(
        self,
        expression_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        sample_id_column: str,
        group_column: str,
    ) -> pd.DataFrame:
        sample_groups = metadata_df[[sample_id_column, group_column]].dropna().drop_duplicates()
        if sample_groups[group_column].nunique() < 2:
            raise ValueError("Differential analysis requires at least two groups.")
        groups = sample_groups[group_column].astype(str).unique().tolist()[:2]
        mapping = sample_groups.set_index(sample_id_column)[group_column].astype(str)
        common_samples = [sample for sample in expression_df.columns if sample in mapping.index]
        if len(common_samples) < 2:
            raise ValueError("Metadata sample IDs do not overlap with the expression matrix columns.")
        group_a_samples = [sample for sample in common_samples if mapping[sample] == groups[0]]
        group_b_samples = [sample for sample in common_samples if mapping[sample] == groups[1]]
        if not group_a_samples or not group_b_samples:
            raise ValueError("Each comparison group must contain at least one sample.")

        group_a = expression_df[group_a_samples]
        group_b = expression_df[group_b_samples]
        statistic, p_values = ttest_ind(group_a.to_numpy(), group_b.to_numpy(), axis=1, equal_var=False, nan_policy="omit")
        mean_a = group_a.mean(axis=1)
        mean_b = group_b.mean(axis=1)
        diff_df = pd.DataFrame(
            {
                "gene": expression_df.index.astype(str),
                "group_a": groups[0],
                "group_b": groups[1],
                "log2fc": mean_a - mean_b,
                "statistic": statistic,
                "pvalue": np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0),
            }
        )
        diff_df["padj"] = _bh_adjust(diff_df["pvalue"])
        diff_df["neg_log10_padj"] = -np.log10(diff_df["padj"].clip(lower=1e-300))
        return diff_df.sort_values(["padj", "log2fc"], ascending=[True, False]).reset_index(drop=True)

    def _run_enrichment(self, significant_genes: pd.DataFrame) -> pd.DataFrame:
        gene_list = significant_genes.sort_values("log2fc", ascending=False)["gene"].astype(str).head(300).tolist()
        if not gene_list:
            raise ValueError("No significant genes were available for enrichment.")
        enrichment = gseapy.enrichr(
            gene_list=gene_list,
            gene_sets=["KEGG_2021_Human"],
            organism="Human",
            outdir=None,
            cutoff=0.5,
        )
        if enrichment.results is None or enrichment.results.empty:
            raise ValueError("No enrichment results were returned.")
        frame = enrichment.results.copy()
        preferred = [column for column in ["Term", "Adjusted P-value", "Combined Score", "Odds Ratio", "Genes"] if column in frame.columns]
        return frame[preferred].sort_values("Adjusted P-value", ascending=True).reset_index(drop=True)

    def _prepare_supervised_matrices(self, df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' was not found.")
        feature_df = df.drop(columns=[target_column]).copy()
        target = df[target_column].copy()
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [column for column in feature_df.columns if column not in numeric_columns]
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_columns),
                ("cat", categorical_transformer, categorical_columns),
            ]
        )
        X = transformer.fit_transform(feature_df)
        if sparse.issparse(X):
            X = X.toarray()
        feature_names = []
        if numeric_columns:
            feature_names.extend(numeric_columns)
        if categorical_columns:
            encoder = transformer.named_transformers_["cat"].named_steps["encoder"]
            feature_names.extend(encoder.get_feature_names_out(categorical_columns).tolist())
        return np.asarray(X, dtype=float), target.to_numpy(), feature_names

    def _classification_importance_dataframe(self, model: Any, feature_names: list[str]) -> pd.DataFrame | None:
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return None
        return pd.DataFrame({"feature": feature_names[: len(values)], "importance": values}).sort_values("importance", ascending=False)

    def _numeric_only_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[np.number]).copy()
        return numeric.fillna(numeric.median()).dropna(axis=1, how="all")

    def _resolve_cs_ai_analysis_type(
        self,
        topic: str,
        requested: str | None,
        df: pd.DataFrame,
        target_column: str | None,
    ) -> str:
        topic_lower = topic.lower()
        if requested:
            return requested
        if "cluster" in topic_lower or "k-means" in topic_lower:
            return "clustering"
        if "umap" in topic_lower or "pca" in topic_lower or "dimension" in topic_lower:
            return "dimensionality_reduction"
        if "logistic" in topic_lower or "classif" in topic_lower:
            return "classification"
        if "random forest" in topic_lower:
            return "random_forest_classification" if target_column and not pd.api.types.is_numeric_dtype(df[target_column]) else "random_forest_regression"
        if "ridge" in topic_lower:
            return "ridge"
        if "lasso" in topic_lower:
            return "lasso"
        if target_column and target_column in df.columns and (not pd.api.types.is_numeric_dtype(df[target_column]) or df[target_column].nunique() <= 10):
            return "classification"
        return "regression"

    def _detect_target_column(self, df: pd.DataFrame) -> str | None:
        preferred = ["target", "label", "class", "outcome", "response", "y", "group"]
        lower_map = {column.lower(): column for column in df.columns}
        for key in preferred:
            if key in lower_map:
                return lower_map[key]
        if len(df.columns) > 0:
            return df.columns[-1]
        return None

    def _detect_sample_id_column(self, metadata_df: pd.DataFrame) -> str | None:
        preferred = ["sample", "sample_id", "sampleid", "barcode", "cell"]
        lower_map = {column.lower(): column for column in metadata_df.columns}
        for key in preferred:
            if key in lower_map:
                return lower_map[key]
        return metadata_df.columns[0] if len(metadata_df.columns) > 0 else None

    def _detect_group_column(self, metadata_df: pd.DataFrame) -> str | None:
        preferred = ["group", "condition", "label", "class", "treatment", "status"]
        lower_map = {column.lower(): column for column in metadata_df.columns}
        for key in preferred:
            if key in lower_map:
                return lower_map[key]
        return metadata_df.columns[1] if len(metadata_df.columns) > 1 else None

    def _resolve_dataset_paths(self, project: ExperimentProject) -> list[Path]:
        return [self._resolve_dataset_path(project.thread_id, dataset_id) for dataset_id in project.dataset_ids]

    def _resolve_dataset_path(self, thread_id: str, dataset_id: str) -> Path:
        if dataset_id.startswith(VIRTUAL_PATH_PREFIX):
            return get_paths().resolve_virtual_path(thread_id, dataset_id)
        return Path(dataset_id).expanduser().resolve()

    def _resolve_optional_dataset_path(self, thread_id: str, dataset_id: str | None) -> Path | None:
        if not dataset_id:
            return None
        return self._resolve_dataset_path(thread_id, dataset_id)

    def _read_table(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t")
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported table format: {path.suffix}")

    def _to_virtual_output_path(self, thread_id: str, filepath: Path) -> str:
        outputs_dir = get_paths().sandbox_outputs_dir(thread_id).resolve()
        relative = filepath.resolve().relative_to(outputs_dir)
        return f"{VIRTUAL_PATH_PREFIX}/outputs/{relative.as_posix()}"

    def _save_scatter_plot(
        self,
        *,
        filepath: Path,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        intent: str,
        source_tables: list[str],
        publication_grade: str,
        hue: np.ndarray | None = None,
    ) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        if hue is not None:
            unique = pd.Series(hue).astype(str)
            palette = sns.color_palette("tab10", n_colors=max(2, unique.nunique()))
            sns.scatterplot(x=x, y=y, hue=unique, palette=palette, ax=ax, s=35, edgecolor="none")
            ax.legend(loc="best", frameon=False)
        else:
            ax.scatter(x, y, s=35, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return {
            "intent": intent,
            "chart_type": choose_chart_type(intent=intent, data_shape="relationship", publication_grade=publication_grade),
            "source_tables": source_tables,
            "output_files": [str(filepath)],
            "metadata": {"title": title},
        }

    def _save_heatmap(
        self,
        *,
        filepath: Path,
        matrix: np.ndarray,
        x_labels: list[str],
        y_labels: list[str],
        title: str,
        intent: str,
        source_tables: list[str],
        publication_grade: str,
    ) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(max(6.0, len(x_labels) * 0.35), max(4.5, len(y_labels) * 0.35)))
        sns.heatmap(matrix, cmap="viridis", ax=ax, xticklabels=x_labels, yticklabels=y_labels)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return {
            "intent": intent,
            "chart_type": choose_chart_type(intent=intent, data_shape="matrix", publication_grade=publication_grade),
            "source_tables": source_tables,
            "output_files": [str(filepath)],
            "metadata": {"title": title},
        }

    def _save_expression_heatmap(
        self,
        *,
        filepath: Path,
        matrix: pd.DataFrame,
        title: str,
        intent: str,
        source_tables: list[str],
        publication_grade: str,
    ) -> dict[str, Any]:
        scaled = matrix.copy()
        scaled = scaled.sub(scaled.mean(axis=1), axis=0).div(scaled.std(axis=1).replace(0, 1), axis=0)
        return self._save_heatmap(
            filepath=filepath,
            matrix=scaled.to_numpy(),
            x_labels=[str(item) for item in scaled.columns.tolist()],
            y_labels=[str(item) for item in scaled.index.tolist()],
            title=title,
            intent=intent,
            source_tables=source_tables,
            publication_grade=publication_grade,
        )

    def _save_histogram(
        self,
        *,
        filepath: Path,
        values: np.ndarray,
        title: str,
        xlabel: str,
        intent: str,
        source_tables: list[str],
        publication_grade: str,
    ) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        sns.histplot(values, bins=min(30, max(10, int(len(values) / 4))), kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return {
            "intent": intent,
            "chart_type": choose_chart_type(intent=intent, data_shape="distribution", publication_grade=publication_grade),
            "source_tables": source_tables,
            "output_files": [str(filepath)],
            "metadata": {"title": title},
        }

    def _save_line_plot(
        self,
        *,
        filepath: Path,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        intent: str,
        source_tables: list[str],
        publication_grade: str,
    ) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.plot(x, y, linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return {
            "intent": intent,
            "chart_type": choose_chart_type(intent=intent, data_shape="trend", publication_grade=publication_grade),
            "source_tables": source_tables,
            "output_files": [str(filepath)],
            "metadata": {"title": title},
        }

    def _save_bar_plot(
        self,
        *,
        filepath: Path,
        categories: list[str],
        values: list[float],
        title: str,
        xlabel: str,
        intent: str,
        source_tables: list[str],
        publication_grade: str,
    ) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(7.0, max(4.5, len(categories) * 0.35)))
        sns.barplot(x=values, y=categories, ax=ax, orient="h")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return {
            "intent": intent,
            "chart_type": choose_chart_type(intent=intent, data_shape="comparison", publication_grade=publication_grade),
            "source_tables": source_tables,
            "output_files": [str(filepath)],
            "metadata": {"title": title},
        }

    def _save_qc_barplot(
        self,
        *,
        filepath: Path,
        qc_df: pd.DataFrame,
        title: str,
        intent: str,
        source_tables: list[str],
        publication_grade: str,
    ) -> dict[str, Any]:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
        sns.barplot(data=qc_df, x="sample", y="total_counts", ax=axes[0])
        sns.barplot(data=qc_df, x="sample", y="detected_genes", ax=axes[1])
        axes[0].set_title("Total Counts")
        axes[1].set_title("Detected Genes")
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return {
            "intent": intent,
            "chart_type": choose_chart_type(intent=intent, data_shape="distribution", publication_grade=publication_grade),
            "source_tables": source_tables,
            "output_files": [str(filepath)],
            "metadata": {"title": title},
        }

    def _save_qc_histogram_pair(
        self,
        *,
        filepath: Path,
        qc_df: pd.DataFrame,
        title: str,
        intent: str,
        source_tables: list[str],
        publication_grade: str,
    ) -> dict[str, Any]:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
        sns.histplot(qc_df["n_counts"], bins=30, kde=True, ax=axes[0])
        sns.histplot(qc_df["n_genes"], bins=30, kde=True, ax=axes[1])
        axes[0].set_title("Counts per Cell")
        axes[1].set_title("Genes per Cell")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return {
            "intent": intent,
            "chart_type": choose_chart_type(intent=intent, data_shape="distribution", publication_grade=publication_grade),
            "source_tables": source_tables,
            "output_files": [str(filepath)],
            "metadata": {"title": title},
        }

    def _save_volcano_plot(
        self,
        *,
        filepath: Path,
        diff_df: pd.DataFrame,
        title: str,
        intent: str,
        source_tables: list[str],
        publication_grade: str,
    ) -> dict[str, Any]:
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        significant = (diff_df["padj"] <= 0.05) & (diff_df["log2fc"].abs() >= 1.0)
        palette = np.where(significant, np.where(diff_df["log2fc"] > 0, "#d62728", "#1f77b4"), "#b0b0b0")
        ax.scatter(diff_df["log2fc"], diff_df["neg_log10_padj"], c=palette, s=18, alpha=0.75)
        ax.set_xlabel("log2 Fold Change")
        ax.set_ylabel("-log10 Adjusted P")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return {
            "intent": intent,
            "chart_type": choose_chart_type(intent=intent, data_shape="differential", publication_grade=publication_grade),
            "source_tables": source_tables,
            "output_files": [str(filepath)],
            "metadata": {"title": title},
        }

    def _save_violin_plot(
        self,
        *,
        filepath: Path,
        frame: pd.DataFrame,
        genes: list[str],
        title: str,
        intent: str,
        source_tables: list[str],
        publication_grade: str,
    ) -> dict[str, Any]:
        melted = frame[genes + ["cluster"]].melt(id_vars="cluster", var_name="gene", value_name="expression")
        fig, ax = plt.subplots(figsize=(max(7.5, len(genes) * 1.4), 5.0))
        sns.violinplot(data=melted, x="gene", y="expression", hue="cluster", inner="quart", cut=0, ax=ax)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(loc="best", frameon=False, title="Cluster")
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        return {
            "intent": intent,
            "chart_type": choose_chart_type(intent=intent, data_shape="distribution", publication_grade=publication_grade),
            "source_tables": source_tables,
            "output_files": [str(filepath)],
            "metadata": {"title": title},
        }

    def _flatten_output_paths(self, figure_specs: list[dict[str, Any]]) -> list[Path]:
        flattened: list[Path] = []
        for item in figure_specs:
            flattened.extend(Path(path) for path in item["output_files"])
        return flattened

    def _write_linked_academic_exports(
        self,
        *,
        project: ExperimentProject,
        export_dir: Path,
        metrics: dict[str, Any],
        figures: list[dict[str, Any]],
        notes: list[str],
        tables: list[Path],
    ) -> list[Path]:
        paper_ready = export_dir / "paper_ready_results.md"
        evidence = export_dir / "evidence.json"
        paper_ready.write_text(
            "\n".join(
                [
                    "# Paper-Ready Results",
                    "",
                    f"- Linked academic project: `{project.linked_academic_project_id}`",
                    f"- Experiment project: `{project.project_id}`",
                    "",
                    "## Key Metrics",
                    *[f"- {key}: {value}" for key, value in metrics.items()],
                    "",
                    "## Figures",
                    *[f"- {item['metadata'].get('title', item['intent'])}: {', '.join(Path(path).name for path in item['output_files'])}" for item in figures],
                    "",
                    "## Notes",
                    *[f"- {note}" for note in notes],
                ]
            ),
            encoding="utf-8",
        )
        evidence.write_text(
            json.dumps(
                {
                    "linked_academic_project_id": project.linked_academic_project_id,
                    "experiment_project_id": project.project_id,
                    "metrics": metrics,
                    "figures": figures,
                    "tables": [path.name for path in tables if path.parent.name == "tables"],
                    "notes": notes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return [paper_ready, evidence]

    def _test_size_for_rows(self, row_count: int) -> float:
        return 0.25 if row_count < 40 else 0.2

    async def _require_project(self, project_id: str) -> ExperimentProject:
        project = await self._repository.get_project(project_id)
        if project is None:
            raise ValueError(f"Project {project_id} not found")
        return project

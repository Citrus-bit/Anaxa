"""Routing helpers for experiment figure selection."""

from __future__ import annotations


def choose_chart_type(*, intent: str, data_shape: str | None = None, publication_grade: str = "paper") -> str:
    normalized_intent = intent.lower().strip()
    normalized_shape = (data_shape or "").lower().strip()

    if normalized_intent in {"roc", "receiver_operating_characteristic"}:
        return "roc"
    if normalized_intent in {"precision_recall", "pr"}:
        return "pr"
    if normalized_intent in {"confusion", "confusion_matrix"}:
        return "heatmap"
    if normalized_intent in {"volcano", "differential"}:
        return "volcano"
    if normalized_intent in {"embedding", "pca", "umap", "scatter"}:
        return "scatter"
    if normalized_intent in {"fit", "residuals", "relationship"}:
        return "scatter"
    if normalized_intent in {"trend", "training_curve", "ablation"}:
        return "line"
    if normalized_intent in {"distribution", "qc_distribution"}:
        return "histogram"
    if normalized_intent in {"importance", "enrichment", "comparison"}:
        return "bar"
    if normalized_intent in {"expression_heatmap", "matrix"}:
        return "heatmap"
    if normalized_intent in {"gene_violin", "violin"}:
        return "violin"
    if normalized_intent in {"dotplot"}:
        return "dotplot"

    if normalized_shape in {"matrix", "gene-by-sample"}:
        return "heatmap"
    if normalized_shape in {"distribution"}:
        return "histogram"
    if normalized_shape in {"comparison"}:
        return "bar"
    if normalized_shape in {"relationship"}:
        return "scatter"
    if normalized_shape in {"trend"}:
        return "line"

    if publication_grade == "paper":
        return "scatter"
    return "bar"

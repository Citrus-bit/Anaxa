from typing import Literal

from langchain.tools import tool


@tool("visual_refinement_check", parse_docstring=True)
def visual_refinement_check_tool(
    output_type: Literal["chart", "ppt", "image"],
    original_request: str,
    current_output_description: str,
    comparison: dict[str, dict],
    overall_score: int,
    refinement_plan: str | None = None,
) -> str:
    """Compare visual output against the original request and decide whether to iterate.

    Use this tool AFTER generating visual output and BEFORE visual_quality_check
    when you suspect the output may not fully match the user's intent. This enables
    structured iterative refinement rather than delivering suboptimal first drafts.

    Workflow:
    1. Generate visual output
    2. Call this tool to compare output vs requirements
    3. If score < 7, follow the refinement_plan and regenerate
    4. Repeat until score >= 7 (max 3 iterations)
    5. Then proceed to visual_quality_check → present_files

    The comparison dict should map each requirement dimension to an assessment:

    For all types:
    - content_accuracy: {"met": true/false, "gap": "description of gap if any"}
    - style_match: {"met": true/false, "gap": "description of gap if any"}
    - color_fidelity: {"met": true/false, "gap": "description of gap if any"}

    Additional for "chart":
    - data_representation: {"met": true/false, "gap": "description of gap if any"}

    Additional for "ppt":
    - slide_consistency: {"met": true/false, "gap": "description of gap if any"}
    - narrative_flow: {"met": true/false, "gap": "description of gap if any"}

    Additional for "image":
    - composition_match: {"met": true/false, "gap": "description of gap if any"}
    - mood_atmosphere: {"met": true/false, "gap": "description of gap if any"}

    Args:
        output_type: Type of visual output being refined.
        original_request: The user's original request or requirement summary.
        current_output_description: Description of what was actually generated.
        comparison: Dict mapping requirement dimensions to {"met": bool, "gap": str} assessments.
        overall_score: Self-assessed score 1-10 of how well output matches request.
        refinement_plan: If score < 7, describe specific changes to make in next iteration.
    """
    required_dims = {"content_accuracy", "style_match", "color_fidelity"}
    type_dims = {
        "chart": {"data_representation"},
        "ppt": {"slide_consistency", "narrative_flow"},
        "image": {"composition_match", "mood_atmosphere"},
    }
    all_required = required_dims | type_dims.get(output_type, set())
    provided = set(comparison.keys())
    missing = all_required - provided

    if missing:
        return f"INCOMPLETE: Missing comparison dimensions: {', '.join(sorted(missing))}. Re-run with all required dimensions."

    gaps = []
    for dim, assessment in comparison.items():
        if isinstance(assessment, dict) and not assessment.get("met", True):
            gap_desc = assessment.get("gap", "unspecified")
            gaps.append(f"{dim}: {gap_desc}")

    score = max(1, min(10, overall_score))

    if score >= 8 and not gaps:
        return f"EXCELLENT ({score}/10): Output closely matches requirements. Proceed to visual_quality_check."

    if score >= 7:
        if gaps:
            return f"GOOD ({score}/10): Minor gaps detected: {'; '.join(gaps)}. Consider a quick fix or proceed to visual_quality_check if acceptable."
        return f"GOOD ({score}/10): Output matches requirements. Proceed to visual_quality_check."

    if not refinement_plan:
        return f"NEEDS_REFINEMENT ({score}/10): Gaps found: {'; '.join(gaps) if gaps else 'overall quality below threshold'}. Provide a refinement_plan describing specific changes, then regenerate."

    return f"REFINING ({score}/10): Gaps: {'; '.join(gaps) if gaps else 'general quality'}. Plan: {refinement_plan}. Execute the refinement plan, regenerate, then re-run this check. Max 3 iterations."

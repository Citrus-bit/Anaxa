from typing import Literal

from langchain.tools import tool


@tool("visual_quality_check", parse_docstring=True)
def visual_quality_check_tool(
    output_type: Literal["chart", "ppt", "image"],
    description: str,
    checklist: dict[str, bool],
    issues_found: list[str] | None = None,
    action_taken: str | None = None,
) -> str:
    """Run a structured quality self-check before delivering visual output to the user.

    You MUST call this tool BEFORE calling present_files for any visual output
    (chart, image, or presentation). This enforces a quality gate that ensures
    professional-grade deliverables.

    Workflow:
    1. Generate the visual output (chart/image/PPT)
    2. Call this tool with your honest self-assessment
    3. If issues_found is non-empty, fix them and regenerate BEFORE presenting
    4. Only call present_files after this tool returns "PASS"

    The checklist keys depend on output_type:

    For "chart":
    - data_integrity: Y-axis not truncated, data accurately represented
    - chart_type_fit: Chart type matches the data story
    - labels_complete: All axes labeled with units, title states the insight
    - color_accessible: Max 5 colors, color-blind safe, sufficient contrast
    - no_chartjunk: No 3D effects, unnecessary gridlines, or decorations

    For "ppt":
    - one_message_per_slide: Each slide has exactly one core idea
    - visual_consistency: All slides share the same style, colors, typography
    - storytelling_arc: Slides follow Hook → Context → Solution → Impact → CTA
    - text_hierarchy: Headlines 48-72pt, body 18-24pt, max 3 bullets per slide
    - negative_space: 40%+ whitespace on each slide

    For "image":
    - prompt_specificity: Prompt was 150+ words with concrete details
    - composition_balanced: Clear focal point, rule of thirds, appropriate framing
    - style_match: Output matches requested style and mood
    - color_harmony: Colors are harmonious and appropriate
    - no_artifacts: No blur, deformation, or quality issues

    Args:
        output_type: Type of visual output being checked (chart, ppt, image).
        description: Brief description of what was generated (e.g., "Q3 revenue bar chart" or "AI product launch keynote, 8 slides").
        checklist: Dict mapping check names to pass/fail booleans. All keys for the output_type must be present.
        issues_found: List of specific issues identified (e.g., "Y-axis starts at 50, should start at 0"). Empty list or None means no issues.
        action_taken: If issues were found and fixed, describe what was done (e.g., "Regenerated with Y-axis starting at 0").
    """
    expected_keys = {
        "chart": {"data_integrity", "chart_type_fit", "labels_complete", "color_accessible", "no_chartjunk"},
        "ppt": {"one_message_per_slide", "visual_consistency", "storytelling_arc", "text_hierarchy", "negative_space"},
        "image": {"prompt_specificity", "composition_balanced", "style_match", "color_harmony", "no_artifacts"},
    }

    required = expected_keys.get(output_type, set())
    provided = set(checklist.keys())
    missing = required - provided
    if missing:
        return f"INCOMPLETE: Missing checklist items: {', '.join(sorted(missing))}. Re-run with all required checks."

    failed = [k for k, v in checklist.items() if not v]
    has_issues = bool(issues_found)

    if failed and not action_taken:
        return f"FAIL: {len(failed)} check(s) failed: {', '.join(failed)}. Issues: {'; '.join(issues_found) if issues_found else 'not specified'}. Fix the issues and regenerate before presenting to user."

    if failed and action_taken:
        return f"FIXED: {len(failed)} issue(s) were identified and addressed. Action: {action_taken}. Verify the fix resolved the issues, then present to user."

    if has_issues and not failed:
        return f"WARNING: All checks passed but issues were noted: {'; '.join(issues_found)}. Consider addressing before delivery, or present with caveats."

    return f"PASS: All quality checks passed for {output_type} ({description}). Safe to present to user."

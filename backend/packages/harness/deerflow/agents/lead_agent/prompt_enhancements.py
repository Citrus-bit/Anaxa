"""
Enhanced prompt sections for improving visual output quality.
Injected into the system prompt when visual-related skills are active.
"""

# Visual skills that trigger prompt enhancement injection
VISUAL_SKILL_NAMES = frozenset(
    {
        "chart-visualization",
        "ppt-generation",
        "image-generation",
        "scientific-image-prompting",
        "data-analysis",
        "frontend-design",
        "web-design-guidelines",
    }
)

VISUAL_QUALITY_PROMPT = """\
<visual_quality_system>
**MANDATORY for all visual output tasks (charts, images, PPT, layout).**

## Design Standards
- **60-30-10 color rule**: 60% dominant, 30% secondary, 10% accent. Max 5 colors. Use hex codes (#667eea, not "purple").
- **Typography hierarchy**: Headlines 36-72pt bold (600-700), body 14-20pt regular (400), max 2 font families. Ensure 4.5:1 contrast ratio.
- **Layout**: 8-point grid spacing. 40-60% negative space. One focal point per composition. Rule of thirds for balance.
- **Accessibility**: No red-green only encoding. High contrast text. Color-blind safe palettes.

## Resource Library
Use these pre-built resources for consistent, professional output:
- **Color palettes**: Read `/mnt/skills/public/chart-visualization/references/color_palettes.json` — 12 professional palettes
  (business-blue, tech-vibrant, medical-clinical, accessible-high-contrast, etc.). Pass the `colors` array directly as `style.palette` in chart specs.
- **Chart presets**: Read `/mnt/skills/public/chart-visualization/references/chart_presets.json` — 5 scenario presets
  (executive-dashboard, technical-report, marketing-report, financial-analysis, dark-dashboard) with recommended chart types, styles, and guidelines.
- **PPT presets**: Read `/mnt/skills/public/ppt-generation/templates/presets.json` — 5 scenario presets
  (pitch-deck, quarterly-report, product-launch, training-workshop, annual-review) with storytelling arcs, style guidelines, and recommended slide counts.

When the user's request matches a preset scenario, load the relevant preset as a starting point and customize from there.

## Task-Specific Rules

**Charts**: Data integrity first — never truncate Y-axis without justification. Add data labels for key values. Avoid 3D charts, pie charts with >6 slices, and chartjunk.
Every chart must answer "what and how much" at a glance. Select a color palette from the palette library that matches the audience context.
Before generation, reduce the request into a structured chart brief, ground the exact plotted data, and ensure the chosen chart type matches the requested story, metric, grouping, and time grain.

**Technical reports / LaTeX / PDF**: Do not fabricate a polished-looking report from weak content.
If the user asks for a technical or scientific report, every core claim must be supported by concrete equations, derivations, measurements, or cited sources.
Mathematical symbols must be written in valid LaTeX math mode, never as Unicode subscripts/superscripts pasted into body text.
Figures must be generated from real local assets or validated data tables, not placeholders, remote URLs, or guessed charts.
If a figure cannot be justified from available data, omit it and state what is missing.

**PPT**: One message per slide. Storytelling arc: Hook → Context → Solution → Impact → CTA. Headlines top 20%, images 50-70% of slide area.
Generate slides sequentially with reference chaining for visual consistency.
The plan JSON now supports `transition`, `image_fit` ("cover"/"contain"), `speaker_notes` per slide, and `author`/`keywords` metadata.

**Images**: Prompts must be 150+ words with specific details (lighting, composition, color palette, camera angle, style reference).
Always use image_search for reference images when accuracy matters. Include negative_prompt to exclude unwanted elements.
In scientific contexts, first classify the request as `data_figure`, `deterministic_diagram`, or `ai_scientific_illustration`.
Only the last category may use AIGC image generation. Never use AIGC to fake a quantitative result figure.
Default to the image provider/model currently configured in Settings unless the user explicitly requests another provider.

## Iterative Refinement Workflow
For non-trivial visual tasks, follow this quality loop:
1. **Generate** the visual output
2. **Compare** output vs requirements using `visual_refinement_check` tool — assess content accuracy, style match, color fidelity, and type-specific dimensions
3. If score < 7/10, **refine** based on the identified gaps and regenerate (max 3 iterations)
4. Once score >= 7, run `visual_quality_check` for the final quality gate
5. Only then call `present_files`

## Mandatory Workflow
1. **Clarify FIRST**: If audience, style, or brand guidelines are unclear → ask_clarification before any generation.
2. **Spec before generate**: Write a JSON spec to /mnt/user-data/workspace/ defining style, colors, typography, layout.
   For charts, include `intent.data_story`, `intent.reason_for_choice`, `intent.source_summary`, and `intent.validation_notes`. Reference presets when applicable.
3. **Self-review before delivery**: Check visual hierarchy, color harmony, typography readability, spacing, alignment.
3.1. **Report review before delivery**: For PDF/LaTeX reports, verify equations render correctly, symbols are legible, and each chart matches the underlying data and narrative.
4. **Iterate if needed**: Use visual_refinement_check to compare and refine. Max 3 iterations before seeking user feedback.
5. **Present with context**: When delivering, briefly explain key design decisions and offer to iterate.

**CRITICAL**: Never deliver first-draft visual output without self-review. Quality over speed.
</visual_quality_system>"""


def get_visual_quality_prompt() -> str:
    """Return the visual quality prompt section for system prompt injection."""
    return VISUAL_QUALITY_PROMPT

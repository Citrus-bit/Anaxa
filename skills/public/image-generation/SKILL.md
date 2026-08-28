---
name: image-generation
description: Use this skill when the user requests to generate, create, imagine, or visualize images including characters, scenes, products, or any visual content. Supports structured prompts and reference images for guided generation.
---

# Image Generation Skill

## Overview

This skill generates high-quality images using structured prompts and a Python script. The workflow includes creating JSON-formatted prompts and executing image generation with optional reference images.

For **scientific illustrations** such as graphical abstracts, mechanism illustrations, concept explainers, or paper cover art, read `/mnt/skills/public/scientific-image-prompting/SKILL.md` first. That skill decides whether the request is allowed to use AIGC at all and defines the paper-ready prompt contract.

## Core Capabilities

- Create structured JSON prompts for AIGC image generation
- Support multiple reference images for style/composition guidance
- Generate images through automated Python script execution
- Handle various image generation scenarios (character design, scenes, products, etc.)

## Workflow

### Step 1: Understand Requirements

When a user requests image generation, identify:

- Subject/content: What should be in the image
- Style preferences: Art style, mood, color palette
- Technical specs: Aspect ratio, composition, lighting
- Reference images: Any images to guide generation
- You don't need to check the folder under `/mnt/user-data`

### Step 1.5: Reject Wrong Scientific Route

If the request is actually a real data figure, do **not** use this skill as the primary renderer.

Examples that must stay out of AIGC image generation:
- ROC / PR curves
- heatmaps
- volcano plots
- PCA / UMAP / t-SNE
- confusion matrices
- bar charts / line charts / histograms driven by real values

Route those requests to validated plotting workflows instead.

### Step 2: Create Structured Prompt

Generate a structured JSON file in `/mnt/user-data/workspace/` with naming pattern: `{descriptive-name}.json`

### Step 3: Execute Generation

Call the Python script:
```bash
python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/prompt-file.json \
  --reference-images /path/to/ref1.jpg /path/to/ref2.png \
  --output-file /mnt/user-data/outputs/generated-image.jpg \
  --aspect-ratio 16:9
```

Parameters:

- `--prompt-file`: Absolute path to JSON prompt file (required)
- `--reference-images`: Absolute paths to reference images (optional, space-separated)
- `--output-file`: Absolute path to output image file (required)
- `--aspect-ratio`: Aspect ratio of the generated image (optional, default: 16:9)
- `--provider`: Optional image provider override (`google-ai-studio` or `openai-compatible`)
- `--model`: Optional image model override
- `--base-url`: Optional OpenAI-compatible image base URL override
- `--image-size`: Optional `1K`, `2K`, or `4K`
- `--output-mime-type`: Optional `image/png` or `image/jpeg`
- `--scientific-mode`: Enables scientific illustration guardrails and 4K/PNG defaults
- `--manifest-file`: Optional manifest output path for generation audit
- `--draft-mode`: Lower-cost mode when no explicit model is provided

[!NOTE]
Do NOT read the python file, just call it with the parameters.
If the user has already configured an active image provider in Settings, use that provider/model by default and do not hardcode Gemini unless the user explicitly asked for Google AI Studio.

## Scientific Illustration Mode

When the request is a scientific illustration that passed the route check:

1. Read `/mnt/skills/public/scientific-image-prompting/SKILL.md`
2. Produce `prompt.json` using that contract
3. Generate the image in scientific mode
4. Keep the final output conceptual, not quantitative

Recommended command:

```bash
python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/outputs/prompt.json \
  --output-file /mnt/user-data/outputs/scientific-illustration-4k.png \
  --manifest-file /mnt/user-data/outputs/generation_manifest.json \
  --aspect-ratio 16:9 \
  --scientific-mode \
  --image-size 4K \
  --output-mime-type image/png
```

Only add `--provider`, `--model`, or `--base-url` when the user explicitly wants to override the configured default provider.

## Character Generation Example

User request: "Create a Tokyo street style woman character in 1990s"

Create prompt file: `/mnt/user-data/workspace/asian-woman.json`
```json
{
  "characters": [{
    "gender": "female",
    "age": "mid-20s",
    "ethnicity": "Japanese",
    "body_type": "slender, elegant",
    "facial_features": "delicate features, expressive eyes, subtle makeup with emphasis on lips, long dark hair partially wet from rain",
    "clothing": "stylish trench coat, designer handbag, high heels, contemporary Tokyo street fashion",
    "accessories": "minimal jewelry, statement earrings, leather handbag",
    "era": "1990s"
  }],
  "negative_prompt": "blurry face, deformed, low quality, overly sharp digital look, oversaturated colors, artificial lighting, studio setting, posed, selfie angle",
  "style": "Leica M11 street photography aesthetic, film-like rendering, natural color palette with slight warmth, bokeh background blur, analog photography feel",
  "composition": "medium shot, rule of thirds, subject slightly off-center, environmental context of Tokyo street visible, shallow depth of field isolating subject",
  "lighting": "neon lights from signs and storefronts, wet pavement reflections, soft ambient city glow, natural street lighting, rim lighting from background neons",
  "color_palette": "muted naturalistic tones, warm skin tones, cool blue and magenta neon accents, desaturated compared to digital photography, film grain texture"
}
```

Execute generation:
```bash
python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/cyberpunk-hacker.json \
  --output-file /mnt/user-data/outputs/cyberpunk-hacker-01.jpg \
  --aspect-ratio 2:3
```

With reference images:
```json
{
  "characters": [{
    "gender": "based on [Image 1]",
    "age": "based on [Image 1]",
    "ethnicity": "human from [Image 1] adapted to Star Wars universe",
    "body_type": "based on [Image 1]",
    "facial_features": "matching [Image 1] with slight weathered look from space travel",
    "clothing": "Star Wars style outfit - worn leather jacket with utility vest, cargo pants with tactical pouches, scuffed boots, belt with holster",
    "accessories": "blaster pistol on hip, comlink device on wrist, goggles pushed up on forehead, satchel with supplies, personal vehicle based on [Image 2]",
    "era": "Star Wars universe, post-Empire era"
  }],
  "prompt": "Character inspired by [Image 1] standing next to a vehicle inspired by [Image 2] on a bustling alien planet street in Star Wars universe aesthetic. Character wearing worn leather jacket with utility vest, cargo pants with tactical pouches, scuffed boots, belt with blaster holster. The vehicle adapted to Star Wars aesthetic with weathered metal panels, repulsor engines, desert dust covering, parked on the street. Exotic alien marketplace street with multi-level architecture, weathered metal structures, hanging market stalls with colorful awnings, alien species walking by as background characters. Twin suns casting warm golden light, atmospheric dust particles in air, moisture vaporators visible in distance. Gritty lived-in Star Wars aesthetic, practical effects look, film grain texture, cinematic composition.",
  "negative_prompt": "clean futuristic look, sterile environment, overly CGI appearance, fantasy medieval elements, Earth architecture, modern city",
  "style": "Star Wars original trilogy aesthetic, lived-in universe, practical effects inspired, cinematic film look, slightly desaturated with warm tones",
  "composition": "medium wide shot, character in foreground with alien street extending into background, environmental storytelling, rule of thirds",
  "lighting": "warm golden hour lighting from twin suns, rim lighting on character, atmospheric haze, practical light sources from market stalls",
  "color_palette": "warm sandy tones, ochre and sienna, dusty blues, weathered metals, muted earth colors with pops of alien market colors",
  "technical": {
    "aspect_ratio": "9:16",
    "quality": "high",
    "detail_level": "highly detailed with film-like texture"
  }
}
```
```bash
python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/star-wars-scene.json \
  --reference-images /mnt/user-data/uploads/character-ref.jpg /mnt/user-data/uploads/vehicle-ref.jpg \
  --output-file /mnt/user-data/outputs/star-wars-scene-01.jpg \
  --aspect-ratio 16:9
```

## Common Scenarios

Use different JSON schemas for different scenarios.

**Character Design**:
- Physical attributes (gender, age, ethnicity, body type)
- Facial features and expressions
- Clothing and accessories
- Historical era or setting
- Pose and context

**Scene Generation**:
- Environment description
- Time of day, weather
- Mood and atmosphere
- Focal points and composition

**Product Visualization**:
- Product details and materials
- Lighting setup
- Background and context
- Presentation angle

## Specific Templates

Read the following template file only when matching the user request.

- [Doraemon Comic](templates/doraemon.md)

## Output Handling

After generation:

- Images are typically saved in `/mnt/user-data/outputs/`
- In scientific mode, also preserve `generation_manifest.json`
- Share generated images with user using present_files tool
- Provide brief description of the generation result
- Offer to iterate if adjustments needed

## Tips: Enhancing Generation with Reference Images

For scenarios where visual accuracy is critical, **use the `image_search` tool first** to find reference images before generation.

**Recommended scenarios for using image_search tool:**
- **Character/Portrait Generation**: Search for similar poses, expressions, or styles to guide facial features and body proportions
- **Specific Objects or Products**: Find reference images of real objects to ensure accurate representation
- **Architectural or Environmental Scenes**: Search for location references to capture authentic details
- **Fashion and Clothing**: Find style references to ensure accurate garment details and styling

**Example workflow:**
1. Call the `image_search` tool to find suitable reference images:
   ```
   image_search(query="Japanese woman street photography 1990s", size="Large")
   ```
2. Download the returned image URLs to local files
3. Use the downloaded images as `--reference-images` parameter in the generation script

This approach significantly improves generation quality by providing the model with concrete visual guidance rather than relying solely on text descriptions.

## Providers (Google AI Studio, OpenAI-compatible, and MiniMax)

The script keeps the configured Anaxa provider by default and also supports
DeerFlow's MiniMax provider. Provider selection is resolved in this order:

- An explicit `--provider` or `provider=` argument.
- `IMAGE_GENERATION_PROVIDER` (accepts `gemini`, `google`, `google-ai-studio`, `minimax`, or `openai-compatible`).
- The legacy Settings value `IMAGE_GEN_ACTIVE_PROVIDER`.
- Google credentials (`GEMINI_API_KEY` or `GOOGLE_API_KEY`), then MiniMax (`MINIMAX_API_KEY`).

For Google AI Studio, `IMAGE_GEN_GOOGLE_MODEL` selects the model. For the
OpenAI-compatible provider, configure `IMAGE_GEN_OPENAI_MODEL`,
`IMAGE_GEN_OPENAI_BASE_URL`, and `IMAGE_GEN_OPENAI_API_KEY`. The base URL is
the API root (for example `https://provider.example.com/v1`); the script
appends `/images/generations`.

When only `MINIMAX_API_KEY` is available, MiniMax is selected automatically
and calls `/v1/image_generation` with model `image-01`. Override the host or
model with `MINIMAX_API_HOST` and `MINIMAX_IMAGE_MODEL`. Structured prompt
files remain provider-agnostic: MiniMax sends only the JSON `prompt` field,
enables `prompt_optimizer`, and rejects prompts over 1500 characters before
calling the API. Reference images are sent as MiniMax `subject_reference`
character images.

The optional `--scientific-mode`, `--image-size`, `--output-mime-type`,
`--draft-mode`, and `--manifest-file` arguments work with all supported
providers. Scientific mode adds the conceptual-illustration guardrail and
defaults to 4K PNG; a generation manifest records the resolved provider,
model, prompt, references, output dimensions, and retry count.

## Quality Standards (Mandatory)

### Before Generation — Clarify Requirements
If ANY of the following are unclear, call `ask_clarification` BEFORE generating:
- What is the subject/scene? (character, product, environment, abstract)
- What style? (photorealistic, illustration, anime, oil painting, etc.)
- What mood/atmosphere? (dramatic, warm, mysterious, professional)
- What is the usage context? (social media, print, presentation, avatar)
- Aspect ratio preference? (16:9 landscape, 9:16 portrait, 1:1 square)

### Image Prompt Checklist
Apply these rules when creating the JSON prompt:

1. **Prompt length**: Minimum 150 words for any non-trivial image. Complex scenes need 200-300 words.
2. **Structure**: Always include [Subject] + [Action/Pose] + [Environment] + [Style] + [Lighting] + [Color palette] + [Technical specs].
3. **Specificity**: Use exact descriptions, not vague terms. "warm golden hour sunlight at 15° angle" not "nice lighting". "#667eea deep purple" not "purple".
4. **Negative prompt**: Always include `negative_prompt` to exclude unwanted elements (blur, deformation, low quality, etc.).
5. **Reference images**: For characters, products, or specific styles — use `image_search` tool first to find references, then pass via `--reference-images`.
6. **Composition**: Specify camera angle, framing (close-up/medium/wide), depth of field, and rule of thirds placement.
7. **Style consistency**: For image series, reference previous outputs and explicitly state "match the style of the reference image exactly".

### After Generation — Self-Review
Before presenting the image to the user:
- [ ] Does the image match the requested subject and style?
- [ ] Is the composition balanced with a clear focal point?
- [ ] Are colors harmonious and appropriate for the mood?
- [ ] Is the quality sufficient? (no artifacts, blur, or deformation)
- [ ] For series: is it visually consistent with previous images?

If quality is insufficient, refine the prompt (add more detail, adjust negative_prompt) and regenerate. Max 2 retries before presenting to user with notes on what could be improved.

## Notes

- Always use English for prompts regardless of user's language
- JSON format ensures structured, parsable prompts
- Reference images enhance generation quality significantly
- Iterative refinement is normal for optimal results
- For character generation, include the detailed character object plus a consolidated prompt field

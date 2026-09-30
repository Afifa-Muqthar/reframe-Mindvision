"""
Centralized comic visual style configuration and dynamic prompt generation.

Ensures that:
1. All three scenes share a persistent character anchor (appearance, hair, clothing).
2. The visual style is coherent and cartoon/comic-oriented (not photorealistic).
3. The prompt is directly derived from the user's specific scenario, setting,
   emotional state, and narrative progression (Problem -> Reframing -> Resolution).
4. No text or words are requested of the image diffusion model.
"""

# Centralized comic / cartoon visual style configuration
COMIC_VISUAL_STYLE = (
    "vibrant modern comic book illustration, crisp ink line art, bold outlines, "
    "expressive cartoon character, colorful flat cel shading, graphic novel panel art, "
    "clean composition, no text, no words, no watermark"
)

NEGATIVE_PROMPT = (
    "photorealistic, real photograph, 3d render, hyperrealistic, deformed, "
    "disfigured, text, words, labels, watermark, blurry, extra limbs"
)


def build_character_anchor(character: dict) -> str:
    """Combines persistent character attributes into a reusable anchor string."""
    appearance = character.get("appearance", "young adult with short dark hair, expressive eyes")
    clothing = character.get("clothing", "wearing a dark gray hoodie and jeans")
    description = character.get("description", "relatable character")
    return f"{appearance}, {clothing}, {description}"


def build_scene_prompts(story: dict) -> list:
    """
    Constructs 3 distinct scene prompts for SD-Turbo from the structured story.
    
    Each prompt combines:
      - The persistent character anchor (shared across all 3 panels)
      - The scene-specific setting, environment, and physical action
      - The evolving emotional expression and body pose
      - The progressive lighting/mood (Problem: dim/moody -> Reframe: warm/reflective -> Resolution: bright/hopeful)
      - The centralized comic art style
    """
    character = story.get("character", {})
    char_anchor = build_character_anchor(character)
    style = story.get("visual_style", COMIC_VISUAL_STYLE)
    emotion = story.get("emotion", "worried")
    scenes = story.get("scenes", [])

    prompts = []

    # Lighting and mood cues progressing across the three stages
    mood_cues = [
        f"feeling overwhelmed and burdened by {emotion}, slumped posture, cool shadow-toned moody lighting",
        "pausing in thoughtful calm reflection, steady balanced posture, soft warm ambient lighting",
        "focused constructive posture with quiet determination, bright clear hopeful morning light",
    ]

    for i, scene in enumerate(scenes[:3]):
        visual_desc = scene.get("visual_description", "")
        mood = mood_cues[i] if i < len(mood_cues) else "balanced comic lighting"

        prompt_parts = [
            char_anchor,
            visual_desc,
            mood,
            style,
        ]
        # Filter empty parts and join
        full_prompt = ", ".join(p.strip() for p in prompt_parts if p.strip())
        prompts.append(full_prompt)

    # Ensure exactly 3 prompts
    while len(prompts) < 3:
        prompts.append(f"{char_anchor}, calm scene, {style}")

    return prompts[:3]

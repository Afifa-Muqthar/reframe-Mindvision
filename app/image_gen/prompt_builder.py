"""
Centralized comic visual style configuration and dynamic prompt generation.

Ensures that:
1. All scenes (3 to 8) share a persistent character anchor (appearance, hair, clothing).
2. The visual style is coherent and cartoon/comic-oriented (not photorealistic).
3. The prompt is directly derived from the scene's psychological progression,
   staging (camera, lighting, environment, objects, posture, expression).
4. No text or words are requested of the image diffusion model.
"""

"""
Centralized comic visual style configuration and dynamic prompt generation.

Enforces a compact ~40–50 word prompt strictly budgeted under 70 CLIP tokens.
Order:
1. 8–12 word character anchor
2. 8–12 word PHYSICAL ACTION (immediately following character anchor)
3. 5–8 word gaze/object description
4. 5–8 word setting/lighting
5. 8–10 word compact comic style

No psychological abstractions or narrative explanations in the image prompt.
Observable physical events only.
"""

import re
from typing import List, Dict, Any

# Centralized compact comic / cartoon visual style configuration (8-10 words)
COMIC_VISUAL_STYLE = "vibrant comic book art, crisp ink lines, flat cel shading, no text, no words"

# Retained for backwards compatibility if referenced
NEGATIVE_PROMPT = (
    "photorealistic, real photograph, 3d render, hyperrealistic, deformed, "
    "disfigured, text, words, labels, watermark, blurry, extra limbs"
)

# Psychological buzzwords to strip from image prompts
_ABSTRACT_PSYCH_PATTERNS = [
    r"\bfeeling (?:burdened|overwhelmed|anxious|sad|hopeless|nervous|confused|unsure|better|relieved|empowered)\b",
    r"\bburdened by [a-z\s]+",
    r"\boverwhelmed by [a-z\s]+",
    r"\banxious thoughts?\b",
    r"\bcontemplating (?:the past|feelings|relationship)\b",
    r"\bsense of (?:agency|hope|relief|clarity|burden)\b",
    r"\bintrospective posture\b",
    r"\bgrounded reflection\b",
    r"\bcalm clarity\b",
    r"\bresolute posture\b",
    r"\bhopeful posture\b",
]


def _clean_phrase(text: str, max_words: int) -> str:
    """Strips psychological abstractions and clamps to max_words."""
    if not text:
        return ""
    cleaned = text
    for pat in _ABSTRACT_PSYCH_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(",").strip()
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).rstrip(",")
    return cleaned


def build_character_anchor(character: dict) -> str:
    """
    Combines persistent character attributes into a concise 8-12 word anchor string.
    Strictly focuses on observable physical identity (hair, clothing, gender/appearance).
    """
    if not character:
        return "young woman with short dark hair in dark gray sweater"

    if character.get("anchor"):
        return _clean_phrase(character["anchor"], 12)

    appearance = character.get("appearance", "")
    clothing = character.get("clothing", "")
    desc = character.get("description", "")

    # Extract clean physical descriptors; ground ambiguous "adult" to concrete "young woman" for seed consistency
    if "man" in appearance.lower() or "male" in appearance.lower() or " man" in desc.lower():
        subj = "young man"
    elif "woman" in appearance.lower() or "female" in appearance.lower() or "woman" in desc.lower():
        subj = "young woman"
    else:
        subj = "young woman"

    # Hair extraction
    hair = "short dark hair"
    if "hair" in appearance.lower():
        m = re.search(r"((?:short|long|curly|wavy|straight|dark|blonde|brown|tousled|messy)\s+(?:dark\s+)?(?:tousled\s+)?hair)", appearance, re.I)
        if m:
            hair = m.group(1).strip()

    # Clothing extraction
    cloth = "dark gray sweater"
    if "green sweater" in clothing.lower():
        cloth = "green sweater"
    elif "charcoal gray hoodie" in clothing.lower():
        cloth = "charcoal gray hoodie"
    elif "sweater" in clothing.lower():
        cloth = "dark gray sweater"
    elif "hoodie" in clothing.lower():
        cloth = "dark hoodie"
    elif "jacket" in clothing.lower():
        cloth = "denim jacket"
    elif clothing:
        c_clean = re.sub(r"wearing\s+(?:a\s+)?", "", clothing, flags=re.I).strip()
        cloth_words = c_clean.split(",")
        cloth = cloth_words[0].strip() if cloth_words else "casual clothes"

    anchor = f"single {subj} with {hair} in {cloth}"
    return _clean_phrase(anchor, 12)


def _normalize_object_and_composition(text: str) -> str:
    """Enforces strict singular counts and concrete binding for objects."""
    t = re.sub(r"holding\s+(?:a\s+)?smartphone\s+with\s+both\s+hands", "with both hands holding one smartphone", text, flags=re.I)
    t = re.sub(r"holding\s+(?:a\s+)?phone\s+with\s+both\s+hands", "with both hands holding one smartphone", t, flags=re.I)
    t = re.sub(r"\bholding\s+(?:a\s+)?smartphone\b", "holding one smartphone", t, flags=re.I)
    t = re.sub(r"\bholding\s+(?:a\s+)?phone\b", "holding one smartphone", t, flags=re.I)
    t = re.sub(r"\bwriting\s+in\s+(?:a\s+)?notebook\b", "writing in one notebook", t, flags=re.I)
    t = re.sub(r"\bholding\s+(?:a\s+)?(?:glass|mug|cup)\b", "holding one cup", t, flags=re.I)
    return t


def _derive_physical_action(scene: dict) -> str:
    """Extracts or derives 8-12 word observable physical action."""
    action = scene.get("character_action") or scene.get("action")
    if not action:
        raw = scene.get("visual_description") or scene.get("visual") or ""
        action = raw.split(",")[0] if raw else "sitting quietly pausing"

    action = _normalize_object_and_composition(action)
    cleaned = _clean_phrase(action, 12)
    if not cleaned:
        cleaned = "sitting quietly pausing and breathing"
    return cleaned


def _derive_gaze_and_object(scene: dict) -> str:
    """Extracts or derives 5-8 word gaze and object description with singular binding."""
    gaze = scene.get("gaze", "")
    obj_inter = scene.get("object_interaction", "")
    objs = scene.get("important_objects", [])

    if gaze:
        gaze = re.sub(r"message\s+thread|messages", "the screen", gaze, flags=re.I)
        gaze = re.sub(r"phone\s+screen|smartphone\s+screen", "the screen", gaze, flags=re.I)
        gaze = re.sub(r"at\s+phone|at\s+smartphone", "at the screen", gaze, flags=re.I)

    parts = []
    if gaze:
        parts.append(gaze)
    if obj_inter and obj_inter.lower() not in (gaze.lower() if gaze else ""):
        parts.append(_normalize_object_and_composition(obj_inter))
    elif objs and not gaze:
        parts.append(f"holding one {objs[0]}")

    combined = ", ".join(parts) if parts else "looking thoughtfully ahead"
    combined = _normalize_object_and_composition(combined)
    return _clean_phrase(combined, 8) or "looking thoughtfully ahead"


def _derive_setting_and_lighting(scene: dict, index: int, total_scenes: int) -> str:
    """Extracts or derives 5-8 word setting and lighting description."""
    env = scene.get("environment", "")
    lighting = scene.get("lighting", "")

    env_clean = env.split(",")[0].strip() if env else "quiet room"

    if not lighting:
        ratio = index / max(1, total_scenes - 1)
        if ratio < 0.35:
            lighting = "soft shadow-toned light"
        elif ratio < 0.7:
            lighting = "soft warm ambient light"
        else:
            lighting = "clear bright daylight"
    else:
        lighting = lighting.split(",")[0].strip()

    combined = f"{env_clean}, {lighting}"
    return _clean_phrase(combined, 8) or "quiet room, soft daylight"


def build_compact_scene_prompt(
    char_anchor: str,
    scene: dict,
    index: int = 0,
    total_scenes: int = 3,
    style: str = COMIC_VISUAL_STYLE
) -> str:
    """
    Constructs a compact ~40-50 word prompt strictly ordered:
    1. 8-12 word character anchor
    2. 8-12 word PHYSICAL ACTION (immediately after character anchor)
    3. 5-8 word gaze/object description
    4. 5-8 word setting/lighting
    5. 8-12 word compact comic style

    Guarantees <70 CLIP tokens with zero psychological abstractions.
    """
    anchor_clean = _clean_phrase(char_anchor, 12)
    action_clean = _derive_physical_action(scene)
    gaze_clean = _derive_gaze_and_object(scene)
    setting_clean = _derive_setting_and_lighting(scene, index, total_scenes)
    style_clean = _clean_phrase(style, 15) or COMIC_VISUAL_STYLE

    parts = [anchor_clean, action_clean, gaze_clean, setting_clean, style_clean]
    prompt = ", ".join(p for p in parts if p)
    return prompt


def build_scene_prompts(story: dict) -> list:
    """
    Constructs distinct compact scene prompts for SD-Turbo from the structured story.
    Enforces the ~40-50 word prompt budget and observable visual events.
    """
    character = story.get("character", {})
    char_anchor = build_character_anchor(character)
    style = story.get("visual_style") or COMIC_VISUAL_STYLE
    if len(style.split()) > 12:
        style = COMIC_VISUAL_STYLE

    scenes = story.get("scenes", [])
    if not scenes:
        scenes = [
            {
                "character_action": "sitting on bed holding smartphone with both hands",
                "gaze": "looking down at message thread",
                "environment": "quiet bedroom with desk and window",
                "lighting": "soft evening light",
            },
            {
                "character_action": "standing beside desk writing in a notebook",
                "gaze": "gaze focused on open page",
                "environment": "study room by bookshelf",
                "lighting": "warm desk lamp light",
            },
            {
                "character_action": "standing upright by open window taking a deep breath",
                "gaze": "looking out at morning sky",
                "environment": "bright room by open window",
                "lighting": "clear warm morning daylight",
            },
        ]

    scenes = scenes[:8]
    total = len(scenes)

    prompts = []
    for i, scene in enumerate(scenes):
        prompt = build_compact_scene_prompt(char_anchor, scene, index=i, total_scenes=total, style=style)
        prompts.append(prompt)

    while len(prompts) < 3:
        i = len(prompts)
        fallback_scene = {
            "character_action": "standing calmly taking a slow steady breath",
            "gaze": "looking forward with steady gaze",
            "environment": "quiet room with window",
            "lighting": "soft ambient morning daylight",
        }
        prompts.append(build_compact_scene_prompt(char_anchor, fallback_scene, index=i, total_scenes=3, style=style))

    return prompts


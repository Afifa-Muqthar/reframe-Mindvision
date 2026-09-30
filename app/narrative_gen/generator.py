"""
Structured 3-Stage Narrative Generation grounded in Psychological Reasoning.

Transforms a user's situation into a personalized 3-stage visual & auditory narrative:
  SCENE 1 — PROBLEM: Grounded in user's trigger, dilemma, and emotional tension.
  SCENE 2 — REFRAMING: Perspective shift validating reality & focusing on controllability.
  SCENE 3 — RESOLUTION: Manageable, realistic action (boundaries, safety, or micro-steps).

Outputs a complete structured story dictionary containing:
- title, emotion, context, psychological pattern
- persistent character anchor (appearance, clothing, description)
- 3 scenes with visual descriptions, comic bubbles (type, dialogue, captions), and narratives
- coherent psychological voice script for TTS
"""

import gc
import re
from typing import Dict, Any

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

from app.image_gen.prompt_builder import COMIC_VISUAL_STYLE

_MODEL_NAME_PRIMARY = "google/flan-t5-base"
_MODEL_NAME_FALLBACK = "google/flan-t5-small"

_BANNED_PHRASES = [
    "you will definitely", "you'll definitely", "guaranteed to", "i promise you",
    "you will certainly", "100% certain", "will always work out",
]

_HARMFUL_PATTERNS = [
    "idiot", "stupid", "worthless", "pathetic", "loser", "moron", "cynical",
    "hopeless case", "give up", "your fault", "deserve this",
]

_ECHO_MARKERS = [
    "describe their situation and feeling",
    "gently introduce the technique",
    "give one small, concrete, doable action",
    "write a short, grounded, second-person narrative",
    "write the three scenes",
]

_PROMPT_TEMPLATE = """You are an empathetic, evidence-informed psychological writer.
Situation: "{raw_text}"
Psychological Pattern: {pattern}
What is controllable: {controllable}
Specific Reframe: {specific_reframe}
Concrete Action: {concrete_action}

Write a short, grounded second-person narrative in exactly three labeled scenes:
SCENE 1 - PROBLEM: Acknowledge the actual trigger and the internal conflict or burden with empathy.
SCENE 2 - REFRAMING: Communicate the specific reframe clearly, validating reality and focusing on what can be controlled.
SCENE 3 - RESOLUTION: Present the concrete constructive next action or boundary.

Keep each scene to 1-2 clear sentences. Do not use generic optimism or words like "guaranteed" or "definitely".
Now write the three scenes:
"""


class NarrativeGenerator:
    """
    Generates a structured 3-stage narrative:
      PROBLEM -> REFRAMING -> RESOLUTION
    anchored in evidence-informed psychological reasoning.
    """

    def __init__(self, use_small: bool = False):
        if AutoTokenizer is None:
            self._tokenizer = None
            self._model = None
            return

        model_name = _MODEL_NAME_FALLBACK if use_small else _MODEL_NAME_PRIMARY
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception:
            if use_small:
                raise
            self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME_FALLBACK)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME_FALLBACK)

    def generate(self, case_frame, technique: dict = None) -> dict:
        """
        Main entry point: returns a structured story dictionary directly
        usable by both the image generator and TTS.
        """
        strategy = getattr(case_frame, "strategy", {}) or {}
        pattern = getattr(case_frame, "pattern", "") or "Personal reflection and stress"
        controllable = getattr(case_frame, "controllable", "") or "Your boundaries and personal response"
        specific_reframe = getattr(case_frame, "specific_reframe", "") or (technique.get("description", "") if technique else "")
        concrete_action = getattr(case_frame, "concrete_action", "") or "Take one small constructive step today."

        raw_output = ""
        if self._model and self._tokenizer:
            prompt = _PROMPT_TEMPLATE.format(
                raw_text=case_frame.raw_text,
                pattern=pattern,
                controllable=controllable,
                specific_reframe=specific_reframe,
                concrete_action=concrete_action,
            )
            try:
                inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    num_beams=3,
                    no_repeat_ngram_size=3,
                )
                raw_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                raw_output = self._sanitize(raw_output)
            except Exception:
                raw_output = ""

        structured_story = self._build_structured_story(raw_output, case_frame, strategy, technique)
        return structured_story

    def _sanitize(self, text: str) -> str:
        lowered = text.lower()
        for phrase in _BANNED_PHRASES:
            if phrase in lowered:
                idx = lowered.find(phrase)
                text = text[:idx] + "this may help" + text[idx + len(phrase):]
                lowered = text.lower()
        return text

    def _build_structured_story(self, text: str, case_frame, strategy: dict, technique: dict) -> dict:
        """
        Extracts validated model text or provides evidence-based scenes
        tailored from the psychological strategy and archetype.
        """
        archetype = strategy.get("comic_archetype", {}) if strategy else {}
        if not archetype:
            # Fallback archetype if not present in strategy
            archetype = {
                "context_name": "Personal Growth & Clarity",
                "character": {
                    "description": "a relatable young adult finding clarity",
                    "appearance": "short dark hair, expressive eyes, grounded posture",
                    "clothing": "wearing a comfortable dark gray sweater and dark pants",
                },
                "scenes": [
                    {
                        "stage": "problem",
                        "title": "1. PROBLEM",
                        "visual": "sitting by a wooden desk looking burdened by anxious thoughts, cool atmospheric shadows",
                        "dialogue": "Everything feels confusing and heavy right now.",
                        "bubble_type": "thought",
                        "caption": "When stressors pile up, it is easy to feel stuck.",
                    },
                    {
                        "stage": "reframing",
                        "title": "2. REFRAMING",
                        "visual": "pausing with hands resting open on table, taking a slow grounding breath, warm ambient light",
                        "dialogue": "Feeling unsure doesn't mean I am stuck. I can focus on what I control.",
                        "bubble_type": "thought",
                        "caption": "Acknowledge the difficulty without losing your sense of agency.",
                    },
                    {
                        "stage": "resolution",
                        "title": "3. RESOLUTION",
                        "visual": "standing upright by a sunlit window with a calm steady posture, holding a glass of water",
                        "dialogue": "I will take one small constructive step today.",
                        "bubble_type": "speech",
                        "caption": "Focus your energy on what is manageable and safe.",
                    },
                ],
            }

        scenes_data = self._extract_or_fallback_scenes(text, case_frame, strategy, archetype)

        voice_script = strategy.get("voice_script")
        if not voice_script:
            voice_script = (
                f"{scenes_data[0]['narrative']} "
                f"{scenes_data[1]['narrative']} "
                f"{scenes_data[2]['narrative']}"
            )

        theme_note = getattr(case_frame, "recurring_theme_note", "")
        strategy_name = strategy.get("name") or (technique.get("name") if technique else "Evidence-Informed Reframing")

        structured = {
            "title": f"Reframing with {strategy_name}",
            "emotion": case_frame.core_emotion or "unclear",
            "context": archetype.get("context_name", "Mental Reframing"),
            "character": archetype.get("character", {}),
            "visual_style": COMIC_VISUAL_STYLE,
            "scenes": scenes_data,
            "voice_script": voice_script,
            "theme_note": theme_note,
            "strategy": strategy,
        }
        return structured

    def _extract_or_fallback_scenes(self, text: str, case_frame, strategy: dict, archetype: dict) -> list:
        required_headers = [r"SCENE\s*1\s*[-—:]?\s*PROBLEM", r"SCENE\s*2\s*[-—:]?\s*REFRAMING?", r"SCENE\s*3\s*[-—:]?\s*RESOLUTION"]
        has_all_headers = all(re.search(h, text, re.IGNORECASE) for h in required_headers)
        lowered = text.lower()
        has_harmful = any(p in lowered for p in _HARMFUL_PATTERNS)
        is_echo = any(p in lowered for p in _ECHO_MARKERS)

        model_narratives = {}
        if has_all_headers and not has_harmful and not is_echo:
            splits = re.split(r"(SCENE\s*[123]\s*[-—:]?\s*(?:PROBLEM|REFRAMING?|RESOLUTION)):\s*", text, flags=re.IGNORECASE)
            for i in range(1, len(splits) - 1, 2):
                hdr = splits[i].upper()
                content = splits[i + 1].strip()
                if "1" in hdr or "PROBLEM" in hdr:
                    model_narratives["problem"] = content
                elif "2" in hdr or "REFRAM" in hdr:
                    model_narratives["reframing"] = content
                elif "3" in hdr or "RESOLUTION" in hdr:
                    model_narratives["resolution"] = content

        # Evidence-informed grounding
        is_threat = getattr(case_frame, "is_external_threat", False)
        trigger = getattr(case_frame, "trigger", "")
        reframe = getattr(case_frame, "specific_reframe", "") or strategy.get("reframe", "")
        action = getattr(case_frame, "concrete_action", "") or strategy.get("concrete_action", "")

        if is_threat:
            fallback_p = (
                f"You're dealing with an unpredictable and tense situation: '{case_frame.raw_text.strip()}'. "
                f"When someone lashes out and later acts as though everything is normal, it naturally triggers confusion, tension, and self-doubt."
            )
            fallback_r = reframe or (
                "Their acting normal afterward does not erase what happened earlier. "
                "You do not have to rewrite the experience or take responsibility for their behavior. "
                "Recognizing the pattern without blaming yourself allows you to decide what keeps you safest and most in control."
            )
            fallback_res = action or (
                "Right now, you don't need to fix their behavior. "
                "Give yourself physical or emotional distance, write down what happened while you remember it clearly, "
                "and reach out to someone you trust for support."
            )
        else:
            fallback_p = (
                f"You're facing a tough moment: '{case_frame.raw_text.strip()}'. "
                f"When this pressure builds, it can feel heavy and overwhelming."
            )
            fallback_r = reframe or (
                f"Taking a step back can help you look at this thought differently: "
                f"Separate what you can directly influence from the outcomes you cannot predict. "
                f"Focus on the pieces within your personal control."
            )
            fallback_res = action or (
                "Focus on taking one realistic constructive step today, "
                "giving yourself credit for what is directly manageable right now."
            )

        p_narrative = model_narratives.get("problem") or fallback_p
        r_narrative = model_narratives.get("reframing") or fallback_r
        res_narrative = model_narratives.get("resolution") or fallback_res

        arch_scenes = archetype.get("scenes", [{}, {}, {}])

        return [
            {
                "stage": "problem",
                "title": "1. PROBLEM",
                "narrative": p_narrative,
                "visual_description": arch_scenes[0].get("visual", "standing in a room with tense atmosphere, visibly burdened"),
                "dialogue": arch_scenes[0].get("dialogue", "This situation feels overwhelming."),
                "bubble_type": arch_scenes[0].get("bubble_type", "thought"),
                "caption": arch_scenes[0].get("caption", "A difficult situation creates internal tension and uncertainty."),
            },
            {
                "stage": "reframing",
                "title": "2. REFRAMING",
                "narrative": r_narrative,
                "visual_description": arch_scenes[1].get("visual", "pausing thoughtfully, taking a calm breath, steady posture"),
                "dialogue": arch_scenes[1].get("dialogue", "I can acknowledge what is happening without blaming myself."),
                "bubble_type": arch_scenes[1].get("bubble_type", "thought"),
                "caption": arch_scenes[1].get("caption", "A shift in perspective allows you to see what is truly within your control."),
            },
            {
                "stage": "resolution",
                "title": "3. RESOLUTION",
                "narrative": res_narrative,
                "visual_description": arch_scenes[2].get("visual", "in a safe supportive space, taking a concrete positive step"),
                "dialogue": arch_scenes[2].get("dialogue", "I'll focus on what I can control and reach out to trusted support."),
                "bubble_type": arch_scenes[2].get("bubble_type", "speech"),
                "caption": arch_scenes[2].get("caption", "Focus on safety, support, and actionable steps forward."),
            },
        ]

    def unload(self):
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()


def format_narrative_text(story: dict) -> str:
    """Formats structured story dictionary into human-readable 3-scene text with captions."""
    lines = []
    for scene in story.get("scenes", []):
        stage_name = scene.get("title", scene.get("stage", "")).upper()
        narrative = scene.get("narrative", "")
        bubble_type = scene.get("bubble_type", "thought").title()
        dialogue = scene.get("dialogue", "")
        caption = scene.get("caption", "")
        lines.append(f"{stage_name}:\n{narrative}")
        if caption:
            lines.append(f"Caption: {caption}")
        lines.append(f"[{bubble_type} Bubble: \"{dialogue}\"]\n")
    return "\n".join(lines).strip()


def parse_narrative_parts(narrative_or_story) -> dict:
    """Helper extracting problem, reframing, resolution from story dict or text."""
    if isinstance(narrative_or_story, dict) and "scenes" in narrative_or_story:
        parts = {}
        for s in narrative_or_story["scenes"]:
            parts[s["stage"]] = s["narrative"]
        parts["current_reality"] = parts.get("problem", "")
        parts["reframe"] = parts.get("reframing", "")
        parts["desired_future"] = parts.get("resolution", "")
        parts["next_step"] = parts.get("resolution", "")
        return parts

    text = str(narrative_or_story)
    parts = {"problem": "", "reframing": "", "resolution": ""}
    pattern = r"(?:SCENE\s*[123]\s*[-—:]?\s*)?(PROBLEM|REFRAMING?|RESOLUTION|CURRENT REALITY|REFRAME|DESIRED FUTURE|NEXT STEP):\s*"
    pieces = re.split(pattern, text, flags=re.IGNORECASE)
    for i in range(1, len(pieces) - 1, 2):
        key = pieces[i].strip().lower().replace(" ", "_")
        val = pieces[i + 1].strip()
        if "problem" in key or "reality" in key:
            parts["problem"] = val
            parts["current_reality"] = val
        elif "refram" in key:
            parts["reframing"] = val
            parts["reframe"] = val
        elif "resolution" in key or "future" in key or "next" in key:
            parts["resolution"] = val
            parts["desired_future"] = val
            parts["next_step"] = val

    return parts

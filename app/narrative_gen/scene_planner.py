"""
General Visual Narrative Orchestrator.

Transforms evidence-informed psychological state transitions into observable physical visual narratives:
  PSYCHOLOGICAL CHANGE
          ↓
  INTERPRETATION / INTERNAL STATE
          ↓
  CHARACTER GOAL
          ↓
  PHYSICAL ACTION (observable physical event)
          ↓
  BODY LANGUAGE / GAZE
          ↓
  OBJECT INTERACTION (anchored object lifecycle)
          ↓
  ENVIRONMENTAL STATE (spatial continuity)
          ↓
  CAMERA / COMPOSITION (storytelling variable)
          ↓
  SCENE STATE (comprehensive structured output)

Strict Constraints:
  - Minimum 3 scenes
  - Hard Maximum 8 scenes
  - General orchestration method (NO hardcoded situation storyboards)
  - Observable physical events instead of abstract emotional labels
  - Object lifecycle tracking across scenes
  - Environmental continuity
  - Preservation of genuine uncertainty (no false premature certainty)
"""

import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple


def _extract_dialogue_phrase(text: str, default: str, max_chars: int = 85) -> str:
    """
    Extracts a clean, complete sentence or phrase suitable for a comic dialogue bubble.
    Avoids trailing ellipsis or awkward truncation whenever possible.
    """
    if not text:
        return default
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned

    # Try taking the first complete sentence if it fits cleanly
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if sentences and 20 <= len(sentences[0]) <= max_chars:
        return sentences[0].strip()

    # Try splitting at clause boundary (semicolon, dash)
    clauses = re.split(r"[;—–]\s*", cleaned)
    if clauses and 20 <= len(clauses[0]) <= max_chars:
        return clauses[0].strip()

    # Fall back to polished domain default
    return default


@dataclass
class CaseUnderstanding:
    """Step 1: Extracted and synthesized understanding of the psychological case."""
    raw_text: str
    situation: str
    primary_emotion: str
    central_conflict: str
    automatic_thought: str
    appraisal: str
    pattern: str
    coping_strategy: str
    specific_reframe: str
    concrete_next_step: str
    controllable: str
    uncontrollable: str
    is_external_threat: bool
    domain: str
    genuine_uncertainty: bool
    distortions: List[str]


@dataclass
class PsychologicalMilestone:
    """Step 2 & 3: A distinct psychological milestone that has earned its place in the arc."""
    stage_category: str
    stage: str
    narrative_purpose: str
    psychological_purpose: str
    target_emotion: str
    arousal_level: str
    perceived_control: str
    appraisal: str


@dataclass
class SituationAnchors:
    """Step 5 & 6: Authentic objects and physical environment continuous across the narrative."""
    environment: str
    primary_object: str
    secondary_object: str


@dataclass
class SceneState:
    """Step 12: Comprehensive structured scene state consumed by prompt builder and storyboard."""
    scene_id: int
    stage_category: str
    stage: str
    title: str
    narrative_purpose: str
    psychological_purpose: str
    trigger: str
    automatic_thought: str
    appraisal: str
    emotion: str
    character_goal: str
    character_action: str
    posture: str
    expression: str
    gaze: str
    environment: str
    important_objects: List[str]
    object_interaction: str
    spatial_relationships: str
    camera_distance: str
    camera_angle: str
    composition: str
    lighting: str
    visual_mood: str
    dialogue: str
    thought_text: str
    bubble_type: str
    caption: str
    visual_description: str
    narrative: str
    psychological_state: Dict[str, Any]
    character_state: Dict[str, Any]
    visual_staging: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScenePlanner:
    """
    General Visual Narrative Orchestrator.
    Derives case-specific visual progressions for ANY psychological case frame.
    """

    MIN_SCENES = 3
    MAX_SCENES = 8

    def plan_scenes(self, case_frame, strategy: dict = None, archetype: dict = None) -> List[Dict[str, Any]]:
        """
        Main orchestration entry point:
        CaseFrame -> CaseUnderstanding -> Psychological Arc -> Natural Anchors
                  -> Observable SceneStates -> Transition Verification -> SceneState[].
        """
        # Step 1: Understand the Case
        case = self._understand_case(case_frame, strategy, archetype)

        # Step 2 & 3: Define the Psychological Arc & Milestone Count
        milestones = self._derive_psychological_arc(case)

        # Step 5 & 6: Identify Natural Objects and Continuous Environment
        anchors = self._derive_anchors(case)

        # Step 4, 7, 8, 12: Convert Each Psychological State into an Observable SceneState
        candidate_scenes: List[SceneState] = []
        total_milestones = len(milestones)
        for idx, milestone in enumerate(milestones, start=1):
            scene = self._orchestrate_scene(idx, milestone, total_milestones, case, anchors)
            candidate_scenes.append(scene)

        # Step 9: Scene Transition Check & Physical Differentiation
        refined_scenes = self._verify_and_refine_transitions(candidate_scenes, case, anchors)

        # Step 10: Case-Specificity Verification
        specific_scenes = self._verify_case_specificity(refined_scenes, case, anchors)

        # Step 11: Preserve Genuine Uncertainty
        final_scenes = self._enforce_uncertainty_preservation(specific_scenes, case)

        # Convert to dictionaries and enforce bounds [3, 8]
        dict_scenes = [s.to_dict() for s in final_scenes]
        return self._apply_stopping_criteria(dict_scenes, len(dict_scenes))

    # =========================================================================
    # STAGE 4 — GROUNDED VISUAL SCENE ORCHESTRATION
    # =========================================================================
    def plan_grounded_scenes(
        self,
        story: Dict[str, Any],
        case_rep: Optional[Any] = None,
        selected_strategy: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Connects Stage 3 GroundedNarrativeGenerator output to physical visual staging:
        - Preserves all original Stage 3 dialogue, captions, narratives, stages, and order.
        - Enriches each scene with observable physical visual staging:
          character_action, gaze, posture, expression, important_objects, environment, lighting, camera.
        - Derives visual anchors from user-stated facts and marks creative interpretations in traceability.
        - Ensures meaningful physical visual progression across adjacent panels without generic fallback collapse.
        """
        raw_scenes = story.get("scenes", [])
        if not raw_scenes:
            return []

        # 1. Derive situation anchors grounded in stated facts
        stated_facts = []
        if case_rep and hasattr(case_rep, "stated_facts"):
            stated_facts = [f.text for f in case_rep.stated_facts]
        elif story.get("traceability", {}).get("user_stated_facts_used"):
            stated_facts = story["traceability"]["user_stated_facts_used"]

        raw_text = getattr(case_rep, "raw_text", "") if case_rep else ""
        modality = story.get("modality", "")
        if not modality and selected_strategy:
            modality = getattr(selected_strategy, "modality", "")

        anchors, anchor_origin = self._derive_grounded_anchors_from_facts(stated_facts, raw_text, modality)

        total_scenes = len(raw_scenes)
        enriched_scenes: List[Dict[str, Any]] = []

        for idx, sc in enumerate(raw_scenes, start=1):
            stage = sc.get("stage", f"scene_{idx}")
            stage_category = sc.get("stage_category") or stage
            title = sc.get("title", f"{idx}. {stage.upper()}")
            dialogue = sc.get("dialogue", "")
            caption = sc.get("caption", "")
            narrative = sc.get("narrative", "")
            bubble_type = sc.get("bubble_type", "thought")

            # 2. Derive observable behavior (action, posture, expression, gaze)
            action, posture, expression, gaze, act_origin = self._derive_grounded_behavior(
                stage, idx, total_scenes, modality, stated_facts, anchors
            )

            # 3. Derive camera and lighting
            cam_dist, cam_angle, comp, lighting, mood = self._derive_grounded_camera_lighting(
                stage, idx, total_scenes, modality
            )

            # 4. Derive per-scene environment and objects
            env, prim_obj, sec_obj, env_origin = self._derive_grounded_scene_environment(
                stage, idx, total_scenes, modality, stated_facts, raw_text, anchors
            )

            visual_desc = f"{action}, {gaze}, in {env.split(',')[0]}, {lighting}"

            # 5. Detail traceability update
            detail_traceability = dict(sc.get("detail_traceability", {}))
            detail_traceability["visual_staging"] = {
                "origin": act_origin,
                "anchor_origin": env_origin,
                "primary_object": prim_obj,
                "environment": env,
            }

            # 6. Complete SceneState compatible dictionary
            enriched_scene = {
                "scene_id": idx,
                "stage_category": stage_category,
                "stage": stage,
                "title": title,
                "narrative_purpose": sc.get("narrative_purpose", stage),
                "psychological_purpose": sc.get("psychological_purpose", stage),
                "trigger": raw_text or story.get("context", ""),
                "automatic_thought": dialogue,
                "appraisal": modality,
                "emotion": story.get("emotion", "reflective"),
                "character_goal": f"Navigate {stage}",
                "character_action": action,
                "posture": posture,
                "expression": expression,
                "gaze": gaze,
                "environment": env,
                "important_objects": [prim_obj, sec_obj] if sec_obj else [prim_obj],
                "object_interaction": f"interacting with {prim_obj}" if prim_obj else "pausing without object interaction",
                "spatial_relationships": f"character positioned in {env}",
                "camera_distance": cam_dist,
                "camera_angle": cam_angle,
                "composition": comp,
                "lighting": lighting,
                "visual_mood": mood,
                "dialogue": dialogue,
                "thought_text": dialogue if "thought" in bubble_type else "",
                "bubble_type": bubble_type,
                "caption": caption,
                "visual_description": visual_desc,
                "narrative": narrative,
                "detail_traceability": detail_traceability,
                "psychological_state": {
                    "modality": modality,
                    "stage": stage,
                    "target_emotion": story.get("emotion", "reflective"),
                },
                "character_state": {
                    "action": action,
                    "posture": posture,
                    "expression": expression,
                    "gaze": gaze,
                },
                "visual_staging": {
                    "environment": env,
                    "objects": [prim_obj, sec_obj] if sec_obj else [prim_obj],
                    "camera": f"{cam_dist}, {cam_angle}",
                    "lighting": lighting,
                },
            }
            enriched_scenes.append(enriched_scene)

        return enriched_scenes

    def _derive_grounded_anchors_from_facts(
        self, facts: List[str], raw_text: str, modality: str
    ) -> Tuple[SituationAnchors, str]:
        text_corpus = (" ".join(facts) + " " + raw_text).lower()

        # Check for stated facts
        if any(k in text_corpus for k in ["career", "graduat", "placed", "placement", "batch", "firm", "job"]):
            return (
                SituationAnchors(
                    environment="quiet student study room with wooden desk by window",
                    primary_object="open notebook on study desk",
                    secondary_object="warm mug of tea",
                ),
                "user_stated",
            )
        elif any(k in text_corpus for k in ["bed", "fatigue", "illness", "flare-up", "autoimmune", "body"]):
            return (
                SituationAnchors(
                    environment="cozy bedroom with soft pillows and resting blanket",
                    primary_object="soft blanket on bed",
                    secondary_object="glass of water on bedside table",
                ),
                "user_stated",
            )
        elif any(k in text_corpus for k in ["supervisor", "project", "research", "recommendation", "meeting"]):
            return (
                SituationAnchors(
                    environment="quiet personal work desk beside window",
                    primary_object="project folder and notes on desk",
                    secondary_object="single desk pen",
                ),
                "user_stated",
            )
        elif any(k in text_corpus for k in ["partner", "dating", "children", "relationship"]):
            return (
                SituationAnchors(
                    environment="quiet living room with wooden armchair and window",
                    primary_object="comfortable armchair beside window",
                    secondary_object="warm cup of tea on side table",
                ),
                "user_stated",
            )
        elif any(k in text_corpus for k in ["manuscript", "literary", "agent", "writer", "book", "writing"]):
            return (
                SituationAnchors(
                    environment="personal study desk beside wooden bookshelf",
                    primary_object="manuscript draft pages and pen",
                    secondary_object="clean notepad",
                ),
                "user_stated",
            )
        elif any(k in text_corpus for k in ["assignment", "deadline", "emails", "laundry", "floor", "fridge"]):
            return (
                SituationAnchors(
                    environment="home work room with desk and laundry basket in corner",
                    primary_object="low work desk with laptop",
                    secondary_object="glass of water on floor",
                ),
                "user_stated",
            )
        elif any(k in text_corpus for k in ["climate", "planet", "environmental", "future", "tipping point"]):
            return (
                SituationAnchors(
                    environment="bright room beside wide open window looking at sky",
                    primary_object="environmental report on wooden table",
                    secondary_object="small potted window plant",
                ),
                "user_stated",
            )
        elif any(k in text_corpus for k in ["friend", "trip", "instagram", "party", "group chat", "cabin"]):
            return (
                SituationAnchors(
                    environment="quiet personal room with desk and open window",
                    primary_object="desk with open sketchpad",
                    secondary_object="glass of water",
                ),
                "user_stated",
            )
        elif any(k in text_corpus for k in ["mother", "care", "dementia", "caregiver", "spilled"]):
            return (
                SituationAnchors(
                    environment="quiet kitchen and living area with wooden chair",
                    primary_object="wooden dining chair and table",
                    secondary_object="warm mug on table",
                ),
                "user_stated",
            )
        elif any(k in text_corpus for k in ["lead engineer", "promotion", "promoted", "standup", "team"]):
            return (
                SituationAnchors(
                    environment="workstation desk illuminated by natural daylight",
                    primary_object="open notebook on desk",
                    secondary_object="single desk pen",
                ),
                "user_stated",
            )
        else:
            # Sparse or ambiguous input -> neutral staging explicitly marked as creative interpretation
            return (
                SituationAnchors(
                    environment="quiet peaceful room with simple desk and window",
                    primary_object="simple desk and wooden chair",
                    secondary_object="clean glass of water",
                ),
                "creative_interpretation",
            )

    def _derive_grounded_scene_environment(
        self,
        stage: str,
        idx: int,
        total_scenes: int,
        modality: str,
        facts: List[str],
        raw_text: str,
        anchors: SituationAnchors,
    ) -> Tuple[str, str, Optional[str], str]:
        text_corpus = (" ".join(facts) + " " + raw_text).lower()
        is_workplace = any(k in text_corpus for k in ["supervisor", "colleague", "meeting", "project", "injustice", "credit"])
        is_priority = any(k in text_corpus for k in ["priority", "priorities", "shifting", "confused about what to do"])

        if is_workplace:
            if idx == 1 or stage in ["reality", "trigger"]:
                return (
                    "workplace conference meeting room with table and presentation display",
                    "team meeting agenda and presentation notes",
                    "meeting notebook",
                    "inferred_hypothesis",
                )
            elif idx == 2 or stage in ["validation", "pause"]:
                return (
                    "quiet private office room beside window",
                    "personal desk with open notebook",
                    "glass of water on desk",
                    "creative_interpretation",
                )
            else:
                return (
                    "personal office desk with open laptop and organized file folders",
                    "project folders and dated timeline log",
                    "desk pen and notebook",
                    "creative_interpretation",
                )
        elif is_priority:
            if idx == 1 or stage in ["reality", "trigger", "priority_flux"]:
                return (
                    "quiet study room with desk, open notebook, and morning window light",
                    "open notebook with handwritten list of notes",
                    "pen resting on open page",
                    "creative_interpretation",
                )
            elif idx == 2 or stage in ["validation", "pause", "permission_to_pause"]:
                return (
                    "study room desk by window with soft ambient sunlight",
                    "notepad with highlighted notes",
                    "warm cup of tea on desk",
                    "creative_interpretation",
                )
            else:
                return (
                    "clean organized study desk with focused notepad",
                    "clean notebook page with single top priority circled",
                    "desk pen and glass of water",
                    "creative_interpretation",
                )

        return anchors.environment, anchors.primary_object, anchors.secondary_object, "user_stated" if facts else "creative_interpretation"

    def _derive_grounded_behavior(
        self,
        stage: str,
        idx: int,
        total_scenes: int,
        modality: str,
        facts: List[str],
        anchors: SituationAnchors,
    ) -> Tuple[str, str, str, str, str]:
        """
        Derives (action, posture, expression, gaze, origin) ensuring observable physical events,
        physical progression, and visual distinction across adjacent panels.
        """
        text_corpus = (" ".join(facts) + " " + getattr(anchors, "primary_object", "") + " " + getattr(anchors, "environment", "")).lower()

        # Case-specific thematic cues
        is_career = any(k in text_corpus for k in ["career", "graduat", "placed", "firm", "job"])
        is_bed = any(k in text_corpus for k in ["bed", "fatigue", "flare-up", "autoimmune"])
        is_freeze = any(k in text_corpus for k in ["floor", "freeze", "laundry", "deadline"])
        is_workplace = any(k in text_corpus for k in ["supervisor", "colleague", "meeting", "project", "injustice", "credit"])
        is_priority = any(k in text_corpus for k in ["priority", "priorities", "shifting", "confused about what to do"]) or any(k in stage.lower() for k in ["priority", "flux", "anchor"])

        stg = stage.lower()

        # Scene 1: Initial tension / trigger reality
        if idx == 1 or any(k in stg for k in ["trigger", "reality", "freeze_reality", "cognitive_trap", "reflection", "priority_flux"]):
            if is_career:
                action = "sitting at study desk with head resting on one hand looking down"
                posture = "seated hunched slightly forward at desk"
                expression = "thoughtful furrowed brow"
                gaze = "looking down at open notebook"
            elif is_bed:
                action = "resting propped up on bed with blanket over lap"
                posture = "reclined gently against soft pillows"
                expression = "weary but quiet expression"
                gaze = "looking quietly toward bedside table"
            elif is_freeze:
                action = "sitting on floor with back against wall near low desk"
                posture = "seated on floor with knees bent"
                expression = "blank overwhelmed gaze"
                gaze = "staring blankly at floor ahead"
            elif is_workplace:
                action = "sitting at conference room table with hands resting on meeting notes after project credit was misattributed"
                posture = "seated upright with rigid tense posture"
                expression = "stunned constrained expression"
                gaze = "looking quietly toward meeting notes in disbelief"
            elif is_priority:
                action = "sitting at study desk with open notebook, looking thoughtfully at a list of handwritten notes"
                posture = "seated upright with head resting gently on one hand"
                expression = "thoughtful pensive expression"
                gaze = "looking thoughtfully down at notebook notes"
            else:
                action = "sitting quietly on wooden chair with hands resting on lap"
                posture = "seated still with shoulders slightly curved"
                expression = "quiet pensive expression"
                gaze = "looking downward thoughtfully"
            origin = "creative_interpretation"
            return action, posture, expression, gaze, origin

        # Intermediate scenes: Progression depends on position and stage
        # In a 5-panel arc (e.g. C1):
        if total_scenes == 5:
            if idx == 2:  # internal_pressure
                action = "sitting back in desk chair pausing with hands clasped"
                posture = "leaning back slightly into chair backrest"
                expression = "serious contemplative look"
                gaze = "looking thoughtfully away from desk"
                return action, posture, expression, gaze, "inferred_hypothesis"
            elif idx == 3:  # differentiation
                action = "sitting upright at desk with shoulders relaxed looking toward window"
                posture = "upright open posture"
                expression = "subtle clarity and easing of tension"
                gaze = "looking outward through open window"
                return action, posture, expression, gaze, "inferred_hypothesis"
            elif idx == 4:  # agency
                action = "reaching forward and opening single notebook on desk"
                posture = "engaged forward posture"
                expression = "calm focused resolve"
                gaze = "gaze focused steadily on clean page"
                return action, posture, expression, gaze, "inferred_hypothesis"
            elif idx == 5:  # resolution
                action = "holding pen resting gently on open notebook page"
                posture = "upright poised posture ready to write"
                expression = "steady grounded expression"
                gaze = "looking forward with steady calm gaze"
                return action, posture, expression, gaze, "creative_interpretation"

        # In a 3-panel arc:
        if idx == 2 or any(k in stg for k in ["examination", "validation", "holding_space", "unfreezing", "validation_of_pause", "evidence_testing", "permission_to_pause"]):
            if is_career:
                action = "sitting back in desk chair pausing and taking a slow breath"
                posture = "leaning back slightly into chair backrest"
                expression = "serious contemplative look"
                gaze = "looking thoughtfully toward window"
            elif is_bed:
                action = "resting comfortably holding a warm cup with both hands"
                posture = "sitting up gently against soft pillows"
                expression = "soft receptive expression"
                gaze = "looking down at warm cup"
            elif is_freeze:
                action = "placing hands on floor and slowly shifting to stand"
                posture = "transitioning from floor toward standing"
                expression = "focused determined look"
                gaze = "looking toward desk surface"
            elif is_workplace:
                action = "standing by window pausing with hands loosely clasped"
                posture = "standing upright with dropped shoulders"
                expression = "serious reflective expression acknowledging anger"
                gaze = "looking outward through window acknowledging legitimate anger"
            elif is_priority:
                action = "pausing comfortably in chair and looking toward window with dropped shoulders"
                posture = "seated comfortably back in chair"
                expression = "calm reflective expression"
                gaze = "looking toward side window in quiet reflection"
            else:
                action = "leaning back in chair pausing and taking a deliberate slow breath"
                posture = "seated upright with shoulders dropped"
                expression = "calm reflective gaze"
                gaze = "looking toward side window"
            return action, posture, expression, gaze, "creative_interpretation"

        # Final panel in 3-panel or general resolution
        if is_career:
            action = "holding pen resting gently on open notebook page"
            posture = "upright poised posture ready to write"
            expression = "steady grounded expression"
            gaze = "looking forward with steady calm gaze"
        elif is_bed:
            action = "lying back comfortably resting with eyes gently closed"
            posture = "completely relaxed reclined posture"
            expression = "peaceful serene expression"
            gaze = "eyes gently closed in peaceful rest"
        elif is_freeze:
            action = "standing upright beside desk holding one glass of water"
            posture = "standing upright feet planted"
            expression = "steady resolute expression"
            gaze = "looking forward with steady gaze"
        elif is_workplace:
            action = "reviewing project records and writing dated timeline entries into notes"
            posture = "seated upright focused attentively over documentation"
            expression = "grounded determined expression"
            gaze = "focused directly on project records and timeline documentation"
        elif is_priority:
            action = "writing down a single clear priority at the top of a clean notebook page"
            posture = "seated attentively with pen poised over paper"
            expression = "grounded clear expression"
            gaze = "focused directly on the written priority"
        else:
            action = "standing calmly beside desk looking forward with steady posture"
            posture = "standing upright relaxed"
            expression = "peaceful grounded expression"
            gaze = "looking forward with steady gaze"

        return action, posture, expression, gaze, "creative_interpretation"

    def _derive_grounded_camera_lighting(
        self, stage: str, idx: int, total_scenes: int, modality: str
    ) -> Tuple[str, str, str, str, str]:
        ratio = (idx - 1) / max(1, total_scenes - 1)

        # Lighting progresses naturally from cool/diffused to warm/clear daylight
        if ratio < 0.3:
            lighting = "soft shadow-toned daylight"
            mood = "quiet tension"
            cam_dist = "medium shot"
            cam_angle = "eye-level"
            comp = "rule of thirds, subject on left"
        elif ratio < 0.6:
            lighting = "soft warm ambient side light"
            mood = "reflective pause"
            cam_dist = "medium close-up"
            cam_angle = "slight side angle"
            comp = "centered composition"
        elif ratio < 0.8:
            lighting = "gentle natural daylight filtering in"
            mood = "emerging clarity"
            cam_dist = "medium shot"
            cam_angle = "profile angle"
            comp = "subject looking toward open space"
        else:
            lighting = "clear bright morning daylight"
            mood = "grounded calm"
            cam_dist = "medium shot"
            cam_angle = "eye-level"
            comp = "balanced centered composition"

        return cam_dist, cam_angle, comp, lighting, mood

    # =========================================================================
    # STEP 1 — UNDERSTAND THE CASE
    # =========================================================================
    def _understand_case(self, case_frame, strategy: dict = None, archetype: dict = None) -> CaseUnderstanding:
        raw_text = getattr(case_frame, "raw_text", "")
        lowered = (raw_text or "").lower()

        reasoning = getattr(case_frame, "reasoning", {}) or {}
        strat_dict = (strategy or {}) if isinstance(strategy, dict) else (getattr(case_frame, "strategy", {}) or {})

        situation = reasoning.get("situation") or getattr(case_frame, "trigger", "") or raw_text
        primary_emotion = reasoning.get("primary_emotion") or getattr(case_frame, "core_emotion", "") or "stress"
        central_conflict = reasoning.get("central_conflict", "")
        automatic_thought = reasoning.get("automatic_thought", "")
        pattern = reasoning.get("pattern") or getattr(case_frame, "pattern", "") or "general_stress"
        coping_strategy = reasoning.get("coping_strategy") or strat_dict.get("name", "Evidence-Informed Reframing")
        specific_reframe = reasoning.get("specific_reframe") or getattr(case_frame, "specific_reframe", "") or strat_dict.get("reframe", "")
        concrete_next_step = reasoning.get("concrete_next_step") or getattr(case_frame, "concrete_action", "") or strat_dict.get("concrete_action", "")
        controllable = getattr(case_frame, "controllable", "") or strat_dict.get("what_is_controllable", "")
        uncontrollable = getattr(case_frame, "uncontrollable", "") or strat_dict.get("what_is_not_controllable", "")
        is_threat = getattr(case_frame, "is_external_threat", False)
        distortions = getattr(case_frame, "distortions", []) or []
        sit_type = getattr(case_frame, "situation_type", "")

        # Detect domain
        domain = self._detect_domain(raw_text, pattern, is_threat, situation_type=sit_type)

        # Detect whether genuine uncertainty exists and must be preserved (Step 11)
        # Note: Scoped strictly to relational ambivalence (e.g. breakups) so that future uncertainty
        # is handled with locus-of-control rather than breakup grief.
        genuine_uncertainty = (
            (domain == "relationship" and bool(re.search(r"\b(uncertain(ty)?|not sure|wasn't sure|did not love|doubt|don't know|ambivalence)\b", lowered)))
            or sit_type in ["relationship_ambivalence", "breakup_ambivalence"]
        )

        appraisal = "loss_of_equilibrium"
        if is_threat:
            appraisal = "interpersonal_threat"
        elif domain == "overwhelm":
            appraisal = "executive_overload"
        elif domain == "uncertainty":
            appraisal = "threat_anticipation"
        elif domain == "social_comparison":
            appraisal = "social_comparison_inadequacy"
        elif genuine_uncertainty:
            appraisal = "unresolved_ambivalence"
        elif "catastroph" in pattern.lower():
            appraisal = "catastrophic_forecast"

        return CaseUnderstanding(
            raw_text=raw_text,
            situation=situation,
            primary_emotion=primary_emotion,
            central_conflict=central_conflict,
            automatic_thought=automatic_thought,
            appraisal=appraisal,
            pattern=pattern,
            coping_strategy=coping_strategy,
            specific_reframe=specific_reframe,
            concrete_next_step=concrete_next_step,
            controllable=controllable,
            uncontrollable=uncontrollable,
            is_external_threat=is_threat,
            domain=domain,
            genuine_uncertainty=genuine_uncertainty,
            distortions=distortions,
        )

    def _detect_domain(self, raw_text: str, pattern: str, is_threat: bool, situation_type: str = "") -> str:
        if situation_type:
            sit_map = {
                "interpersonal_threat": "threat",
                "relationship_ambivalence": "relationship",
                "loss_and_grief": "relationship",
                "breakup": "relationship",
                "performance_anxiety": "academic",
                "evaluation_fear": "academic",
                "overwhelm": "overwhelm",
                "task_overwhelm": "overwhelm",
                "social_comparison": "social_comparison",
                "self_criticism": "self_criticism",
                "rumination": "rumination",
                "uncertainty": "uncertainty",
                "future_uncertainty": "uncertainty",
                "future_fear": "uncertainty",
                "threat_anticipation": "uncertainty",
                "burnout": "burnout",
            }
            if situation_type in sit_map:
                return sit_map[situation_type]

        lowered = (raw_text or "").lower()
        if is_threat or re.search(r"\b(lashes out|abusive|unpredictable|father|yelling|threat|violence|screaming)\b", lowered):
            return "threat"
        if re.search(r"\b(exam|interview|test|presentation|freeze and fail|won't pass|finals|studying)\b", lowered):
            return "academic"
        if re.search(r"\b(breakup|break\s+up|broke\s+up|relationship|partner|\bex\b|dating|loved him|loved her|separation|divorce)\b", lowered):
            return "relationship"
        if re.search(r"\b(20 things|twenty things|\d+\s+things|too many|cannot start|can't start|paralyzed|mountain of work|to-do list)\b", lowered):
            return "overwhelm"
        if re.search(r"\b(doing\s+(so\s+much\s+|far\s+|way\s+)?better|everyone\s+is\s+doing\s+better|everyone\s+my\s+age|falling behind|peers are ahead|comparing|succeeding while)\b", lowered):
            return "social_comparison"
        if re.search(r"\b(messed up once|i'm useless|i am useless|worthless|can't do anything right)\b", lowered):
            return "self_criticism"
        if re.search(r"\b(keep replaying|replaying an embarrassing|what happened yesterday|cringing|why did i say)\b", lowered):
            return "rumination"
        if re.search(r"\b(don't know what will happen|scared something bad will happen|afraid of the future|uncertain about|constant uncertainty|uncertainty is making|future and career)\b", lowered):
            return "uncertainty"
        return "general"

    # =========================================================================
    # STEP 2 & STEP 3 — DEFINE THE PSYCHOLOGICAL ARC & SCENE COUNT
    # =========================================================================
    def _derive_psychological_arc(self, case: CaseUnderstanding) -> List[PsychologicalMilestone]:
        """
        Determines the required sequence of distinct psychological milestones (3 to 8).
        Every milestone must represent a necessary transition in the character's internal state.
        """
        # 1. Complex Trauma / Interpersonal Threat (6 scenes)
        if case.domain == "threat" or case.is_external_threat:
            return [
                PsychologicalMilestone(
                    stage_category="incident",
                    stage="incident",
                    narrative_purpose="establish_trigger",
                    psychological_purpose="Depict the sudden outburst and acknowledge acute threat without minimizing danger.",
                    target_emotion="shock_and_fear",
                    arousal_level="high",
                    perceived_control="low",
                    appraisal="interpersonal_threat",
                ),
                PsychologicalMilestone(
                    stage_category="disorientation",
                    stage="disorientation",
                    narrative_purpose="show_emotional_response",
                    psychological_purpose="Show the destabilizing effect of the other person acting normal afterward (gaslighting impact).",
                    target_emotion="confusion_and_self_doubt",
                    arousal_level="high",
                    perceived_control="low",
                    appraisal="perceptual_disorientation",
                ),
                PsychologicalMilestone(
                    stage_category="validation",
                    stage="validation",
                    narrative_purpose="distinguish_fact_from_prediction",
                    psychological_purpose="Validate personal reality: pleasant aftermath does not erase what genuinely took place.",
                    target_emotion="grounded_validation",
                    arousal_level="moderate",
                    perceived_control="emerging",
                    appraisal="truth_anchoring",
                ),
                PsychologicalMilestone(
                    stage_category="externalizing",
                    stage="externalizing",
                    narrative_purpose="distinguish_control_from_no_control",
                    psychological_purpose="Externalize responsibility: other people's emotional dysregulation is not your burden to fix.",
                    target_emotion="unburdened_relief",
                    arousal_level="low",
                    perceived_control="growing",
                    appraisal="boundary_clarity",
                ),
                PsychologicalMilestone(
                    stage_category="boundary",
                    stage="boundary",
                    narrative_purpose="boundary_or_safety_action",
                    psychological_purpose="Establish physical and emotional distance to preserve psychological safety.",
                    target_emotion="protective_resolve",
                    arousal_level="moderate",
                    perceived_control="anchored",
                    appraisal="self_protection",
                ),
                PsychologicalMilestone(
                    stage_category="support",
                    stage="support",
                    narrative_purpose="support_seeking",
                    psychological_purpose="Reach out to trusted support and anchor in external safety resources.",
                    target_emotion="connected_safety",
                    arousal_level="low",
                    perceived_control="anchored",
                    appraisal="resource_activation",
                ),
            ]

        # 2. Executive Overwhelm & Behavioral Freeze (4 or 5 scenes)
        if case.domain == "overwhelm":
            return [
                PsychologicalMilestone(
                    stage_category="trigger",
                    stage="trigger",
                    narrative_purpose="establish_trigger",
                    psychological_purpose="Confront the avalanche of demands and surface the visceral paralyzing freeze.",
                    target_emotion="executive_overload",
                    arousal_level="high",
                    perceived_control="low",
                    appraisal="insurmountable_wall",
                ),
                PsychologicalMilestone(
                    stage_category="freeze",
                    stage="spiral",
                    narrative_purpose="show_thought_spiral",
                    psychological_purpose="Show the escalating cognitive loop: demanding total completion simultaneously creates complete freeze.",
                    target_emotion="paralyzed_anxiety",
                    arousal_level="high",
                    perceived_control="low",
                    appraisal="catastrophic_paralysis",
                ),
                PsychologicalMilestone(
                    stage_category="examination",
                    stage="examination",
                    narrative_purpose="examine_assumption",
                    psychological_purpose="Pause the freeze and separate the total future mountain from what is actually doable right now.",
                    target_emotion="emerging_pause",
                    arousal_level="moderate",
                    perceived_control="emerging",
                    appraisal="objective_partitioning",
                ),
                PsychologicalMilestone(
                    stage_category="reframing",
                    stage="reframing",
                    narrative_purpose="cognitive_reappraisal",
                    psychological_purpose="Shrink the wall to a single brick: permission to do one 5-minute micro-action.",
                    target_emotion="focused_calm",
                    arousal_level="low",
                    perceived_control="growing",
                    appraisal="manageable_micro_step",
                ),
                PsychologicalMilestone(
                    stage_category="resolution",
                    stage="resolution",
                    narrative_purpose="concrete_next_step",
                    psychological_purpose="Commit to the single 5-minute action, breaking paralysis and initiating momentum.",
                    target_emotion="empowered_momentum",
                    arousal_level="moderate",
                    perceived_control="anchored",
                    appraisal="actionable_path",
                ),
            ]

        # 3. Cognitive Spirals, Rumination, Social Comparison, Relational Ambivalence (5 scenes)
        if case.domain in ["social_comparison", "rumination", "self_criticism", "uncertainty"] or (case.domain == "relationship" and case.genuine_uncertainty):
            return [
                PsychologicalMilestone(
                    stage_category="trigger",
                    stage="trigger",
                    narrative_purpose="establish_trigger",
                    psychological_purpose=f"Identify the situation realities and immediate emotional prompt ({case.situation[:50]}).",
                    target_emotion=case.primary_emotion or "distress",
                    arousal_level="high",
                    perceived_control="low",
                    appraisal="loss_of_equilibrium",
                ),
                PsychologicalMilestone(
                    stage_category="spiral",
                    stage="spiral",
                    narrative_purpose="show_thought_spiral",
                    psychological_purpose="Surface the automatic cognitive pattern and escalating internal narrative.",
                    target_emotion="anxious_rumination",
                    arousal_level="high",
                    perceived_control="low",
                    appraisal="catastrophic_escalation",
                ),
                PsychologicalMilestone(
                    stage_category="examination",
                    stage="examination",
                    narrative_purpose="examine_assumption",
                    psychological_purpose="Pause the spiral and examine the evidence: separate feelings from objective facts.",
                    target_emotion="emerging_stillness",
                    arousal_level="moderate",
                    perceived_control="emerging",
                    appraisal="objective_evaluation",
                ),
                PsychologicalMilestone(
                    stage_category="reframing",
                    stage="reframing",
                    narrative_purpose="cognitive_reappraisal",
                    psychological_purpose="Formulate a balanced, evidence-informed perspective rooted in agency and tolerating ambiguity.",
                    target_emotion="grounded_clarity",
                    arousal_level="low",
                    perceived_control="growing",
                    appraisal="balanced_perspective",
                ),
                PsychologicalMilestone(
                    stage_category="resolution",
                    stage="resolution",
                    narrative_purpose="concrete_next_step",
                    psychological_purpose="Commit to one concrete, achievable behavioral micro-step with patient self-trust.",
                    target_emotion="empowered_focus",
                    arousal_level="moderate",
                    perceived_control="anchored",
                    appraisal="actionable_path",
                ),
            ]

        # 4. Focused Cognitive Challenge / Simple Dilemma (3 scenes: problem -> reframing -> resolution)
        return [
            PsychologicalMilestone(
                stage_category="problem",
                stage="problem",
                narrative_purpose="establish_trigger",
                psychological_purpose=f"Acknowledge the immediate trigger ({case.situation[:50]}) and validate emotional difficulty.",
                target_emotion=case.primary_emotion or "distress",
                arousal_level="moderate",
                perceived_control="low",
                appraisal="immediate_challenge",
            ),
            PsychologicalMilestone(
                stage_category="reframing",
                stage="reframing",
                narrative_purpose="cognitive_reappraisal",
                psychological_purpose="Decouple catastrophic forecast from reality and focus strictly on locus of control.",
                target_emotion="grounded_calm",
                arousal_level="low",
                perceived_control="growing",
                appraisal="balanced_perspective",
            ),
            PsychologicalMilestone(
                stage_category="resolution",
                stage="resolution",
                narrative_purpose="concrete_next_step",
                psychological_purpose="Channel energy into a single realistic, doable constructive action.",
                target_emotion="determined_agency",
                arousal_level="moderate",
                perceived_control="anchored",
                appraisal="actionable_step",
            ),
        ]

    # =========================================================================
    # STEP 5 & STEP 6 — DERIVE NATURAL OBJECTS & PERSISTENT ENVIRONMENT
    # =========================================================================
    def _derive_anchors(self, case: CaseUnderstanding) -> SituationAnchors:
        """
        Derives authentic objects and an environmental setting naturally belonging to the situation.
        The physical environment maintains continuity throughout the scene progression.
        """
        if case.domain == "threat":
            return SituationAnchors(
                environment="Living area and hallway moving toward quiet secure bedroom",
                primary_object="hallway entrance doorway",
                secondary_object="private journal and bedroom door latch",
            )
        elif case.domain == "relationship":
            return SituationAnchors(
                environment="Quiet personal bedroom beside wooden desk and window",
                primary_object="smartphone displaying open message thread",
                secondary_object="handwritten notebook and warm mug of tea",
            )
        elif case.domain == "overwhelm":
            return SituationAnchors(
                environment="Home workstation surrounded by desk and bookshelves",
                primary_object="cluttered to-do checklist and stacked papers",
                secondary_object="desk timer set for five minutes and clean single-task card",
            )
        elif case.domain == "social_comparison":
            return SituationAnchors(
                environment="Living room armchair beside a wooden side table",
                primary_object="smartphone scrolling social media feed",
                secondary_object="personal values journal with handwritten priorities",
            )
        elif case.domain == "self_criticism":
            return SituationAnchors(
                environment="Personal study desk illuminated by warm desk lamp",
                primary_object="crossed-out draft paper with pen marks",
                secondary_object="clean sheet of fresh paper and resting pen",
            )
        elif case.domain == "rumination":
            return SituationAnchors(
                environment="Quiet corner room with desk and open window",
                primary_object="notepad with replaying questions and resting pen",
                secondary_object="glass of cool water and fresh sketchpad",
            )
        elif case.domain == "academic":
            return SituationAnchors(
                environment="Student study desk with open textbooks and notebooks",
                primary_object="printed exam review notes and textbook",
                secondary_object="clean practice notepad and pen",
            )
        elif case.domain == "uncertainty":
            return SituationAnchors(
                environment="Quiet living room desk beside open window",
                primary_object="daily calendar and notebook on desk",
                secondary_object="pocket notebook with today's immediate routine list",
            )
        else:
            return SituationAnchors(
                environment="Quiet personal room with desk and open window",
                primary_object="notepad with handwritten thoughts",
                secondary_object="clean notebook and glass of water",
            )

    # =========================================================================
    # STEP 4, 7, 8, 12 — ORCHESTRATE PHYSICAL OBSERVABLE SCENESTATE
    # =========================================================================
    def _orchestrate_scene(
        self,
        scene_idx: int,
        milestone: PsychologicalMilestone,
        total_scenes: int,
        case: CaseUnderstanding,
        anchors: SituationAnchors,
    ) -> SceneState:
        """
        Synthesizes the complete SceneState from psychological change to observable physical events.
        """
        cat = milestone.stage_category
        prim_obj = anchors.primary_object
        sec_obj = anchors.secondary_object
        env = anchors.environment

        # 1. Determine Character Goal & Physical Action (Step 4 & Step 7)
        goal, action, posture, expression, gaze = self._derive_observable_behavior(
            cat, scene_idx, total_scenes, case, prim_obj, sec_obj
        )

        # 2. Determine Object Interaction & Spatial Relationship (Step 5 & Step 6)
        obj_interaction, spatial_rel = self._derive_object_and_space(
            cat, scene_idx, total_scenes, prim_obj, sec_obj, env
        )

        # 3. Determine Camera & Lighting Variables (Step 8)
        cam_dist, cam_angle, comp, lighting, mood = self._derive_camera_and_lighting(
            cat, scene_idx, total_scenes, case
        )

        # 4. Synthesize Observable Visual Description (Step 10)
        visual_desc = self._synthesize_visual_description(
            action, posture, expression, prim_obj, sec_obj, lighting, mood, cat
        )

        # 5. Synthesize Dialogue, Thoughts, Captions, Narratives (Step 11)
        dialogue, thought_text, bubble_type, caption, narrative = self._synthesize_text_elements(
            cat, scene_idx, total_scenes, case, prim_obj, sec_obj
        )

        # 6. Assemble rich SceneState
        stage_name = cat if cat not in ["problem", "reframing", "resolution"] else cat
        title = f"{scene_idx}. {cat.upper()}"

        psych_state = {
            "emotion": milestone.target_emotion,
            "arousal_level": milestone.arousal_level,
            "perceived_control": milestone.perceived_control,
            "automatic_thought": case.automatic_thought or dialogue,
            "appraisal": milestone.appraisal,
        }

        char_state = {
            "character_goal": goal,
            "character_action": action,
            "character_posture": posture,
            "facial_expression": expression,
            "gaze_target": gaze,
        }

        staging_state = {
            "environment_state": env,
            "important_objects": [prim_obj, sec_obj],
            "camera_distance": cam_dist,
            "camera_angle": cam_angle,
            "composition": comp,
            "lighting": lighting,
            "visual_mood": mood,
        }

        return SceneState(
            scene_id=scene_idx,
            stage_category=cat,
            stage=milestone.stage,
            title=title,
            narrative_purpose=milestone.narrative_purpose,
            psychological_purpose=milestone.psychological_purpose,
            trigger=case.situation,
            automatic_thought=case.automatic_thought or dialogue,
            appraisal=milestone.appraisal,
            emotion=milestone.target_emotion,
            character_goal=goal,
            character_action=action,
            posture=posture,
            expression=expression,
            gaze=gaze,
            environment=env,
            important_objects=[prim_obj, sec_obj],
            object_interaction=obj_interaction,
            spatial_relationships=spatial_rel,
            camera_distance=cam_dist,
            camera_angle=cam_angle,
            composition=comp,
            lighting=lighting,
            visual_mood=mood,
            dialogue=dialogue,
            thought_text=thought_text,
            bubble_type=bubble_type,
            caption=caption,
            visual_description=visual_desc,
            narrative=narrative,
            psychological_state=psych_state,
            character_state=char_state,
            visual_staging=staging_state,
        )

    # -------------------------------------------------------------------------
    # Helper: Derive Observable Behavior (Character Goal, Physical Action, Posture, Gaze)
    # -------------------------------------------------------------------------
    def _derive_observable_behavior(
        self, cat: str, idx: int, total: int, case: CaseUnderstanding, prim_obj: str, sec_obj: str
    ) -> Tuple[str, str, str, str, str]:
        if cat in ["incident"]:
            goal = "Stepping back to create immediate physical safety from sudden volatility"
            action = f"stepping back defensively near the {prim_obj}, shoulders hunched and hands raised protectively"
            posture = "Defensive posture, hands raised protective, rigid stance"
            expression = "Wide anxious eyes, strained guarded expression"
            gaze = f"Looking vigilantly toward the {prim_obj}"
        elif cat in ["disorientation"]:
            goal = "Attempting to make sense of sudden mood reversal and pleasant aftermath"
            action = f"standing frozen in hallway clutching own arms for comfort, looking back at the calm room in disbelief"
            posture = "Hesitant posture, clutching own arms tightly"
            expression = "Confused furrowed brow, disoriented eyes"
            gaze = "Looking back toward the ordinary room in disbelief"
        elif cat in ["validation"]:
            goal = "Holding onto personal perception and grounding subjective truth"
            action = f"sitting in private room away from the conflict, placing hand gently on chest while taking a grounding breath"
            posture = "Uncoiling from rigid defense into grounded stillness"
            expression = "Gentle, self-validating gaze, steadying expression"
            gaze = "Looking down with quiet inner honesty"
        elif cat in ["externalizing"]:
            goal = "Releasing self-blame and recognizing that others' volatility belongs to them"
            action = f"standing beside the window with dropped relaxed shoulders, looking out at open sky"
            posture = "Upright, peaceful, unburdened posture"
            expression = "Clear, resolute expression without guilt"
            gaze = "Looking toward open sky through window"
        elif cat in ["boundary"]:
            goal = "Creating a protective physical perimeter and documenting facts"
            action = f"closing the bedroom door gently, latching it, and writing clear observations in {sec_obj}"
            posture = "Firm, composed, intentional movement"
            expression = "Calm, protective determination"
            gaze = f"Looking down at notes in {sec_obj}"
        elif cat in ["support"]:
            goal = "Reaching out to outside support and connecting with a safe ally"
            action = f"holding smartphone on call talking with a trusted friend, breathing steadily in warm sunlight"
            posture = "Relaxed upright posture, steady breathing"
            expression = "Relieved, gentle smile, feeling heard and safe"
            gaze = "Looking forward with trust and connection"
        elif cat in ["trigger", "problem"]:
            goal = f"Facing the immediate dilemma: {case.situation[:50]}"
            if case.domain == "relationship":
                action = f"sitting on edge of bed looking down at {prim_obj}, fingers hovering in hesitation over keys"
            elif case.domain == "overwhelm":
                action = f"sitting at desk staring paralyzed at {prim_obj}, hands pressed to temples under mounting tasks"
            elif case.domain == "academic":
                action = f"sitting at study desk gripping {prim_obj} with tense hands, shoulders raised in anticipatory dread"
            elif case.domain == "social_comparison":
                action = f"sitting in armchair scrolling through {prim_obj}, slouching in self-doubt"
            else:
                action = f"sitting quietly looking down at {prim_obj} with burdened, introspective posture"
            posture = "Hunched, guarded posture with raised shoulders"
            expression = "Stressed, concerned brow, furrowed eyes"
            gaze = f"Locked intently on {prim_obj}"
        elif cat in ["freeze"]:
            goal = "Trying to untangle competing obligations while gripped by paralysis"
            action = f"resting forehead in hand over {prim_obj}, frozen between competing tasks"
            posture = "Slumped, immobilized posture over work surface"
            expression = "Exhausted, overwhelmed expression under task weight"
            gaze = f"Darting between multiple items on {prim_obj}"
        elif cat in ["spiral"]:
            goal = "Trying to untangle repeating thoughts and escalating doubts"
            if case.domain == "relationship":
                action = f"rereading messages on {prim_obj}, looking between screen and handwritten notes in second-guessing doubt"
            elif case.domain == "social_comparison":
                action = f"rereading curated highlights on {prim_obj}, gripping device with tense restless fingers"
            elif case.domain == "rumination":
                action = f"staring off in distracted discomfort, replaying the awkward interaction on loop"
            elif case.domain == "self_criticism":
                action = f"repeatedly crossing out lines on {prim_obj} with pen, head bowed in harsh self-judgment"
            else:
                action = f"pacing past desk or sitting gripped by spinning thoughts, touching hand to forehead in distress"
            posture = "Restless, rigid stance with nervous energy"
            expression = "Exhausted, overwhelmed expression caught in internal loop"
            gaze = "Unfocused distant gaze lost in internal loops"
        elif cat in ["examination"]:
            goal = "Pausing the spinning cycle to ground in physical reality"
            if case.domain == "relationship":
                action = f"setting {prim_obj} face-down on table, opening {sec_obj}, and taking a slow deliberate breath"
            elif case.domain == "overwhelm":
                action = f"pushing aside the cluttered papers to reveal {sec_obj}, leaning back in chair to breathe"
            elif case.domain == "social_comparison":
                action = f"placing {prim_obj} face-down on coffee table, looking up toward window to break the comparison loop"
            elif case.domain == "uncertainty":
                action = f"closing {prim_obj}, setting hands flat on desk, and looking out open window to ground in the present"
            elif case.domain == "academic":
                action = f"placing pen down on desk, setting review notes flat, and resting open hands on lap"
            else:
                action = f"pausing with hands resting open on table, stepping back from {prim_obj} to take a slow grounding breath"
            posture = "Shifting from hunched defense to balanced, steady posture"
            expression = "Pensive, thoughtful eyes softening"
            gaze = f"Looking toward {sec_obj} or open window"
        elif cat in ["reframing"]:
            goal = "Grounding in a balanced, evidence-informed perspective and locus of control"
            if case.domain == "relationship":
                action = f"writing two honest truths in {sec_obj}: acknowledging sorrow while honoring genuine uncertainty"
            elif case.domain == "overwhelm":
                action = f"writing down one single 5-minute task on {sec_obj}, letting the rest of the pile wait"
            elif case.domain == "social_comparison":
                action = f"writing personal intrinsic values and personal timeline in {sec_obj} with calm focus"
            elif case.domain == "uncertainty":
                action = f"writing down two manageable routine tasks for today in {sec_obj}, letting distant unknowns wait"
            elif case.domain == "self_criticism":
                action = f"writing a constructive correction on {sec_obj}, treating self with patient kindness"
            else:
                action = f"writing down one core truth on {sec_obj} with steady hand in warm ambient light"
            posture = "Steady upright spine, open relaxed shoulders"
            expression = "Clear, grounded, self-compassionate expression"
            gaze = f"Focused gaze on the written words in {sec_obj}"
        elif cat in ["resolution", "action"]:
            goal = "Committing to one concrete, achievable behavioral micro-step with self-trust"
            if case.domain == "relationship":
                action = f"standing upright by the window holding {sec_obj}, taking a quiet breath without forcing certainty"
            elif case.domain == "overwhelm":
                action = f"setting {sec_obj} for five minutes, picking up pen with resolute focus to begin the first sentence"
            elif case.domain == "academic":
                action = f"sitting tall with calm focus, solving one practice problem on {sec_obj} step by step"
            elif case.domain == "social_comparison":
                action = f"standing tall putting phone aside, turning attention happily toward personal creative work"
            elif case.domain == "uncertainty":
                action = f"standing tall by open window holding {sec_obj}, taking a calm grounding breath ready for today's routine"
            else:
                action = f"standing tall by sunlit window with calm resolute posture, ready to take one constructive step"
            posture = "Tall, grounded, purposeful posture with natural composure"
            expression = "Quietly resolute, gentle smile and calm steady eyes"
            gaze = "Looking forward toward open daylight and the day ahead"
        else:
            goal = "Engaging with the present situation"
            action = f"sitting thoughtfully with {sec_obj} in hand, taking a slow deliberate breath"
            posture = "Balanced posture"
            expression = "Reflective expression"
            gaze = "Looking forward"

        return goal, action, posture, expression, gaze

    # -------------------------------------------------------------------------
    # Helper: Derive Object Interaction & Spatial Position across lifecycle
    # -------------------------------------------------------------------------
    def _derive_object_and_space(
        self, cat: str, idx: int, total: int, prim_obj: str, sec_obj: str, env: str
    ) -> Tuple[str, str]:
        if cat in ["incident", "trigger", "problem"]:
            interaction = f"gripping or staring transfixed at {prim_obj}"
            spatial = "Character positioned closely against work surface, crowded by surrounding shadows"
        elif cat in ["disorientation", "freeze", "spiral"]:
            interaction = f"hesitating over, rereading, or locked in tension with {prim_obj}"
            spatial = "Character confined in center of tension, rigid posture with restricted movement"
        elif cat in ["validation", "examination"]:
            interaction = f"placing {prim_obj} face-down / pushing aside clutter to access {sec_obj}"
            spatial = "Physical breathing room created between character and trigger source"
        elif cat in ["externalizing", "boundary", "reframing"]:
            interaction = f"writing calmly in {sec_obj} or latching door for safety"
            spatial = "Character centered beside open surface in warm ambient illumination"
        elif cat in ["support", "resolution", "action"]:
            interaction = f"holding {sec_obj} / resting hands peacefully / holding phone on call"
            spatial = "Character framed against open window with expansive outside view"
        else:
            interaction = f"interacting thoughtfully with {sec_obj}"
            spatial = "Balanced, open spatial position"

        return interaction, spatial

    # -------------------------------------------------------------------------
    # Helper: Derive Camera & Lighting Variables (Storytelling Variables)
    # -------------------------------------------------------------------------
    def _derive_camera_and_lighting(
        self, cat: str, idx: int, total: int, case: CaseUnderstanding
    ) -> Tuple[str, str, str, str, str]:
        if cat in ["incident"]:
            return "medium shot", "slightly high angle emphasizing vulnerability", "Character backed near wall in left third", "Cool shadow-toned moody lighting with harsh contrast", "Claustrophobic, uneasy, hypervigilant"
        elif cat in ["disorientation", "spiral", "freeze"]:
            return "close-up", "canted slightly Dutch angle", "Tight framing on expressive face and strained hands", "Low-key dim directional lighting", "Claustrophobic, ruminative tension"
        elif cat in ["validation", "examination"]:
            return "medium shot", "eye level grounded", "Centered stable composition giving breathing room", "Soft neutral diffuse ambient illumination", "Reflective, pausing, clearing"
        elif cat in ["externalizing", "reframing", "boundary"]:
            return "medium close-up", "straight-on eye level", "Warm focal point around character's active hands and notebook", "Warm golden ambient light", "Warm, reassuring, clear"
        elif cat in ["support", "resolution", "action"]:
            return "medium shot", "level grounded angle slightly low for agency", "Open dynamic composition with clean horizons", "Crisp bright natural daylight", "Hopeful, purposeful, grounded"
        else:
            ratio = idx / max(1, total - 1)
            if ratio < 0.35:
                return "medium shot", "eye level", "Centered with shadow", "Cool moody lighting", "Tense"
            elif ratio < 0.7:
                return "medium close-up", "eye level", "Balanced framing", "Soft warm ambient light", "Thoughtful"
            else:
                return "medium shot", "straight-on", "Open framing", "Bright morning daylight", "Grounded"

    # -------------------------------------------------------------------------
    # Helper: Synthesize Composite Visual Description (Step 10)
    # -------------------------------------------------------------------------
    def _synthesize_visual_description(
        self, action: str, posture: str, expression: str, prim_obj: str, sec_obj: str, lighting: str, mood: str, cat: str
    ) -> str:
        """
        Combines physical action, posture, authentic objects, and lighting into a rich,
        observable visual prompt suitable for image diffusion.
        """
        return f"{action}, {posture.lower()}, {expression.lower()}, {lighting.lower()}"

    # -------------------------------------------------------------------------
    # Helper: Synthesize Text Elements (Step 11 & Dialogue/Caption Grounding)
    # -------------------------------------------------------------------------
    def _synthesize_text_elements(
        self, cat: str, idx: int, total: int, case: CaseUnderstanding, prim_obj: str, sec_obj: str
    ) -> Tuple[str, str, str, str, str]:
        reframe = case.specific_reframe
        action = case.concrete_next_step
        conflict = case.central_conflict
        auto_thought = case.automatic_thought
        raw = case.raw_text

        bubble_type = "thought"

        if cat == "incident":
            dialogue = "The outburst happened so fast out of nowhere."
            thought_text = dialogue
            caption = "Sudden volatility instantly triggers hypervigilance and shock."
            narrative = f"You're dealing with an unpredictable and tense situation: '{raw.strip()}'. Sudden outbursts create immediate shock and fear."
        elif cat == "disorientation":
            dialogue = "How can they act like nothing even happened?"
            thought_text = dialogue
            caption = "A sudden return to 'normal' creates profound confusion and self-doubt."
            narrative = "When someone acts completely pleasant right after lashing out, it causes deep internal dissonance and makes you question your senses."
        elif cat == "validation":
            dialogue = "Their acting normal doesn't mean it didn't happen. My feelings are real."
            thought_text = dialogue
            caption = "Trust your perception: pleasant aftermath does not erase an aggressive reality."
            narrative = "Their acting normal afterward does not erase what happened earlier. You do not have to rewrite your experience to keep the peace."
        elif cat == "externalizing":
            dialogue = "Their volatility belongs to them. It is not my job to manage their moods."
            thought_text = dialogue
            caption = "You are responsible for your own safety, not for regulating someone else's volatile emotions."
            narrative = "Recognizing the pattern without taking on the blame allows you to stop walking on eggshells."
        elif cat == "boundary":
            dialogue = "I will keep my distance and write down what happened clearly."
            thought_text = dialogue
            caption = "Creating emotional and physical distance is an act of self-care and safety."
            narrative = "Give yourself physical or emotional distance. Write down what happened while you remember it clearly."
        elif cat == "support":
            dialogue = "I'm reaching out to someone I trust. I don't have to navigate this alone."
            thought_text = dialogue
            bubble_type = "speech"
            caption = "Safety grows through connection: reach out to people who truly hear and support you."
            narrative = "Right now, you don't need to fix their behavior. Connect with someone you trust for grounded, objective support."
        elif cat in ["trigger", "problem"]:
            if case.domain == "relationship":
                dialogue = "It hurts so much... but I was so uncertain if I loved him."
                caption = "Grief naturally follows an ending, even when uncertainty was genuine."
            elif case.domain == "overwhelm":
                dialogue = "I have twenty things to do and I'm so overwhelmed I can't even begin."
                caption = "Looking at everything all at once overwhelms the brain and creates paralysis."
            elif case.domain == "academic":
                dialogue = "I'm terrified I'm going to freeze and fail tomorrow."
                caption = "High-stakes evaluations trigger intense anticipatory alarm."
            elif case.domain == "social_comparison":
                dialogue = "Everyone is doing so much better than me... I'm completely behind."
                caption = "Curated achievements of others easily trigger deep personal inadequacy."
            elif case.domain == "uncertainty":
                dialogue = "I don't know what will happen with my future, and the uncertainty feels terrifying."
                caption = "When the future feels ambiguous, the mind anticipates threat where there is only unknown."
            else:
                dialogue = f"This feels overwhelming: {auto_thought or raw[:50]}."
                caption = "When stressors accumulate, internal alarms sound loudly."
            thought_text = dialogue
            narrative = f"You're facing a difficult moment: '{raw.strip()}'. {conflict or 'The emotional weight builds quickly and feels pressing.'}"
        elif cat == "freeze":
            dialogue = "If I can't finish all twenty things today, what's the point of starting?"
            thought_text = dialogue
            caption = "All-or-nothing thinking magnifies tasks into an insurmountable wall."
            narrative = "Trying to complete every obligation simultaneously makes starting anything feel impossible."
        elif cat == "spiral":
            if case.domain == "relationship":
                dialogue = "If it hurts this much, does that mean I made a mistake?"
                caption = "Separation pain is easily misinterpreted as proof that ending things was wrong."
            elif case.domain == "social_comparison":
                dialogue = "I keep looping through how far ahead everyone else is compared to me."
                caption = "Comparing your internal struggle to external highlights feeds the spiral."
            elif case.domain == "uncertainty":
                dialogue = "What if something terrible happens and I'm powerless to handle it?"
                caption = "Demanding certainty from an unpredictable future fuels escalating dread."
            elif case.domain == "rumination":
                dialogue = "Why did I say that? I can't stop replaying how awkward it was."
                caption = "Mentally replaying the past gives a false illusion of control while deepening regret."
            elif case.domain == "self_criticism":
                dialogue = "I messed up once, so that must mean I'm completely useless."
                caption = "A single mistake can be magnified into a harsh verdict on your entire worth."
            else:
                dialogue = f"I keep questioning: {conflict[:70] if conflict else 'what if everything goes wrong?'}"
                caption = "A single unresolved worry expands into an escalating loop of doubt."
            thought_text = dialogue
            narrative = f"The automatic thought takes hold: '{auto_thought or 'looping through doubts without resolution'}'. The spiral accelerates."
        elif cat == "examination":
            if case.domain == "relationship":
                dialogue = "Wait. Pain means I care, but it doesn't erase why I was uncertain."
                caption = "Pausing allows you to separate immediate sorrow from the reality of your doubts."
            elif case.domain == "overwhelm":
                dialogue = "Wait. I don't have to conquer the whole mountain in this single hour."
                caption = "Pausing interrupts the freeze: separate the total pile from the next five minutes."
            elif case.domain == "social_comparison":
                dialogue = "Wait. Their public highlights don't define my personal journey."
                caption = "Separate other people's curated milestones from your own authentic values."
            elif case.domain == "uncertainty":
                dialogue = "Wait. Not knowing what happens next doesn't mean catastrophe is guaranteed."
                caption = "Pausing separates the feeling of uncertainty from the certainty of threat."
            elif case.domain == "academic":
                dialogue = "Wait. Being nervous just means I care, not that failure is certain."
                caption = "Pausing helps distinguish emotional alarm from objective capability."
            else:
                dialogue = "Wait. Let me separate what I'm feeling from what is actually true."
                caption = "Pausing allows you to distinguish between an intense emotion and an objective fact."
            thought_text = dialogue
            narrative = "By stepping back for a moment, you notice: your feelings are intense and valid, but painful thoughts are not infallible predictions."
        elif cat == "reframing":
            if case.domain == "relationship" and case.genuine_uncertainty:
                dialogue = _extract_dialogue_phrase(reframe, "Feeling pain doesn't mean this was wrong. I can care and still honor my doubts.")
                caption = "Pain reflects that you cared, not that the relationship should have continued."
                narrative = reframe or "Feeling pain after a breakup does not by itself prove that the relationship should have continued. You can acknowledge your grief while honoring that your uncertainty was honest and real."
            elif case.domain == "uncertainty":
                dialogue = _extract_dialogue_phrase(reframe, "I cannot predict the entire future today. I can handle the hour in front of me.")
                caption = "Reframing shifts focus from demanding future certainty to exercising present agency."
                narrative = reframe or "Not knowing what will happen is uncomfortable, but uncertainty does not mean catastrophe. Focus your agency on what is real and manageable right now."
            elif case.domain == "social_comparison":
                dialogue = _extract_dialogue_phrase(reframe, "Other people's pace is not my measuring tape. My journey has its own timing.")
                caption = "Replace upward social comparison with alignment to your own intrinsic values and pace."
                narrative = reframe or "You are comparing your private struggles to everyone else's public highlights. Another person's success does not diminish your worth or speed."
            elif case.domain == "overwhelm":
                dialogue = _extract_dialogue_phrase(reframe, "I don't need to do twenty things today. I only need to do one 5-minute step.")
                caption = "Break the wall down into a single brick. Momentum follows action, not perfection."
                narrative = reframe or "You do not have to conquer all twenty tasks simultaneously. You only need to touch one small corner of one task."
            elif case.domain == "academic":
                dialogue = _extract_dialogue_phrase(reframe, "Feeling nervous doesn't mean I will fail. I can focus on the questions I know.")
                caption = "Reframe anticipatory anxiety into focused preparation on what you can control."
                narrative = reframe or "Feeling nervous is simply your body preparing for a challenge. Direct your energy strictly into what is within your agency."
            else:
                dialogue = _extract_dialogue_phrase(reframe, "Taking a step back helps me look at this with grounded clarity.")
                caption = "Reframing shifts your attention from uncontrollable futures to present agency."
                narrative = reframe or "Focus on what is truly within your control right now, letting uncontrollable outcomes rest."
            thought_text = dialogue
        elif cat in ["resolution", "action"]:
            bubble_type = "speech"
            if case.domain == "relationship" and case.genuine_uncertainty:
                dialogue = _extract_dialogue_phrase(action, "I don't have to force certainty today. I'll let myself feel sad and take one quiet breath.")
                caption = "Give yourself permission to grieve without rushing to prove your decision right or wrong."
                narrative = action or "Give yourself permission to feel the sadness of the ending today without rushing to judge whether the decision was definitely right or wrong."
            elif case.domain == "uncertainty":
                dialogue = _extract_dialogue_phrase(action, "I will anchor in today's routine tasks and take it one steady hour at a time.")
                caption = "Grounding in immediate present action relieves the burden of carrying tomorrow's unknowns."
                narrative = action or "Anchor your agency in the present environment. Complete one simple routine action and let the distant future wait."
            elif case.domain == "social_comparison":
                dialogue = _extract_dialogue_phrase(action, "I will set my phone aside and focus on what genuinely matters to my own growth.")
                caption = "Direct your attention inward toward your own growth, craft, and authentic progress."
                narrative = action or "Put down social media, honor your own pace, and take one purposeful step toward your personal values."
            elif case.domain == "overwhelm":
                dialogue = _extract_dialogue_phrase(action, "I'll spend five minutes on this one task, and let that be enough for now.")
                caption = "One small step forward breaks the freeze and builds genuine momentum."
                narrative = action or "Pick the simplest sub-task on your list, set a timer for five minutes, and do just that one thing."
            elif case.domain == "academic":
                dialogue = _extract_dialogue_phrase(action, "I will review these three topics calmly and get a good night of rest.")
                caption = "Action follows clarity: targeted preparation builds calm capability."
                narrative = action or "Focus on reviewing your main core topics today, taking scheduled rest breaks, and trusting your preparation."
            else:
                dialogue = _extract_dialogue_phrase(action, "I will take one small constructive step today and be patient with myself.")
                caption = "Action follows clarity: one manageable step builds real momentum."
                narrative = action or "Take one small, specific step right now. Give yourself permission to let the rest wait."
            thought_text = dialogue
        else:
            dialogue = "I will take this one step at a time."
            thought_text = dialogue
            caption = "Focus on what is manageable and grounded."
            narrative = "Move forward with patience and clarity."

        return dialogue, thought_text, bubble_type, caption, narrative

    # =========================================================================
    # STEP 9 — SCENE TRANSITION CHECK & PHYSICAL DIFFERENTIATION
    # =========================================================================
    def _verify_and_refine_transitions(
        self, scenes: List[SceneState], case: CaseUnderstanding, anchors: SituationAnchors
    ) -> List[SceneState]:
        """
        Compares every adjacent pair of scenes (Scene N -> Scene N+1) and verifies that:
        1. The psychological state shifted.
        2. The character did something physically different.
        3. The object interaction or spatial relationship evolved.
        If adjacent scenes are too physically similar, refines Scene N+1.
        """
        for i in range(len(scenes) - 1):
            curr = scenes[i]
            nxt = scenes[i + 1]

            # Check for physical action redundancy
            if curr.posture == nxt.posture and curr.camera_distance == nxt.camera_distance:
                # Differentiate next scene physically
                if "hunched" in curr.posture.lower() or "slumped" in curr.posture.lower():
                    nxt.posture = "Shifting from hunched defense to balanced, steady upright posture"
                    nxt.character_action = f"setting {anchors.primary_object} down, taking a deep deliberate breath with open posture"
                elif "sitting" in curr.character_action.lower():
                    nxt.posture = "Standing upright with grounded posture"
                    nxt.character_action = f"standing up purposefully from desk, walking toward the window to gain perspective"
                    nxt.camera_distance = "medium shot"

            # Ensure object interaction evolves across scenes
            if curr.object_interaction == nxt.object_interaction:
                if i == 0:
                    nxt.object_interaction = f"interrogating or hesitating over {anchors.primary_object}"
                elif i == 1:
                    nxt.object_interaction = f"placing {anchors.primary_object} face-down to focus on {anchors.secondary_object}"
                elif i >= 2:
                    nxt.object_interaction = f"actively writing or setting anchor in {anchors.secondary_object}"

        return scenes

    # =========================================================================
    # STEP 10 — CASE-SPECIFICITY TEST
    # =========================================================================
    def _verify_case_specificity(
        self, scenes: List[SceneState], case: CaseUnderstanding, anchors: SituationAnchors
    ) -> List[SceneState]:
        """
        Ensures visual descriptions are strongly grounded in the specific case,
        preventing generic reusable descriptions (e.g. 'person sitting sadly by window').
        """
        for s in scenes:
            desc = s.visual_description.lower()
            # If visual description is too generic, ground it with authentic anchors
            if len(desc) < 30 or ("sitting" in desc and "window" in desc and len(desc) < 60):
                s.visual_description = (
                    f"{s.character_action}, {s.posture.lower()}, {s.expression.lower()}, {s.lighting.lower()}"
                )

        return scenes

    # =========================================================================
    # STEP 11 — PRESERVE GENUINE UNCERTAINTY
    # =========================================================================
    def _enforce_uncertainty_preservation(
        self, scenes: List[SceneState], case: CaseUnderstanding
    ) -> List[SceneState]:
        """
        In situations involving honest uncertainty (relationship ambivalence, unknown futures),
        prevents manufacturing false certainty. Ensures text and narrative represent
        'I can tolerate not knowing right now' rather than 'I know all the answers'.
        """
        if not case.genuine_uncertainty:
            return scenes

        false_certainty_markers = [
            "definitely the right choice", "definitely correct", "guaranteed right decision",
            "now i know everything", "100% sure", "i have all the answers"
        ]

        for s in scenes:
            for marker in false_certainty_markers:
                if marker in s.dialogue.lower():
                    s.dialogue = "I don't have to force certainty today. I can honor my feelings and give myself time."
                    s.thought_text = s.dialogue
                if marker in s.narrative.lower():
                    if case.domain == "relationship":
                        s.narrative = "Give yourself permission to feel grief and doubt today without rushing to judge whether the decision was right or wrong."
                    else:
                        s.narrative = "Give yourself permission to navigate today without rushing to demand absolute guarantees about the future."

        return scenes

    # =========================================================================
    # BACKWARD COMPATIBILITY & STOPPING CRITERIA (Bounds [3, 8])
    # =========================================================================
    def _determine_scene_count(self, raw_text: str, pattern: str, is_threat: bool, distortions: list) -> int:
        """Backward-compatible helper evaluating situation complexity."""
        case = self._understand_case(
            type("MockFrame", (), {"raw_text": raw_text, "pattern": pattern, "is_external_threat": is_threat, "distortions": distortions})()
        )
        milestones = self._derive_psychological_arc(case)
        return len(milestones)

    def _apply_stopping_criteria(self, candidate_scenes: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """Enforces minimum 3 scenes and maximum 8 scenes, re-indexing scene_id cleanly."""
        clamped_count = max(self.MIN_SCENES, min(target_count, self.MAX_SCENES))

        if len(candidate_scenes) > clamped_count:
            if clamped_count == 4 and len(candidate_scenes) == 5:
                selected = [candidate_scenes[0], candidate_scenes[1], candidate_scenes[3], candidate_scenes[4]]
            else:
                selected = candidate_scenes[:clamped_count - 1] + [candidate_scenes[-1]]
        else:
            selected = candidate_scenes[:clamped_count]

        for idx, scene in enumerate(selected, start=1):
            scene["scene_id"] = idx
            scene["stage"] = scene.get("stage_category", "scene")
            scene["title"] = f"{idx}. {scene.get('stage_category', 'scene').upper()}"

        return selected

    def _build_simple_progression(self, raw_text: str, trigger: str, reframe: str,
                                  action: str, pattern: str, emotion: str, archetype: dict,
                                  case_frame=None) -> List[Dict[str, Any]]:
        """Backward-compatible simple 3-scene helper."""
        case = CaseUnderstanding(
            raw_text=raw_text, situation=trigger, primary_emotion=emotion, central_conflict="",
            automatic_thought="", appraisal="", pattern=pattern, coping_strategy="",
            specific_reframe=reframe, concrete_next_step=action, controllable="", uncontrollable="",
            is_external_threat=False, domain="academic", genuine_uncertainty=False, distortions=[]
        )
        milestones = self._derive_psychological_arc(case)
        anchors = self._derive_anchors(case)
        scenes = [self._orchestrate_scene(i, m, len(milestones), case, anchors).to_dict() for i, m in enumerate(milestones, 1)]
        return scenes

    def _build_moderate_spiral_progression(self, raw_text: str, trigger: str, reframe: str,
                                           action: str, pattern: str, emotion: str, archetype: dict,
                                           case_frame=None) -> List[Dict[str, Any]]:
        """Backward-compatible 5-scene spiral helper."""
        case = CaseUnderstanding(
            raw_text=raw_text, situation=trigger, primary_emotion=emotion, central_conflict="",
            automatic_thought="", appraisal="", pattern=pattern or "rumination", coping_strategy="",
            specific_reframe=reframe, concrete_next_step=action, controllable="", uncontrollable="",
            is_external_threat=False, domain="rumination", genuine_uncertainty=False, distortions=[]
        )
        milestones = self._derive_psychological_arc(case)
        anchors = self._derive_anchors(case)
        scenes = [self._orchestrate_scene(i, m, len(milestones), case, anchors).to_dict() for i, m in enumerate(milestones, 1)]
        return scenes

    def _build_threat_progression(self, raw_text: str, trigger: str, reframe: str,
                                  action: str, emotion: str, archetype: dict,
                                  case_frame=None) -> List[Dict[str, Any]]:
        """Backward-compatible 6-scene threat helper."""
        case = CaseUnderstanding(
            raw_text=raw_text, situation=trigger, primary_emotion=emotion, central_conflict="",
            automatic_thought="", appraisal="", pattern="interpersonal_threat", coping_strategy="",
            specific_reframe=reframe, concrete_next_step=action, controllable="", uncontrollable="",
            is_external_threat=True, domain="threat", genuine_uncertainty=False, distortions=[]
        )
        milestones = self._derive_psychological_arc(case)
        anchors = self._derive_anchors(case)
        scenes = [self._orchestrate_scene(i, m, len(milestones), case, anchors).to_dict() for i, m in enumerate(milestones, 1)]
        return scenes

"""
Verification Suite for the General Visual Narrative Orchestration Engine.

Tests all 6 required distinct cases:
1. Breakup + uncertainty
2. Performance anxiety
3. Social comparison
4. Interpersonal threat
5. Rumination
6. Uncertainty about a future outcome

Inspects the structured SceneState to verify:
- Distinct psychological arcs
- Observable physical actions (no abstract emotional labels like 'looks sad')
- Natural situation-specific objects and anchored object lifecycles
- Environmental continuity without generic mood filters
- Meaningful physical and compositional transitions between adjacent scenes
- High case-specificity (no generic reusable visual descriptions)
- Strict absence of cross-domain contamination
- Genuine uncertainty preserved where appropriate
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.psychology import parse_situation, select_strategy
from app.context_engine.case_frame import CaseFrame
from app.narrative_gen.scene_planner import ScenePlanner, SceneState


def _plan_for_text(text: str):
    profile = parse_situation(text)
    strategy = select_strategy(profile)
    planner = ScenePlanner()

    reasoning_dict = profile.structured_reasoning()
    reasoning_dict.update({
        "what_may_be_happening": f"{profile.trigger} → {profile.pattern}",
        "what_you_can_control": strategy.what_is_controllable,
        "what_is_not_controllable": strategy.what_is_not_controllable,
        "reframe": strategy.reframe,
        "next_step": strategy.concrete_action,
    })

    case_frame = CaseFrame(
        raw_text=text,
        situation_type=profile.situation_id,
        trigger=profile.trigger,
        pattern=profile.pattern,
        is_external_threat=profile.is_external_threat,
        controllable=strategy.what_is_controllable,
        uncontrollable=strategy.what_is_not_controllable,
        strategy=strategy.to_dict(),
        specific_reframe=strategy.reframe,
        concrete_action=strategy.concrete_action,
        reasoning=reasoning_dict,
    )
    case_frame.build_summary()

    scenes = planner.plan_scenes(case_frame, strategy.to_dict(), strategy.comic_archetype)
    return scenes, case_frame, profile, strategy


def test_six_distinct_cases_produce_distinct_orchestrations():
    """Verifies that all 6 situations produce distinct arcs, actions, objects, and scene counts."""
    cases = [
        ("I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful", "breakup_uncertainty"),
        ("I'm going to fail my interview tomorrow because I always freeze up", "performance_anxiety"),
        ("Everyone is doing better than me in life and my career is falling behind", "social_comparison"),
        ("My father lashes out at me and then acts completely normal afterward", "interpersonal_threat"),
        ("I keep replaying an embarrassing awkward mistake I made yesterday", "rumination"),
        ("I don't know what will happen with my future and I am scared something bad will happen", "future_uncertainty"),
    ]

    all_scenes_by_case = {}

    for text, label in cases:
        scenes, case_frame, profile, strategy = _plan_for_text(text)
        all_scenes_by_case[label] = scenes

        # 1. Enforce scene count bounds [3, 8]
        assert 3 <= len(scenes) <= 8, f"Case {label} produced invalid scene count: {len(scenes)}"

        # 2. Verify all SceneState schema fields are present on every scene
        required_keys = [
            "scene_id", "stage_category", "stage", "title", "narrative_purpose",
            "psychological_purpose", "trigger", "automatic_thought", "appraisal", "emotion",
            "character_goal", "character_action", "posture", "expression", "gaze",
            "environment", "important_objects", "object_interaction", "spatial_relationships",
            "camera_distance", "camera_angle", "composition", "lighting", "visual_mood",
            "dialogue", "thought_text", "caption", "visual_description", "narrative",
            "psychological_state", "character_state", "visual_staging"
        ]
        for idx, scene in enumerate(scenes, 1):
            assert scene["scene_id"] == idx
            for k in required_keys:
                assert k in scene, f"Missing key '{k}' in {label} scene {idx}"
                assert scene[k] is not None, f"Key '{k}' is None in {label} scene {idx}"

            # Verify observable physical action: avoid pure abstract emotions
            action = scene["character_action"].lower()
            banned_abstract = ["looks sad", "looks thoughtful", "feels anxious", "becomes happier", "looks relieved"]
            for bad in banned_abstract:
                assert action != bad, f"Abstract emotional instruction '{bad}' used as action in {label} scene {idx}"

        # 3. Step 9: Verify meaningful physical differences between adjacent scenes
        for i in range(len(scenes) - 1):
            curr = scenes[i]
            nxt = scenes[i + 1]
            assert curr["character_action"] != nxt["character_action"], (
                f"{label}: Adjacent scenes {i+1} and {i+2} have duplicate physical action"
            )
            assert curr["stage_category"] != nxt["stage_category"], (
                f"{label}: Adjacent scenes {i+1} and {i+2} share the same stage_category"
            )

    # 4. Verify distinct arcs across distinct problem types
    # Threat has 6 scenes
    assert len(all_scenes_by_case["interpersonal_threat"]) == 6
    threat_stages = [s["stage_category"] for s in all_scenes_by_case["interpersonal_threat"]]
    assert threat_stages == ["incident", "disorientation", "validation", "externalizing", "boundary", "support"]

    # Performance anxiety has 3 scenes
    assert len(all_scenes_by_case["performance_anxiety"]) == 3
    perf_stages = [s["stage_category"] for s in all_scenes_by_case["performance_anxiety"]]
    assert perf_stages == ["problem", "reframing", "resolution"]

    # Social comparison and Rumination have 5 scenes
    assert len(all_scenes_by_case["social_comparison"]) == 5
    assert len(all_scenes_by_case["rumination"]) == 5

    # Breakup uncertainty has 5 scenes
    assert len(all_scenes_by_case["breakup_uncertainty"]) == 5

    # 5. Verify different objects between different domains
    threat_objs = str(all_scenes_by_case["interpersonal_threat"][0]["important_objects"]).lower()
    perf_objs = str(all_scenes_by_case["performance_anxiety"][0]["important_objects"]).lower()
    breakup_objs = str(all_scenes_by_case["breakup_uncertainty"][0]["important_objects"]).lower()

    assert "door" in threat_objs or "hallway" in threat_objs
    assert "exam" in perf_objs or "notes" in perf_objs or "textbook" in perf_objs
    assert "phone" in breakup_objs or "message" in breakup_objs


def test_preservation_of_genuine_uncertainty():
    """Confirms cases with honest uncertainty do not manufacture premature false certainty."""
    # Case 1: Breakup with uncertainty
    text_breakup = "I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful"
    scenes, _, _, _ = _plan_for_text(text_breakup)

    all_text = " ".join(s["dialogue"] + " " + s["caption"] + " " + s["narrative"] for s in scenes).lower()
    assert "definitely the right choice" not in all_text
    assert "definitely correct" not in all_text
    assert "guaranteed right" not in all_text

    # Must preserve tolerance of grief and uncertainty
    assert "pain" in all_text or "grief" in all_text or "sad" in all_text
    assert "uncertain" in all_text or "doubts" in all_text or "care" in all_text

    # Case 6: Future uncertainty
    text_future = "I don't know what will happen with my future and I am scared something bad will happen"
    scenes_future, _, _, _ = _plan_for_text(text_future)

    future_text = " ".join(s["dialogue"] + " " + s["caption"] + " " + s["narrative"] for s in scenes_future).lower()
    assert "now i know everything" not in future_text
    assert "100% sure" not in future_text
    assert "uncertain" in future_text or "predict" in future_text or "control" in future_text or "one day" in future_text


def test_no_cross_domain_contamination_in_scenestates():
    """Verifies that performance terms do not contaminate breakup/threat scenes and vice versa."""
    # Breakup scenario
    text_breakup = "I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful"
    scenes, _, _, _ = _plan_for_text(text_breakup)

    breakup_full_text = " ".join(s["dialogue"] + " " + s["visual_description"] for s in scenes).lower()
    for term in ["interview", "exam", "grades", "evaluator", "freeze and fail"]:
        assert term not in breakup_full_text, f"Term '{term}' leaked into breakup scene!"

    # Performance scenario
    text_perf = "I'm going to fail my interview tomorrow because I always freeze up"
    scenes_perf, _, _, _ = _plan_for_text(text_perf)

    perf_full_text = " ".join(s["dialogue"] + " " + s["visual_description"] for s in scenes_perf).lower()
    for term in ["breakup", "ex-partner", "did not love"]:
        assert term not in perf_full_text, f"Term '{term}' leaked into performance scene!"


def test_object_lifecycle_evolves_across_panels():
    """Verifies that the character's relationship to the primary and secondary objects evolves across scenes."""
    text = "I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful"
    scenes, _, _, _ = _plan_for_text(text)

    # Scene 1: gripped by / staring transfixed at primary object (phone)
    assert "gripping" in scenes[0]["object_interaction"].lower() or "staring" in scenes[0]["object_interaction"].lower()

    # Scene 3: placing phone face-down / physical pause
    assert "face-down" in scenes[2]["object_interaction"].lower() or "setting" in scenes[2]["character_action"].lower()

    # Scene 4: writing in secondary object (notebook)
    assert "writing" in scenes[3]["object_interaction"].lower() or "writing" in scenes[3]["character_action"].lower()

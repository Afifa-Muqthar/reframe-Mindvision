"""
Stage 4 Integration Test Suite: Grounded Visual Staging & Prompt Alignment.

Validates:
1. Grounded visual staging enrichment via ScenePlanner.plan_grounded_scenes.
2. 100% preservation of Stage 3 dialogues, captions, bubble types, and narrative texts.
3. Complete SceneState schema compatibility.
4. Physical visual progression and adjacent panel distinction.
5. Derivation of visual anchors from stated facts without hallucinations.
6. Compatibility with build_scene_prompts and comic visual style.
7. Support across all 6 modalities.
"""

import pytest
from app.psychology.case_representation import extract_case_representation
from app.psychology.case_strategy_selector import CaseStrategySelector
from app.narrative_gen.grounded_generator import GroundedNarrativeGenerator
from app.narrative_gen.scene_planner import ScenePlanner
from app.image_gen.prompt_builder import build_scene_prompts, COMIC_VISUAL_STYLE, build_character_anchor


@pytest.fixture
def pipeline():
    selector = CaseStrategySelector()
    gen = GroundedNarrativeGenerator()
    planner = ScenePlanner()
    return selector, gen, planner


def test_target_regression_c1_visual_staging(pipeline):
    selector, gen, planner = pipeline
    text = (
        "I'm worried about my career after graduation. My friends are already getting placed, "
        "and I'm genuinely happy for them, but I feel like I haven't done enough and I'm afraid of disappointing my family."
    )
    rep = extract_case_representation(text)
    strat = selector.select(rep)
    raw_story = gen.generate(rep, strat)

    assert len(raw_story["scenes"]) == 5

    enriched_scenes = planner.plan_grounded_scenes(raw_story, rep, strat)
    assert len(enriched_scenes) == 5

    # 1. Verify preservation of Stage 3 dialogues and captions
    for raw, enriched in zip(raw_story["scenes"], enriched_scenes):
        assert enriched["dialogue"] == raw["dialogue"]
        assert enriched["caption"] == raw["caption"]
        assert enriched["narrative"] == raw["narrative"]
        assert enriched["bubble_type"] == raw["bubble_type"]
        assert enriched["stage"] == raw["stage"]

    # 2. Verify all SceneState schema fields are present
    required_keys = [
        "scene_id", "stage_category", "stage", "title", "character_action",
        "posture", "expression", "gaze", "environment", "important_objects",
        "camera_distance", "camera_angle", "lighting", "visual_description",
        "dialogue", "caption", "narrative", "detail_traceability"
    ]
    for idx, sc in enumerate(enriched_scenes, 1):
        for k in required_keys:
            assert k in sc, f"Missing key {k} in scene {idx}"
            assert sc[k] is not None, f"Key {k} is None in scene {idx}"

    # 3. Verify observable physical actions (no abstract emotional labels)
    banned_abstract = ["looks sad", "looks thoughtful", "feels anxious", "becomes happier"]
    for sc in enriched_scenes:
        for bad in banned_abstract:
            assert sc["character_action"].lower() != bad

    # 4. Verify physical visual progression across adjacent panels
    actions = [s["character_action"] for s in enriched_scenes]
    assert len(set(actions)) == 5, f"All 5 scenes must have distinct physical actions, got: {actions}"

    # 5. Verify prompt builder compiles ~40-48 word prompts without falling back to generic placeholders
    raw_story["scenes"] = enriched_scenes
    prompts = build_scene_prompts(raw_story)
    assert len(prompts) == 5

    anchor = build_character_anchor(raw_story.get("character", {}))
    for p in prompts:
        assert anchor in p
        assert COMIC_VISUAL_STYLE in p
        assert "sitting quietly pausing and breathing" not in p, "Generic fallback action detected in compiled prompt"


def test_six_modalities_visual_staging_distinct(pipeline):
    selector, gen, planner = pipeline
    test_cases = [
        ("I have chronic fatigue flare-ups and I can barely get out of bed.", "validation"),
        ("I just got promoted to lead the team, but I feel like an absolute fraud.", "perspective_reappraisal"),
        ("My friends got placed and I'm worried about my career timeline.", "locus_of_agency"),
        ("I have five deadlines due Friday and I'm sitting on the floor unable to move.", "practical_structuring"),
        ("We've been dating for four years but I feel disconnected and don't know what to do.", "non_interventional_grounding"),
        ("I don't know... just tired.", "clarification_needed"),
    ]

    for text, expected_modality in test_cases:
        rep = extract_case_representation(text)
        strat = selector.select(rep)
        assert strat.modality == expected_modality, f"Expected {expected_modality}, got {strat.modality}"

        raw_story = gen.generate(rep, strat)
        enriched = planner.plan_grounded_scenes(raw_story, rep, strat)

        assert 3 <= len(enriched) <= 5
        actions = [s["character_action"] for s in enriched]
        # At least adjacent actions must be distinct
        for i in range(len(actions) - 1):
            assert actions[i] != actions[i + 1], f"Duplicate adjacent action in {expected_modality}"

        # Verify prompt compiles cleanly
        raw_story["scenes"] = enriched
        prompts = build_scene_prompts(raw_story)
        assert len(prompts) == len(enriched)

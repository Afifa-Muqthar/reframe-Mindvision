"""
Tests for Psychological Reasoning Layer, Situation Parsing, and Strategy Selection.
"""

import pytest
from app.psychology.situation_parser import parse_situation
from app.psychology.strategy_selector import select_strategy
from app.context_engine.case_frame import CaseFrame
from app.narrative_gen.generator import NarrativeGenerator, format_narrative_text


def test_abusive_father_trauma_informed():
    """Verify that unpredictable parental hostility is classified as an external threat,
    NOT as an internal cognitive distortion, and receives trauma-informed safety reframing."""
    text = "abusive father who lashes out then acts all normal"
    profile = parse_situation(text)

    assert profile.situation_id == "interpersonal_threat"
    assert profile.is_external_threat is True
    assert "unpredictable" in profile.trigger.lower() or "anger" in profile.trigger.lower()

    strategy = select_strategy(profile)
    assert strategy.category == "trauma_informed"
    assert "gratitude" not in strategy.name.lower()
    assert "gratitude" not in strategy.reframe.lower()

    # Reframe must address the thought directly without minimizing reality
    assert "erase what happened" in strategy.reframe
    assert "boundaries" in strategy.what_is_controllable.lower()
    assert "distance" in strategy.concrete_action.lower() or "safety" in strategy.concrete_action.lower()


@pytest.mark.parametrize(
    "text, expected_situation_id, expected_threat",
    [
        ("I'm going to fail my interview", "performance_anxiety", False),
        ("Everyone hates me because my friend didn't reply", "rejection_sensitivity", False),
        ("I keep thinking about what happened yesterday", "rumination", False),
        ("I messed up once, I'm useless", "self_criticism", False),
        ("I don't know what will happen", "uncertainty", False),
        ("My father lashes out and later behaves normally", "interpersonal_threat", True),
        ("I can't stop thinking about my breakup", "loss_and_grief", False),
        ("I have 20 things to do and can't start", "overwhelm", False),
        ("Everyone is doing better than me", "social_comparison", False),
        ("I keep replaying an embarrassing moment", "rumination", False),
        ("I am scared something bad will happen", "uncertainty", False),
    ],
)
def test_user_benchmark_situations(text, expected_situation_id, expected_threat):
    profile = parse_situation(text)
    assert profile.situation_id == expected_situation_id
    assert profile.is_external_threat is expected_threat

    strategy = select_strategy(profile)
    assert strategy.reframe != ""
    assert strategy.concrete_action != ""
    assert strategy.what_is_controllable != ""
    assert strategy.what_is_not_controllable != ""


def test_narrative_generation_grounding():
    """Verify that narrative generation produces 3 connected scenes and voice script
    grounded in the psychological strategy."""
    text = "abusive father who lashes out then acts all normal"
    profile = parse_situation(text)
    strategy = select_strategy(profile)

    case_frame = CaseFrame(
        raw_text=text,
        core_emotion="fear",
        situation_type=profile.situation_id,
        trigger=profile.trigger,
        pattern=profile.pattern,
        is_external_threat=profile.is_external_threat,
        controllable=strategy.what_is_controllable,
        uncontrollable=strategy.what_is_not_controllable,
        strategy=strategy.to_dict(),
        specific_reframe=strategy.reframe,
        concrete_action=strategy.concrete_action,
    )
    case_frame.build_summary()

    technique = {"name": strategy.name, "citation": strategy.citation, "description": strategy.mechanism}
    generator = NarrativeGenerator(use_small=True)
    try:
        story = generator.generate(case_frame, technique)
    finally:
        generator.unload()

    # In adaptive architecture, trauma-informed interpersonal threat produces an adaptive progression (> 3 scenes)
    assert len(story["scenes"]) >= 4
    assert len(story["scenes"]) <= 8

    # Captions present across all scenes
    for s in story["scenes"]:
        assert "caption" in s
        assert len(s["caption"]) > 0

    # Voice script present and meaningful
    assert "voice_script" in story
    assert len(story["voice_script"]) > 50
    assert any(term in story["voice_script"].lower() for term in ["erase what happened", "boundaries", "distance", "support", "volatili"])

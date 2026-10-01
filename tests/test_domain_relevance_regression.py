"""
Regression tests for Cross-Domain Relevance, Content Specificity, and Narrative Integrity.

Validates:
1. Future Uncertainty: Contains ZERO breakup/grief terms (no 'breakup', 'grief', 'loved him',
   'permission to grieve', 'relationship'). Addresses locus of control and present agency.
2. Social Comparison: Accurately classified as social_comparison (not generic stress/overwhelm).
   Contains comparison and personal timeline content, ZERO task-overwhelm dialogue ('twenty tasks', 'timer for five minutes').
3. Task Overwhelm: Accurately classified as overwhelm. Focuses on task decomposition and micro-action.
4. Breakup Ambivalence: Accurately classified as relationship. Preserves genuine sadness and uncertainty
   without forcing false certainty or leaking performance/task-overwhelm advice.
5. Dialogue Integrity: No words are chopped mid-character (no 'automatical', no incomplete slices).
"""

import os
import sys
import pytest
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.psychology import parse_situation, select_strategy
from app.context_engine.case_frame import CaseFrame
from app.narrative_gen.scene_planner import ScenePlanner
from app.psychology.semantic_validator import SemanticValidator
from app.image_gen.prompt_builder import build_scene_prompts, build_character_anchor


BANNED_RELATIONSHIP_TERMS = [
    "breakup", "break up", "broke up", "relationship", "partner", "ex-partner",
    "loved him", "loved her", "permission to grieve", "should have continued",
    "separation pain", "grief is love", "romantic feelings", "did not love",
]

BANNED_OVERWHELM_TERMS = [
    "twenty things", "20 things", "twenty tasks", "mountain of work",
    "timer for five minutes", "five-minute task",
]

BANNED_PERFORMANCE_TERMS = [
    "exam", "interview", "test", "presentation", "freeze and fail", "will fail",
    "talking points", "evaluator",
]


def _build_story_scenes(input_text: str):
    profile = parse_situation(input_text)
    strategy = select_strategy(profile)
    case_frame = CaseFrame(
        raw_text=input_text,
        core_emotion=profile.primary_emotion,
        pattern=profile.pattern,
        situation_type=profile.situation_id,
        is_external_threat=profile.is_external_threat,
        controllable=strategy.what_is_controllable,
        uncontrollable=strategy.what_is_not_controllable,
        specific_reframe=strategy.reframe,
        concrete_action=strategy.concrete_action,
        strategy=strategy.to_dict(),
        reasoning=profile.structured_reasoning(),
    )
    planner = ScenePlanner()
    scenes = planner.plan_scenes(case_frame, strategy.to_dict(), strategy.comic_archetype)

    story = {
        "title": f"Reframing with {strategy.name}",
        "emotion": case_frame.core_emotion,
        "character": strategy.comic_archetype.get("character", {}),
        "scenes": scenes,
    }
    sanitized = SemanticValidator.sanitize_story(case_frame, story)
    return profile, strategy, sanitized["scenes"]


def test_future_uncertainty_has_no_breakup_or_grief_contamination():
    """Future uncertainty must NEVER leak breakup, grief, or separation text."""
    input_text = "I don't know what will happen with my future and career, and the constant uncertainty is making me anxious and terrified."
    profile, strategy, scenes = _build_story_scenes(input_text)

    assert profile.situation_id == "uncertainty"
    assert "uncertainty" in strategy.id.lower() or "control" in strategy.id.lower()
    assert len(scenes) >= 3

    combined_text = " ".join(
        f"{s.get('dialogue', '')} {s.get('caption', '')} {s.get('narrative', '')}"
        for s in scenes
    ).lower()

    for term in BANNED_RELATIONSHIP_TERMS:
        assert term not in combined_text, f"Future uncertainty leaked relationship term: '{term}' in '{combined_text}'"

    for term in BANNED_OVERWHELM_TERMS:
        assert term not in combined_text, f"Future uncertainty leaked overwhelm term: '{term}'"

    # Must contain uncertainty / present grounding concepts
    assert any(w in combined_text for w in ["uncertain", "future", "today", "hour", "present", "agency", "routine"])


def test_social_comparison_classified_correctly_and_free_of_task_overwhelm():
    """Social comparison must not degrade to general stress or task overwhelm."""
    input_text = "Everyone my age seems to be succeeding and doing so much better than me, while I feel like a total failure and stuck."
    profile, strategy, scenes = _build_story_scenes(input_text)

    assert profile.situation_id == "social_comparison"
    assert "comparison" in strategy.id.lower() or "values" in strategy.id.lower()
    assert len(scenes) >= 3

    combined_text = " ".join(
        f"{s.get('dialogue', '')} {s.get('caption', '')} {s.get('narrative', '')}"
        for s in scenes
    ).lower()

    for term in BANNED_OVERWHELM_TERMS:
        assert term not in combined_text, f"Social comparison leaked task-overwhelm term: '{term}'"

    for term in BANNED_RELATIONSHIP_TERMS:
        assert term not in combined_text, f"Social comparison leaked relationship term: '{term}'"

    # Must address comparison, pacing, or personal values
    assert any(w in combined_text for w in ["pace", "timing", "journey", "highlights", "values", "better", "milestone"])


def test_task_overwhelm_properly_decomposed():
    """Task overwhelm must address cognitive overload and micro-actions without contamination."""
    input_text = "I have 20 things to do today and I feel completely overwhelmed and paralyzed, I don't know where to start."
    profile, strategy, scenes = _build_story_scenes(input_text)

    assert profile.situation_id == "overwhelm"
    assert "overwhelm" in strategy.id.lower() or "micro_action" in strategy.id.lower()
    assert len(scenes) >= 3

    combined_text = " ".join(
        f"{s.get('dialogue', '')} {s.get('caption', '')} {s.get('narrative', '')}"
        for s in scenes
    ).lower()

    for term in BANNED_RELATIONSHIP_TERMS:
        assert term not in combined_text, f"Task overwhelm leaked relationship term: '{term}'"

    for term in BANNED_PERFORMANCE_TERMS:
        assert term not in combined_text, f"Task overwhelm leaked performance term: '{term}'"

    assert any(w in combined_text for w in ["five", "5-minute", "one", "brick", "step", "start", "freeze", "mountain"])


def test_breakup_ambivalence_preserves_uncertainty_without_performance_cliches():
    """Breakup ambivalence must address separation pain while honoring honest uncertainty."""
    input_text = "I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful"
    profile, strategy, scenes = _build_story_scenes(input_text)

    assert profile.situation_id in ["relationship_ambivalence", "loss_and_grief"]
    assert len(scenes) >= 3

    combined_text = " ".join(
        f"{s.get('dialogue', '')} {s.get('caption', '')} {s.get('narrative', '')}"
        for s in scenes
    ).lower()

    for term in BANNED_PERFORMANCE_TERMS:
        assert term not in combined_text, f"Breakup leaked performance term: '{term}'"

    for term in BANNED_OVERWHELM_TERMS:
        assert term not in combined_text, f"Breakup leaked overwhelm term: '{term}'"

    # Must address both sadness/caring and honest uncertainty/doubts
    assert any(w in combined_text for w in ["pain", "care", "grief", "sad"])
    assert any(w in combined_text for w in ["doubt", "uncertain", "not sure", "ambivalence"])


def test_dialogue_and_captions_have_no_mid_word_truncations():
    """Ensures dialogue and captions never chop words mid-word across all 4 key scenarios."""
    test_cases = [
        "I don't know what will happen with my future and career, and the constant uncertainty is making me anxious and terrified.",
        "Everyone my age seems to be succeeding and doing so much better than me, while I feel like a total failure and stuck.",
        "I have 20 things to do today and I feel completely overwhelmed and paralyzed, I don't know where to start.",
        "I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful",
    ]

    for text in test_cases:
        _, _, scenes = _build_story_scenes(text)
        for s in scenes:
            dia = s.get("dialogue", "")
            cap = s.get("caption", "")

            # Check that dialogue does not end with incomplete word or trailing ellipsis
            assert not dia.endswith("..."), f"Dialogue ends with trailing ellipsis: '{dia}'"
            assert not cap.endswith("..."), f"Caption ends with trailing ellipsis: '{cap}'"
            # Dialogue and caption must end with valid punctuation (. ! ?)
            assert dia[-1] in ".!?'\"", f"Dialogue ends abruptly without punctuation: '{dia}'"
            assert cap[-1] in ".!?'\"", f"Caption ends abruptly without punctuation: '{cap}'"
            # Specifically check known truncation artifacts
            assert "automatical" not in dia.lower(), f"Mid-word truncation found in: '{dia}'"
            assert "automatical" not in cap.lower(), f"Mid-word truncation found in: '{cap}'"

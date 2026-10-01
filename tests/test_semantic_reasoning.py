"""
Comprehensive semantic relevance and psychological reasoning tests.

Verifies:
1. Exact user benchmark: "I had a breakup because of my uncertainty and also I wasn't
   sure I think I did not love him it's kind of painful".
   - Accurately classifies as relationship ambivalence & separation grief.
   - Rejects performance/failure anxiety categorization.
   - Preserves uncertainty without falsely asserting the breakup was definitely right or wrong.
   - Decouples emotional pain from proof that the relationship was a mistake.
   - Populates all 8 mandatory structured reasoning fields.
   - Generates comic scenes and dialogues directly grounded in the breakup dilemma.
2. Cross-domain contamination prevention:
   - Performance terms (fail, exam, interview) never leak into relational situations.
   - Relationship terms never leak into performance situations.
   - False certainty in ambivalence scenarios is rejected.
   - Generic cliches ("feeling nervous doesn't mean I will fail") are strictly blocked.
3. Distinct psychological situations remain strictly isolated and situation-specific.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.psychology.situation_parser import parse_situation
from app.psychology.strategy_selector import select_strategy
from app.psychology.semantic_validator import SemanticValidator
from app.context_engine.case_frame import CaseFrame
from app.narrative_gen.generator import NarrativeGenerator


def test_breakup_uncertainty_exact_benchmark():
    """
    Validates the exact benchmark prompt:
    'I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful'
    """
    text = "I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful"
    profile = parse_situation(text)

    # 1. Situation & Pattern identification
    assert profile.situation_id == "relationship_ambivalence"
    assert profile.is_external_threat is False
    assert "breakup" in profile.situation.lower() or "relationship" in profile.situation.lower()

    # 2. Strategy selection
    strategy = select_strategy(profile)
    assert strategy.id == "relationship_ambivalence_grief"
    assert "Ambivalence-Informed Grief & Separation Acceptance" in strategy.name

    # 3. All 8 Structured Reasoning Fields must be populated
    reasoning = profile.structured_reasoning()
    required_fields = [
        "situation",
        "primary_emotion",
        "central_conflict",
        "automatic_thought",
        "pattern",
        "coping_strategy",
        "specific_reframe",
        "concrete_next_step",
    ]
    for field in required_fields:
        assert field in reasoning, f"Missing required reasoning field: {field}"
        assert len(reasoning[field]) > 0, f"Reasoning field {field} is empty"

    # 4. Central conflict & automatic thought must directly capture the breakup dilemma
    assert "uncertainty" in reasoning["central_conflict"].lower() or "mistake" in reasoning["central_conflict"].lower()
    assert "pain" in reasoning["central_conflict"].lower() or "separation" in reasoning["central_conflict"].lower()

    # 5. Semantic validity of the reframe
    reframe = reasoning["specific_reframe"]
    reframe_lowered = reframe.lower()

    # Decouples pain from proof of mistake
    assert "pain" in reframe_lowered or "hurt" in reframe_lowered
    assert "prove" in reframe_lowered or "mean" in reframe_lowered or "mistake" in reframe_lowered

    # Preserves uncertainty: does NOT declare the breakup definitely right or wrong
    assert "definitely the right" not in reframe_lowered
    assert "definitely correct" not in reframe_lowered
    assert "guaranteed right" not in reframe_lowered

    # Holds space for doubts / caring
    assert "cared" in reframe_lowered or "loss" in reframe_lowered
    assert "uncertainty" in reframe_lowered or "doubts" in reframe_lowered

    # 6. Absence of cross-domain contamination
    banned_performance_terms = ["exam", "interview", "fail", "freeze", "study", "evaluator", "talking points"]
    for term in banned_performance_terms:
        assert term not in reframe_lowered, f"Cross-domain term '{term}' leaked into breakup reframe"

    # 7. Assemble CaseFrame and generate comic narrative
    case_frame = CaseFrame(
        raw_text=text,
        core_emotion="sadness",
        situation_type=profile.situation_id,
        trigger=profile.trigger,
        pattern=profile.pattern,
        is_external_threat=profile.is_external_threat,
        controllable=strategy.what_is_controllable,
        uncontrollable=strategy.what_is_not_controllable,
        strategy=strategy.to_dict(),
        specific_reframe=strategy.reframe,
        concrete_action=strategy.concrete_action,
        reasoning=reasoning,
    )
    case_frame.build_summary()

    technique = {"name": strategy.name, "citation": strategy.citation, "description": strategy.mechanism}
    generator = NarrativeGenerator(use_small=True)
    try:
        story = generator.generate(case_frame, technique)
    finally:
        generator.unload()

    # 8. Verify generated comic scenes and dialogues are directly relevant
    scenes = story["scenes"]
    assert len(scenes) >= 3

    all_dialogues = " ".join(s.get("dialogue", "") for s in scenes).lower()
    all_narratives = " ".join(s.get("narrative", "") for s in scenes).lower()

    # Must NOT have generic exam/performance anxiety dialogue
    assert "feeling nervous doesn't mean i will fail" not in all_dialogues
    assert "i will fail" not in all_dialogues
    assert "focus on what i control" not in all_dialogues or "pain" in all_dialogues or "care" in all_dialogues

    # Must reflect relationship / breakup / uncertainty / grief
    relational_markers = ["hurt", "pain", "uncertain", "loved", "breakup", "care", "grieve", "doubts"]
    found_relational = any(m in all_dialogues or m in all_narratives for m in relational_markers)
    assert found_relational, f"Story failed to reflect relationship ambivalence: {all_dialogues}"


def test_semantic_validator_cross_domain_detection():
    """Verify SemanticValidator flags cross-domain contamination and generic cliches."""
    # Contaminated relationship reframe
    invalid_reframe = "Feeling nervous doesn't mean I will fail my interview. I can review my talking points."
    is_valid, reason = SemanticValidator.validate_reframe(
        raw_text="I had a breakup because of my uncertainty",
        situation_id="relationship_ambivalence",
        central_conflict="Wrestling with breakup doubts",
        reframe=invalid_reframe,
        dialogues=["Feeling nervous doesn't mean I will fail."],
    )
    assert is_valid is False
    assert "Cross-domain" in reason or "cliche" in reason

    # False certainty in ambivalence
    false_certainty_reframe = "You made definitely the right choice in breaking up with him."
    is_valid, reason = SemanticValidator.validate_reframe(
        raw_text="I had a breakup because of my uncertainty",
        situation_id="relationship_ambivalence",
        central_conflict="Wrestling with breakup doubts",
        reframe=false_certainty_reframe,
    )
    assert is_valid is False
    assert "False certainty" in reason or "Incomplete" in reason

    # Toxic positivity in interpersonal threat
    toxic_reframe = "Just focus on gratitude and think positive thoughts about him."
    is_valid, reason = SemanticValidator.validate_reframe(
        raw_text="My father lashes out and later acts normal",
        situation_id="interpersonal_threat",
        central_conflict="Unpredictable hostility",
        reframe=toxic_reframe,
    )
    assert is_valid is False
    assert "positivity" in reason.lower()


def test_distinct_situations_remain_strictly_isolated():
    """Verify other benchmark situations maintain strict domain specificity."""
    situations = [
        {
            "text": "I'm going to fail my interview tomorrow",
            "expected_id": "performance_anxiety",
            "forbidden_terms": ["breakup", "partner", "grief", "father"],
            "required_terms": ["interview", "preparation", "fail", "freeze"],
        },
        {
            "text": "My father lashes out and later behaves normally",
            "expected_id": "interpersonal_threat",
            "forbidden_terms": ["interview", "grades", "exam", "think positive"],
            "required_terms": ["volatility", "boundaries", "safety", "erasing"],
        },
        {
            "text": "I have 20 things to do and can't start",
            "expected_id": "overwhelm",
            "forbidden_terms": ["breakup", "partner", "father", "interview"],
            "required_terms": ["micro-action", "momentum", "one", "timer", "task"],
        },
        {
            "text": "Everyone is doing better than me",
            "expected_id": "social_comparison",
            "forbidden_terms": ["breakup", "interview", "father"],
            "required_terms": ["comparison", "timeline", "values", "highlights"],
        },
        {
            "text": "I messed up once, I'm useless",
            "expected_id": "self_criticism",
            "forbidden_terms": ["breakup", "interview", "father"],
            "required_terms": ["mistake", "worth", "event", "compassion"],
        },
    ]

    for item in situations:
        profile = parse_situation(item["text"])
        assert profile.situation_id == item["expected_id"]
        strategy = select_strategy(profile)

        reframe_and_action = (strategy.reframe + " " + strategy.concrete_action + " " + strategy.mechanism).lower()

        # Check forbidden terms
        for term in item["forbidden_terms"]:
            assert term not in reframe_and_action, f"Forbidden term '{term}' found in {item['expected_id']}"

        # Check at least some required domain markers are present
        found_marker = any(m in reframe_and_action for m in item["required_terms"])
        assert found_marker, f"No required domain markers found for {item['expected_id']}"

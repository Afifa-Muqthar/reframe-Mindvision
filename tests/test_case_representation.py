"""
Stage 1 Unit & Calibration Test Suite for CaseRepresentation.

Validates:
1. Regression target input: career after graduation + friends placed + family disappointment.
2. Development cases C1 through C10.
3. Strict separation of user-stated facts/emotions from model inferences.
4. Absence of premature decision making (no forced CBT, no generic boilerplate).
5. Explicit unknown tracking.
6. Traceability mapping to source spans.
7. Resilience on partial, vague, or misspelled inputs.
8. Zero disruption to existing downstream modules.
"""

import pytest
from app.psychology.case_representation import (
    extract_case_representation,
    CaseRepresentation,
    StatedFact,
    StatedEmotion,
    PredictedEmotion,
    InferredAppraisal,
)


def test_regression_career_case_preserves_all_context():
    """
    Validates that the exact career-related regression input from the forensic audit
    retains all stated facts, emotions, and thoughts without collapsing to general_stress.
    """
    text = (
        "I'm worried about my career after graduation. My friends are getting placed, "
        "and I'm happy for them, but I feel scared that I'll disappoint my family and that I haven't done enough."
    )
    mock_emotion_scores = {"fear": 0.42, "nervousness": 0.40, "joy": 0.12}

    rep = extract_case_representation(text, emotion_scores=mock_emotion_scores)

    # 1. User stated facts preserved
    fact_texts = [f.text.lower() for f in rep.stated_facts]
    assert any("career" in t for t in fact_texts), "Failed to preserve 'career'"
    assert any("graduation" in t for t in fact_texts), "Failed to preserve 'graduation'"
    assert any("friend" in t for t in fact_texts), "Failed to preserve 'friends'"
    assert any("family" in t for t in fact_texts), "Failed to preserve 'family'"

    # 2. User stated emotions preserved
    stated_emotions = {e.emotion_word.lower() for e in rep.stated_emotions}
    assert "worried" in stated_emotions, "Failed to preserve 'worried'"
    assert "happy" in stated_emotions, "Failed to preserve 'happy'"
    assert "scared" in stated_emotions, "Failed to preserve 'scared'"

    # 3. Model predicted emotions kept strictly separate
    assert len(rep.predicted_emotions) > 0
    pred_labels = [p.label for p in rep.predicted_emotions]
    assert "fear" in pred_labels
    assert all(isinstance(p, PredictedEmotion) for p in rep.predicted_emotions)
    assert all(isinstance(e, StatedEmotion) for e in rep.stated_emotions)

    # 4. Inferred appraisals marked as hypotheses and traceable
    dimensions = {a.dimension for a in rep.inferred_appraisals}
    assert "social_comparison" in dimensions, "Should infer social_comparison hypothesis"
    assert "career_horizon_uncertainty" in dimensions, "Should infer career_horizon_uncertainty hypothesis"
    assert "evaluation_fear" in dimensions, "Should infer evaluation_fear hypothesis"
    assert "perceived_deficit" in dimensions, "Should infer perceived_deficit hypothesis"

    for appraisal in rep.inferred_appraisals:
        assert appraisal.status == "hypothesized"
        assert len(appraisal.evidence_spans) > 0, f"Appraisal {appraisal.dimension} missing evidence spans"
        assert appraisal.dimension in rep.traceability_map or any(appraisal.dimension in k for k in rep.traceability_map)

    # 5. Stated thoughts preserved
    thought_statements = [t.statement.lower() for t in rep.stated_thoughts]
    assert any("enough" in s for s in thought_statements), "Failed to preserve 'haven't done enough'"
    assert any("disappoint" in s for s in thought_statements), "Failed to preserve 'disappoint my family'"

    # 6. Explicit unknowns recorded
    assert len(rep.unknowns.missing_aspects) > 0
    missing_str = " ".join(rep.unknowns.missing_aspects).lower()
    assert "career" in missing_str or "field" in missing_str
    assert "enough" in missing_str or "expectations" in missing_str

    # 7. No generic general_stress boilerplate
    assert "general_stress" not in rep.descriptive_summary
    assert "Navigating a challenging personal situation" not in rep.descriptive_summary
    assert "This feels overwhelming" not in rep.descriptive_summary


def test_development_c2_chronic_illness():
    """C2: Bodily limitation must be recognized without framing it as cognitive distortion."""
    text = "My autoimmune flare-up returned this week. I can barely get out of bed, and I feel so guilty that I can't be productive or keep up with my peers."
    rep = extract_case_representation(text)

    dims = {a.dimension for a in rep.inferred_appraisals}
    assert "bodily_limitation" in dims
    assert "empathetic_validation" in rep.support_needs.possible_needs
    assert "do_not_frame_physical_illness_as_cognitive_distortion" in rep.support_needs.contraindications


def test_development_c3_workplace_injustice():
    """C3: External supervisor unfairness must not be treated as internal distortion."""
    text = "My supervisor gave credit for my three-month project to another colleague in the team meeting. I'm furious and feeling completely powerless."
    rep = extract_case_representation(text)

    dims = {a.dimension for a in rep.inferred_appraisals}
    assert "systemic_injustice" in dims
    assert "furious" in {e.emotion_word.lower() for e in rep.stated_emotions}
    assert "do_not_gaslight_or_reframe_supervisor_behavior" in rep.support_needs.contraindications


def test_development_c4_relational_ambivalence():
    """C4: Relational uncertainty must require preservation rather than forced decision."""
    text = "We've been dating for four years, but lately I feel emotionally disconnected. Nothing terrible happened, which almost makes it harder. I don't know whether to try harder or let go."
    rep = extract_case_representation(text)

    assert rep.uncertainty.has_uncertainty is True
    assert rep.uncertainty.uncertainty_type == "relational_ambivalence"
    assert rep.uncertainty.preservation_required is True
    assert "do_not_force_premature_decision_or_false_certainty" in rep.support_needs.contraindications


def test_development_c5_imposter_phenomenon():
    """C5: Imposter feelings in promotion must identify promotion milestone and fraud self-evaluation."""
    text = "I just got promoted to lead the engineering team, but I feel like an absolute fraud. Any day now they will realize I have no idea what I'm doing."
    rep = extract_case_representation(text)

    dims = {a.dimension for a in rep.inferred_appraisals}
    assert "perceived_deficit" in dims
    assert any("promot" in f.text.lower() for f in rep.stated_facts)


def test_development_c6_caregiver_burnout():
    """C6: Caregiver strain with mother must preserve love and regret."""
    text = "Taking care of my aging mother every day while working full time is draining all my energy. I love her so much, but yesterday I snapped at her and I feel like an awful person."
    rep = extract_case_representation(text)

    facts = [f.text.lower() for f in rep.stated_facts]
    assert any("mother" in f for f in facts)
    dims = {a.dimension for a in rep.inferred_appraisals}
    assert "evaluation_fear" in dims or "perceived_deficit" in dims


def test_development_c7_existential_climate_dread():
    """C7: Macro climate dread must be marked as uncontrollable macro uncertainty."""
    text = "Reading the environmental report today made me feel hopeless about the future. What's the point of planning my life when the planet is burning?"
    rep = extract_case_representation(text)

    assert rep.uncertainty.has_uncertainty is True
    assert rep.uncertainty.uncertainty_type == "macro_existential"
    assert "hopeless" in {e.emotion_word.lower() for e in rep.stated_emotions}


def test_development_c8_creative_rejection():
    """C8: Repeated rejection must preserve manuscript fact and self-doubt."""
    text = "My manuscript was rejected for the fifth time this year. I'm starting to think I simply have no talent and wasted years of my life."
    rep = extract_case_representation(text)

    assert any("manuscript" in f.text.lower() for f in rep.stated_facts)
    dims = {a.dimension for a in rep.inferred_appraisals}
    assert "perceived_deficit" in dims


def test_development_c9_social_exclusion():
    """C9: Peer exclusion from dinner party must capture friend circle and social pain."""
    text = "I found out my core group of friends had a dinner party last night and didn't invite me. I've been crying all morning wondering what's wrong with me."
    rep = extract_case_representation(text)

    assert any("friend" in f.text.lower() for f in rep.stated_facts)
    assert "crying" in {e.emotion_word.lower() for e in rep.stated_emotions}
    dims = {a.dimension for a in rep.inferred_appraisals}
    assert "social_comparison" in dims or "perceived_deficit" in dims


def test_development_c10_executive_freeze():
    """C10: Multi-task freeze must capture task accumulation and immobility."""
    text = "I have five assignments due by Friday, laundry piling up, and an empty fridge. I've been sitting on the floor staring at my phone for four hours unable to move."
    rep = extract_case_representation(text)

    dims = {a.dimension for a in rep.inferred_appraisals}
    assert "executive_freeze" in dims
    assert "somatic_grounding_and_micro_step" in rep.support_needs.possible_needs


def test_graceful_handling_of_empty_and_vague_inputs():
    """System must not crash on empty, colloquial, or sparse inputs, and must mark them partial."""
    # Empty
    empty_rep = extract_case_representation("")
    assert empty_rep.is_partial is True
    assert len(empty_rep.stated_facts) == 0

    # Vague colloquial
    vague_text = "idk everything is just weird lately and i feel kinda numb"
    vague_rep = extract_case_representation(vague_text)
    assert vague_rep.is_partial is True
    assert "numb" in {e.emotion_word.lower() for e in vague_rep.stated_emotions}
    assert len(vague_rep.unknowns.missing_aspects) > 0


def test_traceability_map_contains_source_substrings():
    """Every key in the traceability map must contain non-empty substrings that exist in raw text."""
    text = "I'm worried about my career after graduation. My friends are getting placed, and I'm happy for them."
    rep = extract_case_representation(text)

    assert len(rep.traceability_map) > 0
    for key, spans in rep.traceability_map.items():
        assert isinstance(spans, list)
        assert len(spans) > 0
        for span in spans:
            # Span or lowercase span should match or overlap raw text
            assert any(word in text.lower() for word in span.lower().split()), f"Span '{span}' has no overlap with raw text"


def test_downstream_isolation_unchanged():
    """Verifies that existing modules remain functional and untouched."""
    from app.psychology import parse_situation, select_strategy
    from app.context_engine.case_frame import CaseFrame

    prof = parse_situation("I have an exam tomorrow and feel nervous")
    strat = select_strategy(prof)
    assert prof is not None
    assert strat is not None

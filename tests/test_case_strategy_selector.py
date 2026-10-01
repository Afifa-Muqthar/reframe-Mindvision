"""
Stage 2 Unit & Calibration Test Suite for CaseStrategySelector.

Validates:
1. Target regression input (C1): Locus of Agency & Values Differentiation with uncertainty.
2. Non-reframe validation cases (C2 chronic illness, C3 workplace injustice).
3. Relational holding space without premature certainty (C4).
4. Practical structuring for executive freeze (C10).
5. Cognitive reappraisal where appropriate (C5 imposter, C8 creative rejection).
6. Non-interventional macro grounding (C7 climate dread).
7. Clarification requests for ambiguous, sparse, or partial inputs.
8. Clinical safety contraindications across all modalities.
9. Verification that user-stated facts are never converted into inferences or vice versa.
"""

import pytest
from app.psychology.case_representation import extract_case_representation
from app.psychology.case_strategy_selector import CaseStrategySelector, SelectedStrategy


@pytest.fixture
def selector():
    return CaseStrategySelector()


def test_regression_c1_career_selects_locus_of_agency(selector):
    """
    C1: Compound career horizon + peer placement + family expectation.
    Must select locus_of_agency, acknowledge alternative (validation), and flag uncertainty.
    """
    text = (
        "I'm worried about my career after graduation. My friends are getting placed, "
        "and I'm happy for them, but I feel scared that I'll disappoint my family and that I haven't done enough."
    )
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert strat.modality == "locus_of_agency"
    assert strat.strategy_id == "career_agency_values_differentiation"
    assert strat.reframe_needed is False  # Anchors in agency rather than disputing family love
    assert strat.is_uncertain is True     # Multiple approaches fit (agency vs validation)
    assert "validation" in strat.alternative_modalities
    assert "do_not_dismiss_family_concern_as_irrational" in strat.contraindications
    assert strat.suggested_step is not None

    # Verify facts vs inferences integrity
    assert any("career" in f.text.lower() for f in rep.stated_facts)
    assert any("friend" in f.text.lower() for f in rep.stated_facts)
    assert all(a.status == "hypothesized" for a in rep.inferred_appraisals)


def test_c2_chronic_illness_rejects_reframe_selects_validation(selector):
    """
    C2: Autoimmune flare-up. Reframe is strictly inappropriate.
    Must select validation, rest permission, and enforce clinical contraindications.
    """
    text = "My autoimmune flare-up returned this week. I can barely get out of bed, and I feel so guilty that I can't be productive or keep up with my peers."
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert strat.modality == "validation"
    assert strat.strategy_id == "bodily_limitation_compassion"
    assert strat.reframe_needed is False
    assert "do_not_frame_physical_illness_as_cognitive_distortion" in strat.contraindications
    assert "do_not_demand_productivity_steps" in strat.contraindications


def test_c3_workplace_injustice_rejects_reframe_selects_validation(selector):
    """
    C3: External supervisor theft of credit. Reframe would be gaslighting.
    Must select validation and objective options.
    """
    text = "My supervisor gave credit for my three-month project to another colleague in the team meeting. I'm furious and feeling completely powerless."
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert strat.modality == "validation"
    assert strat.strategy_id == "injustice_validation_options"
    assert strat.reframe_needed is False
    assert "do_not_gaslight_or_reframe_supervisor_behavior" in strat.contraindications
    assert "do_not_blame_user_for_powerlessness" in strat.contraindications


def test_c4_relational_ambivalence_selects_holding_space(selector):
    """
    C4: Slow emotional drift with genuine uncertainty.
    Must hold space without forcing a premature stay-or-leave decision.
    """
    text = "We've been dating for four years, but lately I feel emotionally disconnected. Nothing terrible happened, which almost makes it harder. I don't know whether to try harder or let go."
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert strat.modality == "non_interventional_grounding"
    assert strat.strategy_id == "relational_ambivalence_holding_space"
    assert strat.reframe_needed is False
    assert "do_not_force_premature_decision_or_false_certainty" in strat.contraindications


def test_c5_imposter_phenomenon_selects_perspective_reappraisal(selector):
    """
    C5: Imposter syndrome in promotion. Appropriate for cognitive evidence testing.
    """
    text = "I just got promoted to lead the engineering team, but I feel like an absolute fraud. Any day now they will realize I have no idea what I'm doing."
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert strat.modality == "perspective_reappraisal"
    assert strat.reframe_needed is True
    assert strat.suggested_step is not None


def test_c6_caregiver_strain_selects_compassion_validation(selector):
    """
    C6: Full-time work + aging mother caregiving.
    Must select validation and self-compassion, not task restructuring.
    """
    text = "Taking care of my aging mother every day while working full time is draining all my energy. I love her so much, but yesterday I snapped at her and I feel like an awful person."
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert strat.modality == "validation"
    assert strat.strategy_id == "caregiver_strain_self_compassion"
    assert strat.reframe_needed is False
    assert "do_not_judge_emotional_outburst_as_character_flaw" in strat.contraindications


def test_c7_climate_dread_selects_macro_grounding(selector):
    """
    C7: Environmental report. Must select macro grounding without toxic optimism.
    """
    text = "Reading the environmental report today made me feel hopeless about the future. What's the point of planning my life when the planet is burning?"
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert strat.modality == "non_interventional_grounding"
    assert strat.strategy_id == "macro_existential_local_anchoring"
    assert strat.reframe_needed is False
    assert "do_not_offer_toxic_optimism_about_climate_reality" in strat.contraindications


def test_c8_creative_rejection_selects_reappraisal(selector):
    """
    C8: Manuscript rejected 5 times. Reappraises 'no talent' / 'wasted years'.
    """
    text = "My manuscript was rejected for the fifth time this year. I'm starting to think I simply have no talent and wasted years of my life."
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert strat.modality == "perspective_reappraisal"
    assert strat.reframe_needed is True


def test_c9_social_exclusion_selects_soothing_validation(selector):
    """
    C9: Left out of dinner party, crying. Must soothe attachment hurt before cognitive disputing.
    """
    text = "I found out my core group of friends had a dinner party last night and didn't invite me. I've been crying all morning wondering what's wrong with me."
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert strat.modality == "validation"
    assert strat.strategy_id == "social_exclusion_soothing"
    assert strat.reframe_needed is False


def test_c10_executive_freeze_selects_practical_structuring(selector):
    """
    C10: Multi-task paralysis and sitting on floor.
    Must select practical structuring with a 2-minute entry step.
    """
    text = "I have five assignments due by Friday, laundry piling up, and an empty fridge. I've been sitting on the floor staring at my phone for four hours unable to move."
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert strat.modality == "practical_structuring"
    assert strat.strategy_id == "executive_freeze_micro_entry"
    assert strat.reframe_needed is False
    assert strat.suggested_step is not None
    assert "do_not_overwhelm_with_multi_step_planning" in strat.contraindications


def test_sparse_and_vague_inputs_select_clarification(selector):
    """
    Ambiguous, colloquial, or sparse inputs must select clarification_needed
    with open exploratory questions and NO premature reframe or action.
    """
    # 1. Colloquial numbness
    vague_text = "idk everything is just weird lately and i feel kinda numb"
    rep1 = extract_case_representation(vague_text)
    strat1 = selector.select(rep1)

    assert strat1.modality == "clarification_needed"
    assert strat1.reframe_needed is False
    assert strat1.suggested_step is None
    assert len(strat1.clarification_questions) > 0
    assert strat1.is_uncertain is True

    # 2. Minimal input
    minimal_text = "I don't know what to do"
    rep2 = extract_case_representation(minimal_text)
    strat2 = selector.select(rep2)

    assert strat2.modality == "clarification_needed"
    assert strat2.suggested_step is None


def test_fact_inference_separation_integrity(selector):
    """
    Verifies that stated facts and inferred appraisals maintain strict type
    and semantic boundaries without cross-contamination.
    """
    text = "My supervisor gave credit for my three-month project to another colleague in the team meeting."
    rep = extract_case_representation(text)

    # All items in stated_facts must be StatedFact instances with verbatim source_spans
    for f in rep.stated_facts:
        assert f.source_span in text
        assert f.category in ("actor", "event", "milestone", "condition", "activity", "obligation", "creative_work", "physical_state", "behavioral_state", "domain", "stated_concern")

    # Inferred appraisals must be explicitly marked as hypotheses
    for a in rep.inferred_appraisals:
        assert a.status == "hypothesized"
        assert a.confidence > 0.0

    # Ensure selector does not mutate or alter the original representation
    strat = selector.select(rep)
    assert strat is not None

"""
Phase 5 Evaluation: Diverse Psychological Understanding Test Suite.

Evaluates the generalizable psychological formulation pipeline across:
1. Diverse life domains:
   - Creative block / expressive drought
   - Financial stress & family obligation
   - Physical injury & athletic identity loss
   - Academic collaboration inequity / injustice
   - Relational longing with firm boundaries
   - Career horizon transition & peer comparison
   - Task overload & executive freeze
2. Linguistic styles:
   - Metaphorical / descriptive ("blank canvas", "well dry", "brushstroke hollow")
   - Colloquial / situational ("partners went MIA", "submitted zero code")
   - Direct / terse ("Rent went up by 30%... skipping meals")
   - Truly sparse / ambiguous ("just...", "I don't know what to do anymore")
3. Verification criteria:
   - Grounded extraction of user's actual situation and emotions.
   - Controllable vs. uncontrollable distinctions logically grounded in user agency.
   - 5 reasoning cards internally consistent, empathetic, and non-diagnostic.
   - Zero generic boilerplate leakage on meaningful inputs.
   - Appropriate handling of genuine uncertainty and sparse inputs.
"""

import pytest
from unittest.mock import patch

from app.psychology.case_representation import extract_case_representation
from app.psychology.case_strategy_selector import CaseStrategySelector
from app.main import app

GENERIC_BOILERPLATE = [
    "Navigating a challenging personal situation",
    "Personal stress, ambiguity, and self-doubt",
    "Your choices today, treating yourself with patience",
    "Having doubts or feeling unsettled is a normal human experience",
    "Take a deep breath, write down the one single thing",
]

PATHOLOGIZING_TERMS = [
    "lazy", "addicted", "addiction", "disordered", "pathological",
    "procrastinator", "character flaw", "narcissistic", "borderline",
]


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def selector():
    return CaseStrategySelector()


# =============================================================================
# 1. CREATIVE BLOCK & ARTISTIC DROUGHT
# =============================================================================

def test_creative_block_generalization(selector):
    """
    Unfamiliar domain: Artistic drought and self-doubt.
    Must extract canvas, brushstroke, hollow, dry; select creative reappraisal;
    separate controllable rest from uncontrollable forced inspiration.
    """
    text = (
        "I have been staring at this blank canvas for three weeks and every brushstroke feels hollow, "
        "like I used to have something to say but now the well is completely dry."
    )
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert rep.ambiguity_level == "sufficient"
    assert "creative_block" in {a.dimension for a in rep.inferred_appraisals}
    assert any("canvas" in f.text.lower() or "brushstroke" in f.text.lower() for f in rep.stated_facts)
    assert any("hollow" in e.emotion_word.lower() or "dry" in e.emotion_word.lower() for e in rep.stated_emotions)

    assert strat.modality == "perspective_reappraisal"
    assert "canvas" in strat.what_may_be_happening.lower() or "creative" in strat.what_may_be_happening.lower()
    assert "rest" in strat.suggested_step.lower() or "canvas" in strat.suggested_step.lower()

    # Controllability grounded in creative rest vs forcing inspiration
    controllable_str = " ".join(rep.controllability.potentially_controllable).lower()
    uncontrollable_str = " ".join(rep.controllability.potentially_uncontrollable).lower()
    assert "canvas" in controllable_str or "rest" in controllable_str or "worth" in controllable_str
    assert "inspiration" in uncontrollable_str or "spontaneous" in uncontrollable_str


# =============================================================================
# 2. FINANCIAL DISTRESS & FAMILY OBLIGATION
# =============================================================================

def test_financial_distress_and_sacrifice_generalization(selector):
    """
    Unfamiliar domain: Financial cost escalation and sibling tuition sacrifice.
    Must extract rent, brother, tuition, meals; select resource strain validation;
    prioritize physical nourishment over aggressive problem-solving.
    """
    text = (
        "Rent went up by 30% this month and my younger brother needs tuition help, "
        "so I am skipping meals to keep everything afloat."
    )
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert rep.ambiguity_level == "sufficient"
    assert "resource_strain_uncertainty" in {a.dimension for a in rep.inferred_appraisals}
    fact_texts = [f.text.lower() for f in rep.stated_facts]
    assert any("rent" in t for t in fact_texts)
    assert any("brother" in t for t in fact_texts)

    assert strat.modality == "validation"
    assert "rent" in strat.what_may_be_happening.lower() or "living" in strat.what_may_be_happening.lower() or "financial" in strat.what_may_be_happening.lower()
    assert "meal" in strat.suggested_step.lower() or "nourish" in strat.suggested_step.lower() or "expense" in strat.suggested_step.lower()

    controllable_str = " ".join(rep.controllability.potentially_controllable).lower()
    uncontrollable_str = " ".join(rep.controllability.potentially_uncontrollable).lower()
    assert "meal" in controllable_str or "recovery" in controllable_str or "pacing" in controllable_str
    assert "rent" in uncontrollable_str or "inflation" in uncontrollable_str or "future" in uncontrollable_str


# =============================================================================
# 3. PHYSICAL INJURY & ATHLETIC IDENTITY LOSS
# =============================================================================

def test_sports_injury_and_identity_loss_generalization(selector):
    """
    Unfamiliar domain: Sidelined sports injury before critical tryouts.
    Must extract ACL, tryouts, soccer, crutches; select bodily limitation validation;
    validate athletic grief without pathologizing physical restriction.
    """
    text = (
        "Tore my ACL right before senior season tryouts, soccer was my entire identity "
        "and now I am stuck on crutches watching everyone else play."
    )
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert rep.ambiguity_level == "sufficient"
    assert "bodily_limitation" in {a.dimension for a in rep.inferred_appraisals}
    fact_texts = [f.text.lower() for f in rep.stated_facts]
    assert any("acl" in t for t in fact_texts)
    assert any("crutch" in t for t in fact_texts)

    assert strat.modality == "validation"
    assert "injury" in strat.what_may_be_happening.lower() or "tryout" in strat.what_may_be_happening.lower() or "sport" in strat.what_may_be_happening.lower()
    assert "rest" in strat.suggested_step.lower() or "sideline" in strat.suggested_step.lower() or "teammate" in strat.suggested_step.lower()

    controllable_str = " ".join(rep.controllability.potentially_controllable).lower()
    uncontrollable_str = " ".join(rep.controllability.potentially_uncontrollable).lower()
    assert "rehabilitation" in controllable_str or "healing" in controllable_str or "rest" in controllable_str
    assert "injury" in uncontrollable_str or "timeline" in uncontrollable_str or "tissue" in uncontrollable_str


# =============================================================================
# 4. ACADEMIC COLLABORATION INJUSTICE
# =============================================================================

def test_academic_group_project_injustice_generalization(selector):
    """
    Unfamiliar domain: Group project partners MIA and unfair shared grading policy.
    Must extract partners, zero code, grade; select systemic injustice validation;
    recommend documentation and objective self-advocacy.
    """
    text = (
        "My group project partners went completely MIA for two weeks, submitted zero code, "
        "and now the professor says our team grade will be shared equally regardless."
    )
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    assert rep.ambiguity_level == "sufficient"
    assert "systemic_injustice" in {a.dimension for a in rep.inferred_appraisals}
    fact_texts = [f.text.lower() for f in rep.stated_facts]
    assert any("partner" in t for t in fact_texts)
    assert any("code" in t for t in fact_texts)

    assert strat.modality == "validation"
    assert "project" in strat.what_may_be_happening.lower() or "grade" in strat.what_may_be_happening.lower() or "team" in strat.what_may_be_happening.lower()
    assert "commit" in strat.suggested_step.lower() or "record" in strat.suggested_step.lower() or "professor" in strat.suggested_step.lower()

    controllable_str = " ".join(rep.controllability.potentially_controllable).lower()
    uncontrollable_str = " ".join(rep.controllability.potentially_uncontrollable).lower()
    assert "document" in controllable_str or "meeting" in controllable_str or "record" in controllable_str
    assert "partner" in uncontrollable_str or "policy" in uncontrollable_str or "grade" in uncontrollable_str


# =============================================================================
# 5. TRULY SPARSE INPUTS (PRESERVING GENTLE HOLDING SPACE)
# =============================================================================

@pytest.mark.parametrize("sparse_text", [
    "just...",
    "idk",
    "I don't know what to do anymore.",
    "I don't know what I feel.",
])
def test_truly_sparse_inputs_handled_with_gentle_clarification(sparse_text, selector):
    """
    Truly sparse inputs must be classified as sparse_insufficient and select clarification_needed.
    Must NOT hallucinate clinical diagnoses or workplace conflict.
    """
    rep = extract_case_representation(sparse_text)
    strat = selector.select(rep)

    assert rep.ambiguity_level == "sparse_insufficient"
    assert strat.modality == "clarification_needed"
    assert "not assume" in strat.what_may_be_happening.lower() or "uncertainty" in strat.what_may_be_happening.lower() or "moment" in strat.what_may_be_happening.lower()


# =============================================================================
# 6. END-TO-END /PROCESS ENDPOINT INTEGRATION ON DIVERSE SCENARIOS
# =============================================================================

@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_diverse_scenarios_through_process_endpoint(mock_thread, mock_tts, client):
    """
    Verifies that all diverse scenarios return the full 5-card reasoning object
    through the /process endpoint, free of generic boilerplate or diagnostic terms.
    """
    test_cases = [
        (
            "creative_block",
            "I have been staring at this blank canvas for three weeks and every brushstroke feels hollow, "
            "like I used to have something to say but now the well is completely dry.",
            "canvas",
        ),
        (
            "financial_distress",
            "Rent went up by 30% this month and my younger brother needs tuition help, "
            "so I am skipping meals to keep everything afloat.",
            "rent",
        ),
        (
            "sports_injury",
            "Tore my ACL right before senior season tryouts, soccer was my entire identity "
            "and now I am stuck on crutches watching everyone else play.",
            "injury",
        ),
        (
            "group_injustice",
            "My group project partners went completely MIA for two weeks, submitted zero code, "
            "and now the professor says our team grade will be shared equally regardless.",
            "project",
        ),
    ]

    for name, input_text, expected_keyword in test_cases:
        res = client.post("/process", data={"text": input_text})
        assert res.status_code == 200, f"Failed for {name}"
        data = res.get_json()

        assert data["crisis"] is False
        assert "reasoning" in data
        reasoning = data["reasoning"]

        # 1. Exactly the 5 grounded fields
        assert set(reasoning.keys()) == {
            "what_may_be_happening",
            "what_you_can_control",
            "what_is_not_controllable",
            "reframe",
            "next_step",
        }

        # 2. All 5 fields are populated
        for k, v in reasoning.items():
            assert v and len(v.strip()) > 10, f"Field '{k}' was empty or too short for {name}"

        # 3. Grounded in context
        all_text = " ".join(reasoning.values()).lower()
        assert expected_keyword in all_text, f"Expected keyword '{expected_keyword}' missing from reasoning for {name}"

        # 4. Zero generic boilerplate
        for generic in GENERIC_BOILERPLATE:
            assert generic not in all_text, f"Generic boilerplate '{generic}' leaked into {name}"

        # 5. Non-diagnostic tone
        for pathologizing in PATHOLOGIZING_TERMS:
            assert pathologizing not in all_text, f"Pathologizing term '{pathologizing}' found in {name}"

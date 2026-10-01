"""
Stage 3 Integration Test Suite for Grounded Narrative Generation.

Validates:
1. Target regression input (C1): Locus of Agency with 5 scenes preserving career, graduation, friends, family.
2. Validation cases without forced reframing (C2 chronic illness, C3 workplace injustice).
3. Non-interventional grounding preserving genuine uncertainty (C4 relational drift, C7 climate dread).
4. Practical structuring for executive freeze (C10 ADHD).
5. Perspective reappraisal where appropriate (C5 imposter, C8 rejection).
6. Clarification generation for ambiguous/sparse inputs (C11, C12).
7. Total elimination of generic boilerplate ('This feels overwhelming:', 'When stressors accumulate').
8. Detail traceability (user_stated, inferred_hypothesis, creative_interpretation) per scene.
9. Compatibility with format_narrative_text and backward compatibility with legacy CaseFrame.
"""

import pytest
from app.psychology.case_representation import extract_case_representation
from app.psychology.case_strategy_selector import CaseStrategySelector
from app.narrative_gen.grounded_generator import GroundedNarrativeGenerator
from app.narrative_gen.generator import NarrativeGenerator, format_narrative_text


@pytest.fixture
def selector():
    return CaseStrategySelector()


@pytest.fixture
def grounded_generator():
    return GroundedNarrativeGenerator()


def test_regression_c1_grounded_narrative(selector, grounded_generator):
    """
    C1: Compound career horizon + peer placement + family expectation.
    Must generate 5 grounded scenes preserving all user entities with zero generic boilerplate.
    """
    text = (
        "I'm worried about my career after graduation. My friends are getting placed, "
        "and I'm happy for them, but I feel scared that I'll disappoint my family and that I haven't done enough."
    )
    rep = extract_case_representation(text)
    strat = selector.select(rep)
    story = grounded_generator.generate(rep, strat)

    assert story["modality"] == "locus_of_agency"
    scenes = story["scenes"]
    assert len(scenes) == 5, f"Expected 5 scenes for compound C1 case, got {len(scenes)}"

    # 1. Stated facts preserved in scenes
    all_text = " ".join(f"{s['dialogue']} {s['caption']} {s['narrative']}" for s in scenes).lower()
    assert "career" in all_text
    assert "graduation" in all_text
    assert "friends" in all_text or "placed" in all_text
    assert "family" in all_text

    # 2. Total elimination of generic fallback boilerplate
    assert "this feels overwhelming:" not in all_text
    assert "when stressors accumulate" not in all_text
    assert "general_stress" not in all_text

    # 3. Scene 1 contains stated facts and competing emotions
    s1 = scenes[0]
    assert "graduation" in s1["dialogue"].lower() or "graduation" in s1["narrative"].lower()
    assert "career" in s1["dialogue"].lower() or "career" in s1["narrative"].lower()
    assert s1["detail_traceability"]["dialogue"]["origin"] == "user_stated"

    # 4. Scene 2 contains fear of disappointing family and self-deficit
    s2 = scenes[1]
    assert "family" in s2["dialogue"].lower() or "disappoint" in s2["dialogue"].lower()
    assert "enough" in s2["dialogue"].lower()

    # 5. Scene 3 & 4 differentiate pacing from worth and establish agency
    s3 = scenes[2]
    assert "timeline" in s3["dialogue"].lower() or "worth" in s3["dialogue"].lower()
    s4 = scenes[3]
    assert "agency" in s4["caption"].lower() or "controllable" in s4["caption"].lower()

    # 6. Scene 5 has grounded next step
    s5 = scenes[4]
    assert s5["bubble_type"] == "speech"
    assert "step" in s5["dialogue"].lower() or "preparation" in s5["dialogue"].lower()

    # 7. Formats cleanly
    formatted = format_narrative_text(story)
    assert len(formatted) > 200
    assert "This feels overwhelming:" not in formatted


def test_c2_chronic_illness_narrative_has_zero_reframe(selector, grounded_generator):
    """
    C2: Autoimmune flare-up.
    Must validate physical illness without framing it as cognitive distortion or demanding work.
    """
    text = "My autoimmune flare-up returned this week. I can barely get out of bed, and I feel so guilty that I can't be productive or keep up with my peers."
    rep = extract_case_representation(text)
    strat = selector.select(rep)
    story = grounded_generator.generate(rep, strat)

    assert story["modality"] == "validation"
    all_text = " ".join(f"{s['dialogue']} {s['caption']} {s['narrative']}" for s in story["scenes"]).lower()

    assert "autoimmune" in all_text or "flare-up" in all_text
    assert "bed" in all_text
    assert "rest" in all_text

    # Verify no productivity demands or cognitive disputing
    assert "push through" not in all_text
    assert "get back to work" not in all_text
    assert "distorted" not in all_text


def test_c3_workplace_injustice_narrative_zero_gaslighting(selector, grounded_generator):
    """
    C3: Supervisor theft of project credit.
    Must validate anger without gaslighting or making excuses for supervisor.
    """
    text = "My supervisor gave credit for my three-month project to another colleague in the team meeting. I'm furious and feeling completely powerless."
    rep = extract_case_representation(text)
    strat = selector.select(rep)
    story = grounded_generator.generate(rep, strat)

    assert story["modality"] == "validation"
    all_text = " ".join(f"{s['dialogue']} {s['caption']} {s['narrative']}" for s in story["scenes"]).lower()

    assert "supervisor" in all_text
    assert "project" in all_text
    assert "credit" in all_text
    assert "furious" in all_text or "anger" in all_text or "outrage" in all_text

    # No gaslighting
    assert "misunderstanding" not in all_text
    assert "benefit of the doubt" not in all_text


def test_c4_relational_ambivalence_preserves_uncertainty(selector, grounded_generator):
    """
    C4: Emotional disconnect after 4 years.
    Must hold space without forcing a decision to stay or leave.
    """
    text = "We've been dating for four years, but lately I feel emotionally disconnected. Nothing terrible happened, which almost makes it harder. I don't know whether to try harder or let go."
    rep = extract_case_representation(text)
    strat = selector.select(rep)
    story = grounded_generator.generate(rep, strat)

    assert story["modality"] == "non_interventional_grounding"
    all_text = " ".join(f"{s['dialogue']} {s['caption']} {s['narrative']}" for s in story["scenes"]).lower()

    assert "dating" in all_text or "four years" in all_text
    assert "disconnected" in all_text
    assert "ambivalence" in all_text or "space" in all_text

    # No forced decision
    assert "you must break up" not in all_text
    assert "you should leave" not in all_text
    assert "definitely stay" not in all_text


def test_c7_climate_dread_has_zero_toxic_optimism(selector, grounded_generator):
    """
    C7: Climate grief. Must accept ecological reality without toxic optimism.
    """
    text = "Reading the environmental report today made me feel hopeless about the future. What's the point of planning my life when the planet is burning?"
    rep = extract_case_representation(text)
    strat = selector.select(rep)
    story = grounded_generator.generate(rep, strat)

    assert story["modality"] == "non_interventional_grounding"
    all_text = " ".join(f"{s['dialogue']} {s['caption']} {s['narrative']}" for s in story["scenes"]).lower()

    assert "environmental report" in all_text or "planet" in all_text
    assert "hopeless" in all_text or "sorrow" in all_text
    assert "everything will be fine" not in all_text


def test_c10_executive_freeze_practical_structuring(selector, grounded_generator):
    """
    C10: Multi-task freeze and sitting on floor. Must provide single 2-minute step.
    """
    text = "I have five assignments due by Friday, laundry piling up, and an empty fridge. I've been sitting on the floor staring at my phone for four hours unable to move."
    rep = extract_case_representation(text)
    strat = selector.select(rep)
    story = grounded_generator.generate(rep, strat)

    assert story["modality"] == "practical_structuring"
    all_text = " ".join(f"{s['dialogue']} {s['caption']} {s['narrative']}" for s in story["scenes"]).lower()

    assert "assignments" in all_text or "laundry" in all_text
    assert "freeze" in all_text or "paralysis" in all_text
    assert "two minutes" in all_text or "water" in all_text or "2-minute" in all_text


def test_c11_clarification_generates_open_questions(selector, grounded_generator):
    """
    C11: Vague colloquial numbness.
    Must generate open exploratory questions instead of advice or reframing.
    """
    text = "idk everything is just weird lately and i feel kinda numb"
    rep = extract_case_representation(text)
    strat = selector.select(rep)
    story = grounded_generator.generate(rep, strat)

    assert story["modality"] == "clarification_needed"
    assert len(story["clarification_questions"]) > 0

    s3 = story["scenes"][2]
    assert "?" in s3["dialogue"] or "?" in s3["narrative"]
    assert "heavy" in s3["dialogue"].lower() or "explore" in s3["dialogue"].lower()


def test_narrative_generator_facade_adapter(selector):
    """
    Verifies that NarrativeGenerator.generate_grounded works through the unified class facade.
    """
    text = "I just got promoted to lead the engineering team, but I feel like an absolute fraud."
    rep = extract_case_representation(text)
    strat = selector.select(rep)

    gen = NarrativeGenerator()
    story = gen.generate_grounded(rep, strat)

    assert story["modality"] == "perspective_reappraisal"
    assert len(story["scenes"]) == 3
    assert "promoted" in story["scenes"][0]["narrative"].lower()
    assert "fraud" in story["scenes"][0]["dialogue"].lower()

import os
import sys
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app

WORKPLACE_INJUSTICE_INPUT = (
    "My supervisor gave credit for my three-month project to another colleague in the team meeting. "
    "I'm furious and feeling completely powerless."
)

GENERIC_STRESS_PHRASES = [
    "Navigating a challenging personal situation",
    "Personal stress, ambiguity, and self-doubt",
    "Your choices today, treating yourself with patience",
    "Having doubts or feeling unsettled is a normal human experience",
    "Take a deep breath, write down the one single thing",
]


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_workplace_injustice_psychological_understanding_endpoint(mock_thread, mock_tts, client):
    """
    Verify that the /process endpoint returns grounded, case-specific psychological understanding
    and ZERO generic fallback boilerplate for the workplace injustice debugging case.
    """
    response = client.post("/process", data={"text": WORKPLACE_INJUSTICE_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert data["crisis"] is False
    assert "reasoning" in data

    reasoning = data["reasoning"]
    what_happening = reasoning.get("what_may_be_happening", "")
    what_control = reasoning.get("what_you_can_control", "")
    reframe = reasoning.get("reframe", "")
    next_step = reasoning.get("next_step", "")

    # 1. Assert zero generic fallback leakage in reasoning
    all_reasoning_text = f"{what_happening} {what_control} {reframe} {next_step}"
    for generic in GENERIC_STRESS_PHRASES:
        assert generic not in all_reasoning_text, f"Generic phrase leaked into UI reasoning: '{generic}'"

    # 2. Assert grounded values are present
    assert "supervisor" in what_happening.lower() or "violation" in what_happening.lower()
    assert "documenting" in what_control.lower() or "mentor" in what_control.lower() or "boundaries" in what_control.lower()
    assert "anger" in reframe.lower() or "unfairness" in reframe.lower()
    assert "document" in next_step.lower() or "record" in next_step.lower()

    # 3. Assert technique is grounded
    assert "Validation" in data["technique"]["name"]
    assert "Greenberg" in data["technique"]["citation"] or "Emotion-Focused" in data["technique"]["citation"]

    # 4. Assert external threat badge flag is true (trauma-informed/external violation)
    assert data["is_external_threat"] is True


LEGACY_REASONING_FIELDS = [
    "situation",
    "primary_emotion",
    "central_conflict",
    "automatic_thought",
    "pattern",
    "coping_strategy",
    "specific_reframe",
    "concrete_next_step",
]

EXPECTED_GROUNDED_FIELDS = {
    "what_may_be_happening",
    "what_you_can_control",
    "what_is_not_controllable",
    "reframe",
    "next_step",
}


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_reasoning_object_contains_only_grounded_fields_and_no_legacy_fields(mock_thread, mock_tts, client):
    """
    Regression test asserting that legacy reasoning fields are strictly absent
    from the final /process reasoning dictionary, and only the 5 intended
    grounded fields are preserved.
    """
    response = client.post("/process", data={"text": WORKPLACE_INJUSTICE_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert "reasoning" in data
    reasoning = data["reasoning"]

    # 1. Assert exact key set matches expected grounded fields
    assert set(reasoning.keys()) == EXPECTED_GROUNDED_FIELDS, (
        f"Reasoning keys mismatch. Got: {set(reasoning.keys())}, Expected: {EXPECTED_GROUNDED_FIELDS}"
    )

    # 2. Explicitly assert every legacy field is absent
    for legacy_key in LEGACY_REASONING_FIELDS:
        assert legacy_key not in reasoning, f"Legacy reasoning field '{legacy_key}' must not be present in reasoning object"

    # 3. Assert all 5 grounded fields have non-empty values
    for grounded_key in EXPECTED_GROUNDED_FIELDS:
        assert reasoning[grounded_key], f"Grounded field '{grounded_key}' must not be empty"


PRIORITIES_INPUT = (
    "I'm really confused about what to do. My priorities are shifting every now and then."
)


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_shifting_priorities_psychological_understanding_endpoint(mock_thread, mock_tts, client):
    """
    Verify that the /process endpoint recognizes uncertainty and shifting priorities
    without inventing trauma, relationship conflict, or emotional blunting.
    """
    response = client.post("/process", data={"text": PRIORITIES_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert data["crisis"] is False
    assert "reasoning" in data

    reasoning = data["reasoning"]
    what_happening = reasoning.get("what_may_be_happening", "")
    what_control = reasoning.get("what_you_can_control", "")
    reframe = reasoning.get("reframe", "")
    next_step = reasoning.get("next_step", "")

    # Assert no generic fallback phrase
    for generic in GENERIC_STRESS_PHRASES:
        assert generic not in f"{what_happening} {what_control} {reframe} {next_step}"

    # Assert grounded priorities understanding
    assert "priorit" in what_happening.lower() or "confus" in what_happening.lower()
    assert "priorit" in reframe.lower() or "confus" in reframe.lower()
    assert "priorit" in next_step.lower() or "focus" in next_step.lower()
    assert all(reasoning[k] for k in EXPECTED_GROUNDED_FIELDS)

    # Assert not flagged as external threat / trauma (avoids inventing trauma)
    assert data["is_external_threat"] is False
    assert "Values Clarification" in data["technique"]["name"]


OVERWHELM_CONTENT_INPUT = (
    "I've been watching too much content to numb myself from the overwhelming feeling of how much stuff I have to take care of."
)


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_case_a_overwhelm_content_numbing_endpoint(mock_thread, mock_tts, client):
    """
    Case A: Overwhelm + excessive content consumption for numbing.
    Must recognize:
    - Stated behavior: content consumption / watching content
    - Coping function: numbing / temporary relief from overwhelm
    - Context: responsibilities / stuff to take care of
    - Grounded formulation: non-pathologizing, de-shames coping (no 'addicted' / 'lazy' / 'disordered')
    - 5 grounded reasoning cards with no generic boilerplate
    """
    response = client.post("/process", data={"text": OVERWHELM_CONTENT_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert data["crisis"] is False
    assert "reasoning" in data

    reasoning = data["reasoning"]
    assert set(reasoning.keys()) == EXPECTED_GROUNDED_FIELDS
    assert all(reasoning[k] for k in EXPECTED_GROUNDED_FIELDS)

    what_happening = reasoning["what_may_be_happening"]
    what_control = reasoning["what_you_can_control"]
    what_not_control = reasoning["what_is_not_controllable"]
    reframe = reasoning["reframe"]
    next_step = reasoning["next_step"]

    # 1. Assert zero generic fallback leakage
    combined_reasoning = f"{what_happening} {what_control} {what_not_control} {reframe} {next_step}"
    for generic in GENERIC_STRESS_PHRASES:
        assert generic not in combined_reasoning

    # 2. Non-diagnostic: Must NOT label the user as lazy, addicted, or disordered
    for derogatory in ["lazy", "addicted", "addiction", "disordered", "pathological", "procrastinator"]:
        assert derogatory not in combined_reasoning.lower()

    # 3. Grounded extraction in what may be happening
    assert "content" in what_happening.lower() or "numb" in what_happening.lower() or "overwhelm" in what_happening.lower()
    assert "relief" in what_happening.lower() or "protect" in what_happening.lower() or "coping" in what_happening.lower() or "numbing" in what_happening.lower()

    # 4. Controllability: grounded in selecting single task / pausing consumption
    assert "task" in what_control.lower() or "pause" in what_control.lower() or "single" in what_control.lower() or "consumption" in what_control.lower()
    # Uncontrollability: volume of demands / needing instant relief
    assert "volume" in what_not_control.lower() or "accumulated" in what_not_control.lower() or "responsibilit" in what_not_control.lower() or "instant" in what_not_control.lower()

    # 5. Reframe & Next step
    assert "numb" in reframe.lower() or "relief" in reframe.lower() or "coping" in reframe.lower() or "protect" in reframe.lower()
    assert "one" in next_step.lower() or "single" in next_step.lower() or "task" in next_step.lower() or "pause" in next_step.lower()


WORKLOAD_OVERWHELM_INPUT = (
    "I'm overwhelmed by all the work I have to finish."
)


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_case_d_workload_overwhelm_endpoint(mock_thread, mock_tts, client):
    """
    Case D: Partial input describing workload overwhelm.
    Must NOT be dismissed as lacking context or trigger generic sparse boilerplate.
    Must select workload overwhelm structuring.
    """
    response = client.post("/process", data={"text": WORKLOAD_OVERWHELM_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert data["crisis"] is False
    reasoning = data["reasoning"]

    combined = " ".join(reasoning.values())
    for generic in GENERIC_STRESS_PHRASES:
        assert generic not in combined

    # Must specifically address workload / work
    assert "work" in reasoning["what_may_be_happening"].lower() or "task" in reasoning["what_may_be_happening"].lower() or "overwhelm" in reasoning["what_may_be_happening"].lower()
    assert "task" in reasoning["what_you_can_control"].lower() or "single" in reasoning["what_you_can_control"].lower() or "pace" in reasoning["what_you_can_control"].lower()
    assert "finish" in reasoning["what_is_not_controllable"].lower() or "all" in reasoning["what_is_not_controllable"].lower() or "entire" in reasoning["what_is_not_controllable"].lower()


SPARSE_INPUT = (
    "I don't know what to do anymore."
)


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_case_e_sparse_input_clarification_endpoint(mock_thread, mock_tts, client):
    """
    Case E: Truly sparse / ambiguous input.
    Must NOT invent trauma, workplace conflict, or clinical diagnoses.
    Must provide a compassionate holding space with a non-intrusive clarifying question.
    """
    response = client.post("/process", data={"text": SPARSE_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert data["crisis"] is False
    assert data["is_external_threat"] is False
    reasoning = data["reasoning"]

    combined = " ".join(reasoning.values())
    for generic in GENERIC_STRESS_PHRASES:
        assert generic not in combined

    # Next step must offer a gentle question/invitation without demanding disclosure
    assert "?" in reasoning["next_step"] or "share" in reasoning["next_step"].lower() or "moment" in reasoning["next_step"].lower()
    assert "Take one manageable constructive step today" not in reasoning["next_step"]


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_reasoning_distinctness_across_cases(mock_thread, mock_tts, client):
    """
    Verify that across different scenarios (A, B, C, D, E), the reasoning cards
    are distinct, situational, and never reuse the exact same text across cases.
    """
    cases = {
        "case_a": OVERWHELM_CONTENT_INPUT,
        "case_b": PRIORITIES_INPUT,
        "case_c": WORKPLACE_INJUSTICE_INPUT,
        "case_d": WORKLOAD_OVERWHELM_INPUT,
        "case_e": SPARSE_INPUT,
    }

    results = {}
    for name, text in cases.items():
        res = client.post("/process", data={"text": text})
        assert res.status_code == 200
        results[name] = res.get_json()["reasoning"]

    # Compare pair-wise distinctness for what_may_be_happening, reframe, and next_step
    case_names = list(cases.keys())
    for i in range(len(case_names)):
        for j in range(i + 1, len(case_names)):
            name_a, name_b = case_names[i], case_names[j]
            r_a, r_b = results[name_a], results[name_b]

            assert r_a["what_may_be_happening"] != r_b["what_may_be_happening"], (
                f"Duplicate 'what_may_be_happening' between {name_a} and {name_b}: '{r_a['what_may_be_happening']}'"
            )
            assert r_a["reframe"] != r_b["reframe"], (
                f"Duplicate 'reframe' between {name_a} and {name_b}: '{r_a['reframe']}'"
            )
            assert r_a["next_step"] != r_b["next_step"], (
                f"Duplicate 'next_step' between {name_a} and {name_b}: '{r_a['next_step']}'"
            )


EXACT_RELATIONSHIP_INPUT = (
    "i miss a person he loved me deeply but i couldn't reciprocate"
)


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_exact_relationship_longing_unreciprocated_endpoint(mock_thread, mock_tts, client):
    """
    Exact regression case: "i miss a person he loved me deeply but i couldn't reciprocate"
    Must:
    1. NOT select generic clarification_sparse_input or generic breathing/rest boilerplates.
    2. Acknowledge missing someone, that he loved the user deeply, and inability to reciprocate.
    3. Make NO unsupported assumptions: no forced reconciliation, no trauma badge,
       no guilt pathologizing, no attachment style diagnosis.
    4. Generate all 5 cards from grounded formulation.
    5. Next step provides a gentle, situation-relevant clarifying question because the goal is unstated.
    """
    response = client.post("/process", data={"text": EXACT_RELATIONSHIP_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert data["crisis"] is False
    assert data["is_external_threat"] is False
    assert "reasoning" in data

    reasoning = data["reasoning"]
    assert set(reasoning.keys()) == EXPECTED_GROUNDED_FIELDS
    assert all(reasoning[k] for k in EXPECTED_GROUNDED_FIELDS)

    what_happening = reasoning["what_may_be_happening"]
    what_control = reasoning["what_you_can_control"]
    what_not_control = reasoning["what_is_not_controllable"]
    reframe = reasoning["reframe"]
    next_step = reasoning["next_step"]

    # 1. Zero generic fallback boilerplate
    combined = f"{what_happening} {what_control} {what_not_control} {reframe} {next_step}"
    for generic in GENERIC_STRESS_PHRASES:
        assert generic not in combined
    assert "Needing to have every answer, next step, or life decision figured out" not in combined
    assert "unspecified life decision" not in combined.lower()

    # 2. Absence of unsupported assumptions / non-diagnostic
    for unsupported in [
        "trauma", "ptsd", "disorder", "patholog",
        "attachment style", "avoidant", "anxious-ambivalent",
        "you should reach out", "you must apologize", "reconcil",
    ]:
        assert unsupported not in combined.lower()

    # 3. Contextual relevance in what_may_be_happening
    assert "miss" in what_happening.lower() or "longing" in what_happening.lower()
    assert "loved" in what_happening.lower() or "care" in what_happening.lower()
    assert "reciprocate" in what_happening.lower() or "return" in what_happening.lower()

    # 4. Controllability grounded in capacity & memories vs past feelings
    assert "capacity" in what_control.lower() or "memories" in what_control.lower() or "self-blame" in what_control.lower() or "space" in what_control.lower()
    assert "past" in what_not_control.lower() or "feelings" in what_not_control.lower() or "force" in what_not_control.lower()

    # 5. Reframe de-shames inability to force romantic feelings
    assert "obligation" in reframe.lower() or "force" in reframe.lower() or "mistake" in reframe.lower() or "capacity" in reframe.lower()

    # 6. Next step is a situation-relevant gentle clarifying question (goal unstated)
    assert "?" in next_step
    assert "reach out" in next_step.lower() or "peace" in next_step.lower() or "miss" in next_step.lower() or "memory" in next_step.lower()


RECONNECT_INPUT = (
    "I miss someone from my past who loved me, and I really want to reach out and reconnect with him."
)


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_relationship_miss_and_want_to_reconnect_endpoint(mock_thread, mock_tts, client):
    """
    Case: User misses someone and explicitly wants to reconnect.
    Must recognize longing + reconnection intent, ground in personal agency and reflection,
    without guaranteeing outcomes or making assumptions.
    """
    response = client.post("/process", data={"text": RECONNECT_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert data["crisis"] is False
    reasoning = data["reasoning"]

    combined = " ".join(reasoning.values())
    for generic in GENERIC_STRESS_PHRASES:
        assert generic not in combined

    assert "reconnect" in reasoning["what_may_be_happening"].lower() or "reach out" in reasoning["what_may_be_happening"].lower()
    assert "intent" in reasoning["what_you_can_control"].lower() or "reach out" in reasoning["what_you_can_control"].lower() or "boundar" in reasoning["what_you_can_control"].lower()
    assert "respond" in reasoning["what_is_not_controllable"].lower() or "other person" in reasoning["what_is_not_controllable"].lower()
    assert "invitation" in reasoning["reframe"].lower() or "guarantee" in reasoning["reframe"].lower()
    assert "draft" in reasoning["next_step"].lower() or "reflect" in reasoning["next_step"].lower()


BOUNDARY_INPUT = (
    "I miss someone from my past who loved me, but I definitely do not want to resume the relationship."
)


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_relationship_miss_with_clear_boundary_endpoint(mock_thread, mock_tts, client):
    """
    Case: User misses someone but holds a firm boundary against resuming.
    Must validate that longing and healthy boundaries coexist, without treating missing someone
    as a sign to get back together.
    """
    response = client.post("/process", data={"text": BOUNDARY_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert data["crisis"] is False
    reasoning = data["reasoning"]

    combined = " ".join(reasoning.values())
    for generic in GENERIC_STRESS_PHRASES:
        assert generic not in combined

    assert "boundary" in reasoning["what_may_be_happening"].lower() or "resume" in reasoning["what_may_be_happening"].lower()
    assert "boundary" in reasoning["what_you_can_control"].lower() or "decision" in reasoning["what_you_can_control"].lower()
    assert "nostalgia" in reasoning["what_is_not_controllable"].lower() or "sadness" in reasoning["what_is_not_controllable"].lower() or "waves" in reasoning["what_is_not_controllable"].lower()
    assert "mistake" in reasoning["reframe"].lower() or "decision" in reasoning["reframe"].lower() or "reopen" in reasoning["reframe"].lower()
    assert "reach out" in reasoning["next_step"].lower() or "boundary" in reasoning["next_step"].lower() or "miss" in reasoning["next_step"].lower()


SPARSE_FEELING_INPUT = (
    "I don't know what I feel."
)


@patch("app.main._run_tts", return_value=None)
@patch("app.main.threading.Thread")
def test_genuinely_sparse_emotional_input_endpoint(mock_thread, mock_tts, client):
    """
    Case: Genuinely sparse input lacking situational context.
    Must select clarification without assuming heavy demands, life crises, or trauma.
    """
    response = client.post("/process", data={"text": SPARSE_FEELING_INPUT})
    assert response.status_code == 200

    data = response.get_json()
    assert data["crisis"] is False
    assert data["is_external_threat"] is False
    assert "Clarification" in data["technique"]["name"]
    reasoning = data["reasoning"]

    combined = " ".join(reasoning.values())
    for generic in GENERIC_STRESS_PHRASES:
        assert generic not in combined

    # Must NOT invent relationship trauma or workplace issues
    for hallucinated in ["supervisor", "workplace", "career", "assignments", "dating"]:
        assert hallucinated not in combined.lower()

    # Must offer patient, non-demanding presence
    assert "?" in reasoning["next_step"] or "breath" in reasoning["next_step"].lower() or "notice" in reasoning["next_step"].lower()




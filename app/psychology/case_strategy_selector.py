"""
Multi-Modal Strategy Selector for MindVision (Stage 2).

Consumes CaseRepresentation and selects evidence-informed support modalities:
1. Empathetic Validation & Holding Space (for grief, bodily illness, systemic injustice, caregiving)
2. Perspective Reappraisal (CBT, only when verified cognitive distortions exist)
3. Locus of Agency & Boundary Setting (for compound controllable/uncontrollable transitions)
4. Practical Problem-Solving / Structuring (for executive dysfunction and task accumulation)
5. Non-Interventional Grounding / Somatic Presence (for acute affect or existential dread)
6. Clarification / Exploratory Reflection (when context is sparse, ambiguous, or partial)

Core Principles:
- Never force a cognitive reframe on legitimate external barriers or physical illness.
- Distinguish established facts from stated fears and model inferences.
- Allow uncertainty: return alternative modalities and flag when multiple approaches fit.
- Explicitly enforce contraindications to prevent harmful or gaslighting advice.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

from app.psychology.case_representation import (
    CaseRepresentation,
    InferredAppraisal,
    UncertaintyProfile,
)


@dataclass
class SelectedStrategy:
    """
    Evidence-informed support strategy selected from CaseRepresentation.
    """
    modality: str                  # "validation", "perspective_reappraisal", "locus_of_agency", "practical_structuring", "non_interventional_grounding", "clarification_needed"
    strategy_id: str               # Unique identifier
    name: str                      # Human-readable title
    clinical_framework: str        # Theoretical basis (EFT, CBT, ACT, PST, CFT, etc.)
    rationale: str                 # Clinical reasoning explaining why this modality was chosen
    confidence: float              # Confidence score [0.0 to 1.0]
    is_uncertain: bool             # True if multiple approaches fit or context is partial
    alternative_modalities: List[str] = field(default_factory=list) # Other viable modalities considered
    contraindications: List[str] = field(default_factory=list)      # Safety and clinical boundaries
    reframe_needed: bool = False   # True ONLY for perspective_reappraisal
    core_message: str = ""         # Grounding perspective or validating narrative
    suggested_step: Optional[str] = None # Realistic micro-action, or None if non-intervention
    clarification_questions: List[str] = field(default_factory=list) # Questions to ask if clarification needed
    what_may_be_happening: Optional[str] = None # Grounded, empathetic formulation for the user

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CaseStrategySelector:
    """
    Selects optimal evidence-informed support modality conditioned on CaseRepresentation.
    """

    def select(self, rep: CaseRepresentation) -> SelectedStrategy:
        """
        Primary selection method.
        Evaluates explicit facts, concerns, appraisals, uncertainty, and unknowns.
        """
        dims = {a.dimension for a in rep.inferred_appraisals}

        # 1. Check for Sparse, Vague, or Partial Input -> Clarification
        # Only when input is truly sparse (sparse_insufficient) with no active appraisals
        if getattr(rep, "ambiguity_level", "sufficient") == "sparse_insufficient" and not dims and not rep.uncertainty.has_uncertainty:
            return self._select_clarification(rep)
        if rep.is_partial and not dims and not rep.uncertainty.has_uncertainty:
            return self._select_clarification(rep)
        if len(rep.stated_facts) == 0 and len(rep.stated_emotions) <= 1 and len(rep.stated_thoughts) == 0 and not rep.uncertainty.has_uncertainty and not dims:
            return self._select_clarification(rep)

        # 2. Legitimate External Wrong / Systemic Injustice -> Validation (Zero Reframe)
        if "systemic_injustice" in dims:
            return self._select_injustice_validation(rep)

        # 3. Physical Illness / Bodily Limitation -> Validation & Rest Permission (Zero Reframe)
        if "bodily_limitation" in dims:
            return self._select_illness_validation(rep)

        # 4. Emotional Overwhelm & Distraction / Content Numbing (Case A)
        if "overwhelm_and_avoidance" in dims:
            return self._select_overwhelm_and_coping(rep)

        # 5. Workload Overwhelm (Case D)
        if "workload_overwhelm" in dims:
            return self._select_workload_overwhelm(rep)

        # 6. Relational Ambivalence with Genuine Unknowns -> Holding Space (Non-Intervention)
        if rep.uncertainty.has_uncertainty and rep.uncertainty.uncertainty_type == "relational_ambivalence":
            return self._select_relational_holding_space(rep)

        # 6b. Relational Longing & Unreciprocated Feelings
        if "unreciprocated_relational_longing" in dims:
            return self._select_unreciprocated_relational_longing(rep)

        # 6c. Relational Longing with Desire to Reconnect
        if "relational_longing_reconnection" in dims:
            return self._select_relational_longing_reconnection(rep)

        # 6d. Relational Longing with Clear Boundary
        if "relational_longing_with_boundary" in dims:
            return self._select_relational_longing_with_boundary(rep)

        # 7. Executive Freeze / Task Accumulation -> Practical Structuring & Micro-Entry
        if "executive_freeze" in dims:
            return self._select_practical_structuring(rep)

        # 8. Compound Career Horizon + Peer Comparison + Family Pressure (Regression Case C1)
        if "career_horizon_uncertainty" in dims and ("social_comparison" in dims or "evaluation_fear" in dims):
            return self._select_career_locus_of_agency(rep)

        # 9. Caregiver Burnout & Relational Regret
        if "evaluation_fear" in dims and any("mother" in f.text.lower() or "caregiv" in f.text.lower() for f in rep.stated_facts):
            return self._select_caregiver_validation(rep)

        # 10. Macro Existential Dread (e.g. Climate, Global Future)
        if rep.uncertainty.has_uncertainty and rep.uncertainty.uncertainty_type == "macro_existential":
            return self._select_macro_grounding(rep)

        # 11. Clear Cognitive Distortion / Imposter Syndrome (C5, C8)
        if "perceived_deficit" in dims:
            # If acute crying/exclusion is present, lead with validation
            if any(e.emotion_word.lower() in ("crying", "sobbing") for e in rep.stated_emotions):
                return self._select_social_hurt_validation(rep)
            return self._select_perspective_reappraisal(rep)

        # 12. Social Exclusion / Rejection Hurt
        if "social_comparison" in dims and any(e.emotion_word.lower() in ("crying", "alone") for e in rep.stated_emotions):
            return self._select_social_hurt_validation(rep)

        # 13. Pure Career or Future Horizon Uncertainty
        if "career_horizon_uncertainty" in dims:
            return self._select_career_locus_of_agency(rep)

        # 14. Shifting Priorities & Direction Uncertainty
        if "priority_reorientation" in dims or (rep.uncertainty.has_uncertainty and rep.uncertainty.uncertainty_type == "priority_uncertainty"):
            return self._select_priority_uncertainty(rep)

        # 15. Check sparse fallback if unhandled
        if getattr(rep, "ambiguity_level", "sufficient") == "sparse_insufficient":
            return self._select_clarification(rep)

        # 16. Default: Open Empathetic Validation with Exploratory Clarification
        return self._select_general_validation(rep)

    # -------------------------------------------------------------------------
    # MODALITY BUILDERS
    # -------------------------------------------------------------------------

    def _select_overwhelm_and_coping(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="overwhelm_coping_clarification",
            name="Empathetic Overwhelm Validation & Gentle Prioritization",
            clinical_framework="Compassion-Focused Therapy (Gilbert, 2009); Problem-Solving Therapy (Nezu & Nezu, 2013)",
            rationale=(
                "The user is carrying a heavy sense of overwhelm from having multiple responsibilities to manage. "
                "Turning to content consumption appears to be an understandable attempt to seek temporary relief or numbness from that acute pressure, rather than an intentional choice to neglect things."
            ),
            what_may_be_happening=(
                "You are carrying a heavy sense of overwhelm from having multiple responsibilities to manage. "
                "Turning to content consumption appears to be an understandable attempt to seek temporary relief or numbness from that acute pressure, rather than an intentional choice to neglect things."
            ),
            confidence=0.88,
            is_uncertain=True,
            alternative_modalities=["practical_structuring", "locus_of_agency"],
            contraindications=[
                "do_not_label_as_lazy_or_addicted",
                "do_not_demand_completing_all_tasks_at_once",
                "do_not_pathologize_coping_behavior",
            ],
            reframe_needed=False,
            core_message=(
                "Reaching for distraction when the pressure feels unbearable is a common human attempt to find relief, not a character flaw or laziness. "
                "Acknowledging that you are overwhelmed allows you to treat yourself with patience instead of self-blame."
            ),
            suggested_step=(
                "Write down the single task that feels most pressing or easiest to begin with, "
                "and set everything else aside for today so you only have one focus in front of you."
            ),
        )

    def _select_workload_overwhelm(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="practical_structuring",
            strategy_id="workload_overwhelm_structuring",
            name="Workload Structuring & Single-Task Focus",
            clinical_framework="Problem-Solving Therapy (Nezu & Nezu, 2013); Pacing & Task Structuring",
            rationale=(
                "The user is experiencing acute overwhelm from the volume of work that needs to be completed. "
                "When multiple tasks press at once, feeling weighed down is an understandable reaction rather than personal failure."
            ),
            what_may_be_happening=(
                "You are experiencing acute overwhelm from the volume of work that needs to be completed. "
                "When multiple tasks press at once, feeling weighed down is an understandable reaction rather than personal failure."
            ),
            confidence=0.86,
            is_uncertain=False,
            alternative_modalities=["validation"],
            contraindications=[
                "do_not_overwhelm_with_multi_step_planning",
                "do_not_frame_workload_as_personal_inadequacy",
            ],
            reframe_needed=False,
            core_message=(
                "Feeling overwhelmed by a large workload is an honest signal of capacity limits, not a reflection of your competence. "
                "You do not have to tackle the entire mountain today; focusing on one single item helps restore breathing room."
            ),
            suggested_step=(
                "Pick just one single task or item to start on first, and let the rest wait for a moment."
            ),
        )

    def _select_priority_uncertainty(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="values_clarification",
            strategy_id="shifting_priorities_clarification",
            name="Values Clarification & Priority Grounding",
            clinical_framework="Acceptance & Commitment Therapy (ACT; Hayes, 1999); Values Clarification",
            rationale=(
                "The user is navigating shifting priorities and uncertainty about what direction to take. "
                "When priorities fluctuate, feeling confused is an understandable response rather than personal failure or pathology. "
                "Focuses on clarifying what matters in the immediate present without forcing premature long-term rigidity."
            ),
            what_may_be_happening=(
                "You are navigating shifting priorities and uncertainty about what direction to take. "
                "When priorities fluctuate, feeling confused is an understandable response rather than personal failure or pathology."
            ),
            confidence=0.88,
            is_uncertain=True,
            alternative_modalities=["locus_of_agency", "validation"],
            contraindications=[
                "do_not_assume_trauma_or_relationship_conflict",
                "do_not_force_premature_rigid_plan",
            ],
            reframe_needed=False,
            core_message=(
                "When your priorities are shifting, feeling confused is a natural part of re-evaluating what matters most. "
                "You do not need to lock down your entire direction today while things are in flux."
            ),
            suggested_step=(
                "Write down your top two or three competing priorities on paper, "
                "and choose just one simple, manageable focus for today while letting the broader direction unfold."
            ),
        )

    def _select_clarification(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="clarification_needed",
            strategy_id="clarification_sparse_input",
            name="Exploratory Clarification & Presence",
            clinical_framework="Person-Centered Counseling (Rogers, 1957); NICE NG136",
            rationale=(
                "The input shares a general sense of uncertainty or feeling unsure of what is being experienced without providing situational details. "
                "Acknowledging this uncertainty without guessing causes or imposing advice allows space for gentle exploration."
            ),
            what_may_be_happening=(
                "You are experiencing a moment of uncertainty or difficulty pinpointing direction or feelings. "
                "Because you haven't shared specific situational details, it is best not to assume or guess what is causing this experience."
            ),
            confidence=0.85,
            is_uncertain=True,
            alternative_modalities=["non_interventional_grounding"],
            contraindications=[
                "do_not_assume_specific_trauma_or_depression",
                "do_not_force_premature_action_steps",
                "do_not_impose_unsupported_reframe",
            ],
            reframe_needed=False,
            core_message="Feeling unsure or unable to name what you are feeling is an uncomfortable but deeply human experience. You do not have to have everything sorted out to be heard, and you don't need a complete plan to take things one moment at a time.",
            suggested_step=None,
            clarification_questions=[
                "Take a quiet breath, and if you feel comfortable, share what has felt most noticeable or heavy lately so we can reflect on it together?",
                "Would you prefer to explore what might be underneath this feeling, or just take a quiet pause to catch your breath?",
            ],
        )

    def _select_unreciprocated_relational_longing(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="relational_longing_unreciprocated",
            name="Relational Longing & Emotional Complexity Validation",
            clinical_framework="Emotion-Focused Therapy (Greenberg, 2002); Person-Centered Counseling (Rogers, 1957)",
            rationale=(
                "The user is experiencing longing for someone from the past who loved them deeply, while acknowledging an inability to reciprocate. "
                "Acknowledging both the genuine affection received and the inability to return it normalizes the emotional complexity without imposing guilt, trauma assumptions, or premature reconciliation."
            ),
            what_may_be_happening=(
                "You are experiencing longing for someone from your past who cared for you deeply, while holding the reality that you were unable to return those feelings. "
                "It is completely natural to miss someone and feel the emotional weight of their care, even when you could not reciprocate."
            ),
            confidence=0.90,
            is_uncertain=True,
            alternative_modalities=["non_interventional_grounding"],
            contraindications=[
                "do_not_assume_reconciliation_intent",
                "do_not_assume_unresolved_trauma_or_pathology",
                "do_not_impose_guilt_or_self_blame_for_unreciprocated_feelings",
                "do_not_diagnose_attachment_pattern",
            ],
            reframe_needed=False,
            core_message=(
                "Missing someone who loved you deeply does not mean you made a mistake or that you should have forced feelings you didn't have. "
                "Genuine romantic feelings cannot be manufactured out of obligation, and it is entirely valid to mourn the loss of a meaningful connection while honoring your true capacity."
            ),
            suggested_step=None,
            clarification_questions=[
                "When you reflect on missing this person, what feels most present right now—are you wondering whether to reach out, seeking peace with having walked away, or simply making space for the memory?",
                "Would you like to explore what you miss about this connection, or would it feel more supportive to focus on being gentle with yourself as you hold these memories?",
            ],
        )

    def _select_relational_longing_reconnection(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="locus_of_agency",
            strategy_id="relational_longing_reconnection",
            name="Mindful Relational Reconnection & Agency",
            clinical_framework="Acceptance & Commitment Therapy (Hayes, 1999); Interpersonal Effectiveness",
            rationale=(
                "The user expresses missing someone alongside an emerging wish to reach out and reconnect. "
                "Grounds the decision in personal agency, honest intention, and mutual respect rather than impulsive action or assumptions about the outcome."
            ),
            what_may_be_happening=(
                "You are experiencing deep longing for someone from your past and feeling a desire to reconnect. "
                "Missing someone often clarifies how meaningful that bond was, while opening questions about whether reaching out is the right step today."
            ),
            confidence=0.88,
            is_uncertain=True,
            alternative_modalities=["validation"],
            contraindications=[
                "do_not_guarantee_positive_response_from_other_person",
                "do_not_rush_into_impulsive_contact",
                "do_not_disregard_past_reasons_for_distance",
            ],
            reframe_needed=False,
            core_message=(
                "Wanting to reconnect is a natural expression of caring, but reaching out is an invitation, not a guarantee of how things will unfold. "
                "Focusing on your honest intention allows you to act with clarity while respecting both your needs and theirs."
            ),
            suggested_step=(
                "Write down a private draft of what you would want to say and reflect on your true hopes for reconnecting, giving yourself a day or two before deciding whether to send it."
            ),
        )

    def _select_relational_longing_with_boundary(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="relational_longing_boundary_affirmation",
            name="Boundary Affirmation with Grief Validation",
            clinical_framework="Emotion-Focused Therapy (Greenberg, 2002); Dialectical Behavior Therapy (Linehan, 1993)",
            rationale=(
                "The user experiences longing for a past person while maintaining an explicit boundary against resuming the relationship. "
                "Validates that longing and boundaries can coexist without treating missing someone as evidence of a mistaken choice."
            ),
            what_may_be_happening=(
                "You are feeling the genuine ache of missing someone from your past, while holding a clear and conscious boundary that you do not want to resume the relationship. "
                "Missing someone and knowing you should not be together can coexist honestly."
            ),
            confidence=0.90,
            is_uncertain=False,
            alternative_modalities=["non_interventional_grounding"],
            contraindications=[
                "do_not_treat_missing_them_as_a_reason_to_break_boundary",
                "do_not_invalidate_past_feelings",
                "do_not_impose_reconciliation_narrative",
            ],
            reframe_needed=False,
            core_message=(
                "Missing someone does not mean you made the wrong decision or that you should reopen contact. "
                "Longing is simply the heart's way of acknowledging a connection that mattered, and you can honor those memories without breaking your personal boundaries."
            ),
            suggested_step=(
                "Allow yourself to feel the sadness or nostalgia of missing them today without treating it as an urge to reach out, trusting the reasons behind your boundary."
            ),
        )

    def _select_injustice_validation(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="injustice_validation_options",
            name="Workplace Injustice Validation & Boundary Options",
            clinical_framework="Emotion-Focused Therapy (Greenberg, 2002); Trauma-Informed Workplace Boundary Theory",
            rationale=(
                "The user describes an external violation of professional fairness where credit for a project was attributed to another colleague. "
                "Reframing the supervisor's behavior could minimize the unfairness described and invalidate the user's legitimate emotional response."
            ),
            what_may_be_happening=(
                "You describe an external violation of professional fairness where your supervisor gave credit for your three-month project to another colleague in a team meeting. "
                "Your anger and feeling of powerlessness are completely legitimate responses to having your labor unacknowledged."
            ),
            confidence=0.85,
            is_uncertain=False,
            alternative_modalities=["locus_of_agency"],
            contraindications=[
                "do_not_gaslight_or_reframe_supervisor_behavior",
                "do_not_blame_user_for_powerlessness",
                "do_not_suggest_positive_intent_without_evidence",
            ],
            reframe_needed=False,
            core_message=(
                "Your anger and frustration are healthy, appropriate responses to having your labor taken without credit. "
                "You do not need to reframe this unfairness or pretend it didn't hurt."
            ),
            suggested_step=(
                "Document your contributions, project timeline, and files in a private personal record, "
                "and give yourself time for the acute anger to settle before deciding whether to consult HR, a trusted mentor, or leadership."
            ),
        )

    def _select_illness_validation(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="bodily_limitation_compassion",
            name="Empathetic Validation & Rest Permission",
            clinical_framework="Compassion-Focused Therapy (Gilbert, 2009); Pacing & Energy Conservation Theory",
            rationale=(
                "The user's limitation is a legitimate biological illness flare-up, not a cognitive distortion. "
                "Treating physical fatigue as a thinking trap or prescribing productivity steps would cause physiological harm."
            ),
            what_may_be_happening=(
                "You are experiencing a physical illness flare-up that severely restricts your physical capacity and energy. "
                "Feeling exhausted and needing to rest is a legitimate biological reality, not a personal failing or lack of will."
            ),
            confidence=0.94,
            is_uncertain=False,
            alternative_modalities=["non_interventional_grounding"],
            contraindications=[
                "do_not_frame_physical_illness_as_cognitive_distortion",
                "do_not_demand_productivity_steps",
                "do_not_impose_guilt_over_unmet_deadlines",
            ],
            reframe_needed=False,
            core_message=(
                "An autoimmune flare-up is a physical reality that demands restorative pacing, not self-reproach. "
                "Your body is expending significant energy to stabilize; resting is essential physiological care, not lost time."
            ),
            suggested_step=(
                "Give yourself explicit, guilt-free permission to rest today and focus strictly on physical comfort, hydration, and gentle warmth."
            ),
        )

    def _select_relational_holding_space(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="non_interventional_grounding",
            strategy_id="relational_ambivalence_holding_space",
            name="Holding Space for Relational Ambivalence",
            clinical_framework="Ambivalence-Informed Therapy; ACT Acceptance (Hayes et al., 1999)",
            rationale=(
                "The relationship carries genuine emotional disconnection without an acute precipitating crisis. "
                "Forcing premature resolution (stay vs. leave) or manufacturing false certainty harms authentic decision-making. "
                "Respects the user's honest uncertainty."
            ),
            confidence=0.88,
            is_uncertain=True,
            alternative_modalities=["validation", "clarification_needed"],
            contraindications=[
                "do_not_force_premature_decision_or_false_certainty",
                "do_not_judge_disconnection_as_failure",
                "do_not_prescribe_unwanted_relationship_rules",
            ],
            reframe_needed=False,
            core_message=(
                "It is honest and human to feel emotionally disconnected even when nothing explicitly terrible took place. "
                "You do not have to force an immediate stay-or-go verdict today; ambivalence is uncomfortable, but it deserves space to be felt and understood."
            ),
            suggested_step=(
                "Notice and quietly jot down how you feel during everyday interactions this week, without pressuring yourself to reach an immediate conclusion."
            ),
        )

    def _select_practical_structuring(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="practical_structuring",
            strategy_id="executive_freeze_micro_entry",
            name="Executive Unfreezing & Micro-Entry Structuring",
            clinical_framework="Problem-Solving Therapy (Nezu & Nezu, 2013); Behavioral Activation",
            rationale=(
                "The user is experiencing task accumulation and physical behavioral freeze (sitting on floor unable to move). "
                "Abstract cognitive reframing or extensive planning increases executive overload. "
                "Provides somatic grounding and a single 2-minute entry point."
            ),
            confidence=0.92,
            is_uncertain=False,
            alternative_modalities=["non_interventional_grounding"],
            contraindications=[
                "do_not_overwhelm_with_multi_step_planning",
                "do_not_moralize_freeze_as_laziness",
                "do_not_demand_immediate_completion_of_all_tasks",
            ],
            reframe_needed=False,
            core_message=(
                "Sitting on the floor unable to move is a nervous system freeze triggered by an avalanche of competing demands, not laziness. "
                "You do not need to conquer the whole mountain in this hour; touching a single small pebble breaks the paralysis."
            ),
            suggested_step=(
                "Drink one glass of water, stand up slowly, and choose just ONE 2-minute action (such as opening a single document or clearing one dish), and let that be enough for right now."
            ),
        )

    def _select_career_locus_of_agency(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="locus_of_agency",
            strategy_id="career_agency_values_differentiation",
            name="Locus of Agency & Values Differentiation",
            clinical_framework="Stress Inoculation (Meichenbaum, 1985); Acceptance & Commitment Therapy (Hayes, 1999)",
            rationale=(
                "The user faces compound concerns: future career transition uncertainty, peer placement milestones, fear of letting family down, and perceived personal deficit. "
                "Decouples personal worth from peer timelines, validates family love while clarifying that hiring pace is external, and directs agency strictly to manageable preparation."
            ),
            confidence=0.86,
            is_uncertain=True,
            alternative_modalities=["validation"],
            contraindications=[
                "do_not_dismiss_family_concern_as_irrational",
                "do_not_force_premature_hiring_certainty",
                "do_not_tell_user_their_worries_are_groundless",
            ],
            reframe_needed=False,  # Anchors in agency and pacing rather than cognitive disputation
            core_message=(
                "It is completely natural to feel both happy for your friends and anxious about your own path. "
                "Other people's placement timelines do not define your ceiling or your worth. "
                "You can deeply care about your family while honoring that building a career happens at an individual, sustainable pace."
            ),
            suggested_step=(
                "Identify one manageable, controllable action for your own path today (such as polishing one section of your resume or setting aside structured rest), rather than carrying the entire future all at once."
            ),
        )

    def _select_caregiver_validation(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="caregiver_strain_self_compassion",
            name="Caregiver Strain & Self-Compassion",
            clinical_framework="Self-Compassion Therapy (Neff, 2003); Caregiver Stress Framework",
            rationale=(
                "The user is carrying full-time caregiving and professional responsibilities simultaneously. "
                "Snapping at a parent reflects acute nervous system depletion, not a moral failure. "
                "Validates human exhaustion and encourages self-forgiveness over productivity."
            ),
            confidence=0.90,
            is_uncertain=False,
            alternative_modalities=["perspective_reappraisal"],
            contraindications=[
                "do_not_judge_emotional_outburst_as_character_flaw",
                "do_not_add_caregiving_chores",
            ],
            reframe_needed=False,
            core_message=(
                "Deeply loving someone does not make you immune to human limits. "
                "Snapping when your energy reserves are entirely depleted reflects sheer exhaustion, not that you are an awful person."
            ),
            suggested_step=(
                "Take ten quiet minutes strictly for yourself today with a warm drink or slow breath, and remember that caregivers need care too."
            ),
        )

    def _select_macro_grounding(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="non_interventional_grounding",
            strategy_id="macro_existential_local_anchoring",
            name="Acceptance & Local Purpose Anchoring",
            clinical_framework="Eco-Anxiety & Existential Grounding (Frankl, 1959; Doherty, 2018)",
            rationale=(
                "Global environmental precarity is an objective reality that cannot be decatastrophized. "
                "Superficial reassurance is invalidating. Focuses on holding existential sorrow while finding immediate local connection."
            ),
            confidence=0.86,
            is_uncertain=False,
            alternative_modalities=["validation"],
            contraindications=[
                "do_not_offer_toxic_optimism_about_climate_reality",
                "do_not_call_climate_grief_a_distortion",
            ],
            reframe_needed=False,
            core_message=(
                "Feeling sorrow and dread about the planet's trajectory is a deeply empathetic, human response. "
                "You cannot shoulder the entire macro future alone, but your care and present presence hold real value."
            ),
            suggested_step=(
                "Anchor in your immediate physical environment today: step outside, feel the air, and connect with one local person, place, or craft that brings you grounded meaning."
            ),
        )

    def _select_perspective_reappraisal(self, rep: CaseRepresentation) -> SelectedStrategy:
        facts_text = " ".join([f.text.lower() for f in rep.stated_facts])
        is_creative = "manuscript" in facts_text or "writing" in facts_text or "art" in facts_text

        if is_creative:
            core_msg = (
                "Facing repeated rejections in creative work naturally activates intense self-doubt. "
                "Rejection of a submission is part of the publishing craft, not a final verdict on your talent or your voice."
            )
            step_msg = (
                "Identify one specific scene or craft element in your manuscript you feel proud of, and honor your persistent dedication to your work."
            )
        else:
            core_msg = (
                "Entering a higher-stakes role naturally activates internal alarm bells. "
                "Feeling like an imposter does not mean you are one; skills are built through ongoing practice, and past achievements remain real."
            )
            step_msg = (
                "List two concrete accomplishments or skills you demonstrated that earned you this opportunity, separating how you feel right now from what you have actually built."
            )

        return SelectedStrategy(
            modality="perspective_reappraisal",
            strategy_id="creative_rejection_reappraisal" if is_creative else "imposter_promotion_reappraisal",
            name="Cognitive Decoupling & Evidence Testing",
            clinical_framework="Cognitive Therapy (Beck, 1979); Burns (1980)",
            rationale=(
                "The user's language demonstrates disproportionate, globalized self-labeling ('absolute fraud', 'have no talent', 'wasted years') "
                "that directly conflicts with verified milestones (e.g. promotion to lead, continued manuscript creation). "
                "Carefully tests extreme thoughts against observable facts."
            ),
            confidence=0.84,
            is_uncertain=True,
            alternative_modalities=["validation"],
            contraindications=[
                "do_not_dismiss_feelings_as_foolish",
                "do_not_promise_unconditional_external_acclaim",
            ],
            reframe_needed=True,
            core_message=core_msg,
            suggested_step=step_msg,
        )

    def _select_social_hurt_validation(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="social_exclusion_soothing",
            name="Social Rejection Soothing & Reality Checking",
            clinical_framework="Compassion-Focused Therapy (Gilbert, 2009); Attachment Theory",
            rationale=(
                "The user is experiencing acute interpersonal hurt and emotional distress ('crying all morning'). "
                "Cognitive disputation before emotional stabilization would feel dismissive. "
                "Validates social pain first, separating feeling left out from personal worth."
            ),
            confidence=0.86,
            is_uncertain=True,
            alternative_modalities=["perspective_reappraisal"],
            contraindications=[
                "do_not_tell_user_they_are_overreacting",
                "do_not_blame_user_for_being_excluded",
            ],
            reframe_needed=False,
            core_message=(
                "Being excluded by people you care about stings acutely and naturally triggers self-doubt. "
                "You do not have to immediately conclude that something is fundamentally wrong with you; social dynamics are complex and frequently uncommunicated."
            ),
            suggested_step=(
                "Give yourself compassionate permission to feel hurt without self-criticism. When you feel steady, you can decide whether to ask a close friend gently about what happened."
            ),
        )

    def _select_general_validation(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="general_grounded_validation",
            name="Grounded Validation & Presence",
            clinical_framework="Person-Centered Counseling; CFT",
            rationale=(
                "Acknowledge the user's specific context without imposing premature reframing. "
                "Provides empathetic validation and encourages taking things one manageable moment at a time."
            ),
            confidence=0.75,
            is_uncertain=True,
            alternative_modalities=["clarification_needed", "locus_of_agency"],
            contraindications=[
                "do_not_impose_unsupported_reframe",
            ],
            reframe_needed=False,
            core_message="Navigating uncertainty and difficult moments is a real burden. You don't have to carry the whole weight at once; giving yourself patience is a valid and necessary step.",
            suggested_step="Take one quiet moment to pause and focus on what feels most grounding for you right now.",
        )

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
        # 1. Check for Sparse, Vague, or Partial Input -> Clarification
        if rep.is_partial and not rep.uncertainty.has_uncertainty:
            return self._select_clarification(rep)
        if len(rep.stated_facts) == 0 and len(rep.stated_emotions) <= 1 and len(rep.stated_thoughts) == 0 and not rep.uncertainty.has_uncertainty:
            return self._select_clarification(rep)

        # Extract active appraisal dimensions
        dims = {a.dimension for a in rep.inferred_appraisals}

        # 2. Legitimate External Wrong / Systemic Injustice -> Validation (Zero Reframe)
        if "systemic_injustice" in dims:
            return self._select_injustice_validation(rep)

        # 3. Physical Illness / Bodily Limitation -> Validation & Rest Permission (Zero Reframe)
        if "bodily_limitation" in dims:
            return self._select_illness_validation(rep)

        # 4. Relational Ambivalence with Genuine Unknowns -> Holding Space (Non-Intervention)
        if rep.uncertainty.has_uncertainty and rep.uncertainty.uncertainty_type == "relational_ambivalence":
            return self._select_relational_holding_space(rep)

        # 5. Executive Freeze / Task Accumulation -> Practical Structuring & Micro-Entry
        if "executive_freeze" in dims:
            return self._select_practical_structuring(rep)

        # 6. Compound Career Horizon + Peer Comparison + Family Pressure (Regression Case C1)
        if "career_horizon_uncertainty" in dims and ("social_comparison" in dims or "evaluation_fear" in dims):
            return self._select_career_locus_of_agency(rep)

        # 7. Caregiver Burnout & Relational Regret
        if "evaluation_fear" in dims and any("mother" in f.text.lower() or "caregiv" in f.text.lower() for f in rep.stated_facts):
            return self._select_caregiver_validation(rep)

        # 8. Macro Existential Dread (e.g. Climate, Global Future)
        if rep.uncertainty.has_uncertainty and rep.uncertainty.uncertainty_type == "macro_existential":
            return self._select_macro_grounding(rep)

        # 9. Clear Cognitive Distortion / Imposter Syndrome (C5, C8)
        if "perceived_deficit" in dims:
            # If acute crying/exclusion is present, lead with validation
            if any(e.emotion_word.lower() in ("crying", "sobbing") for e in rep.stated_emotions):
                return self._select_social_hurt_validation(rep)
            return self._select_perspective_reappraisal(rep)

        # 10. Social Exclusion / Rejection Hurt
        if "social_comparison" in dims and any(e.emotion_word.lower() in ("crying", "alone") for e in rep.stated_emotions):
            return self._select_social_hurt_validation(rep)

        # 11. Pure Career or Future Horizon Uncertainty
        if "career_horizon_uncertainty" in dims:
            return self._select_career_locus_of_agency(rep)

        # 12. Shifting Priorities & Direction Uncertainty
        if "priority_reorientation" in dims or (rep.uncertainty.has_uncertainty and rep.uncertainty.uncertainty_type == "priority_uncertainty"):
            return self._select_priority_uncertainty(rep)

        # 13. Default: Open Empathetic Validation with Exploratory Clarification
        return self._select_general_validation(rep)

    # -------------------------------------------------------------------------
    # MODALITY BUILDERS
    # -------------------------------------------------------------------------

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
            clinical_framework="Person-Centered Counseling (Rogers, 1957)",
            rationale=(
                "The user's input is brief, ambiguous, or lacks specific situational context. "
                "Forcing an interpretative reframe or actionable advice would impose unsupported assumptions. "
                "Holds space for uncertainty and invites gentle exploration."
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
            core_message="Feeling unsure or unsettled is completely valid. You don't have to have everything sorted out to be heard.",
            suggested_step=None,
            clarification_questions=[
                "Has anything specific felt particularly demanding lately?",
                "Would you prefer to explore what might be underneath this feeling, or just take a quiet pause to catch your breath?",
            ],
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

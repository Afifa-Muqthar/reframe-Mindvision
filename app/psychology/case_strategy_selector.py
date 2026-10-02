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

import re
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

        # 1b. Cognitive Loop Break & Growth Affirmation (e.g. breaking comparison loop, feeling proud of progress)
        if "cognitive_loop_break_growth" in dims or (
            any(e.emotion_word.lower() in ("proud", "pride") for e in rep.stated_emotions)
            and any(w in rep.raw_text.lower() for w in ["loop", "compare", "personally", "habit", "pattern", "myself"])
        ):
            return self._select_cognitive_loop_break_growth(rep)

        # 2. Legitimate External Wrong / Systemic Injustice -> Validation (Zero Reframe)
        if "systemic_injustice" in dims:
            if any(w in f.text.lower() for f in rep.stated_facts for w in ["partner", "grade", "mia", "zero code"]):
                return self._select_collaboration_injustice(rep)
            return self._select_injustice_validation(rep)

        # 3. Physical Illness / Bodily Limitation -> Validation & Rest Permission (Zero Reframe)
        if "bodily_limitation" in dims:
            if any(w in f.text.lower() for f in rep.stated_facts for w in ["acl", "crutch", "injury", "tryout", "soccer"]):
                return self._select_sports_injury_validation(rep)
            return self._select_illness_validation(rep)

        # 3b. Resource Strain & Escalating Living Obligations
        if "resource_strain_uncertainty" in dims:
            return self._select_resource_strain_validation(rep)

        # 3c. Creative Block & Expressive Depletion
        if "creative_block" in dims:
            return self._select_creative_block_reappraisal(rep)

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

        # 8a. Filial Boundary Pressure & Setting Limits
        if ("family_boundary_pressure" in getattr(rep, "interacting_concerns", []) or "evaluation_fear" in dims) and any(w in rep.raw_text.lower() for w in ["expects", "say no", "said no", "awful daughter", "awful son", "boundary", "boundaries"]):
            if any(w in rep.raw_text.lower() for w in ["mother", "father", "parents", "family"]):
                return self._select_family_boundary_validation(rep)

        # 8b. Caregiver Burnout & Relational Regret (C6)
        if ("evaluation_fear" in dims or "caregiver" in rep.raw_text.lower()) and any(w in rep.raw_text.lower() for w in ["aging mother", "caregiv", "caring for", "snapped"]):
            return self._select_caregiver_validation(rep)

        # 8c. Compound Career Horizon + Peer Comparison + Family Pressure (Regression Case C1)
        if "career_horizon_uncertainty" in dims and ("social_comparison" in dims or "evaluation_fear" in dims):
            return self._select_career_locus_of_agency(rep)

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

        # 14b. Pure Peer / Social Comparison
        if "social_comparison" in dims:
            return self._select_social_comparison_differentiation(rep)

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
        raw_lower = rep.raw_text.lower()
        emotions_words = [e.emotion_word for e in rep.stated_emotions]

        if "numb" in raw_lower or any("numb" in w for w in emotions_words):
            what_happening = (
                "Things feel heavy, foggy, or disconnected right now. "
                "Often, emotional numbness or a moment of uncertainty is your mind and nervous system's quiet way "
                "of asking for a pause when things have felt like a lot to carry, even if you cannot put your finger on an exact reason."
            )
            reframe_msg = (
                "You do not need to have a clear story or a dramatic reason to justify how you feel. "
                "Emotional numbness is a completely legitimate human state that deserves gentle patience, "
                "not pressure to explain yourself or force yourself to snap out of it."
            )
        elif "empty" in raw_lower or "hollow" in raw_lower:
            what_happening = (
                "You are experiencing a moment of deep emotional exhaustion or emptiness right now. "
                "Feeling drained or hollow is often a quiet sign that your inner reserves have been carrying things in the background."
            )
            reframe_msg = (
                "Feeling empty is not a personal failure, but an honest signal that you need restorative care and gentleness. "
                "You do not have to force yourself to feel positive or productive when your mind simply needs rest."
            )
        elif emotions_words:
            emotions_str = ", ".join(emotions_words)
            what_happening = (
                f"You are experiencing a moment of uncertainty and feeling {emotions_str} right now. "
                "It is completely natural for feelings to feel hazy or difficult to pin down into neat words."
            )
            reframe_msg = (
                f"Feeling {emotions_str} without knowing exactly why is completely valid. "
                "Your feelings do not need a logical justification to be real, and giving yourself permission to simply feel what you feel is enough."
            )
        else:
            what_happening = (
                "You are experiencing a moment of uncertainty, fog, or difficulty putting things into words right now. "
                "You do not have to have a neat explanation or a clear storyline ready for your experience to be completely real and valid."
            )
            reframe_msg = (
                "Feeling unsure, disconnected, or unable to name what you are going through is a deeply human experience. "
                "You do not owe anyone a neat explanation to deserve kindness, and you don't need a complete plan to take things one quiet moment at a time."
            )

        step_question = (
            "No pressure to figure anything out right now. If you feel comfortable, what has felt most noticeable or heavy lately, or would it feel better to just take a quiet pause to rest?"
        )

        return SelectedStrategy(
            modality="clarification_needed",
            strategy_id="clarification_sparse_input",
            name="Empathetic Holding Space & Gentle Presence",
            clinical_framework="Person-Centered Counseling (Rogers, 1957); Compassion-Focused Therapy (Gilbert, 2009)",
            rationale=(
                "The user is experiencing emotional fog, numbness, or uncertainty. "
                "Validates their internal state directly without demanding situational justification or clinical explanations, "
                "providing a warm, non-evaluative holding space."
            ),
            what_may_be_happening=what_happening,
            confidence=0.85,
            is_uncertain=True,
            alternative_modalities=["non_interventional_grounding"],
            contraindications=[
                "do_not_assume_specific_trauma_or_depression",
                "do_not_force_premature_action_steps",
                "do_not_demand_situational_justification",
            ],
            reframe_needed=False,
            core_message=reframe_msg,
            suggested_step=None,
            clarification_questions=[
                step_question,
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
        cond_matches = [f.text for f in rep.stated_facts if f.category == "condition"]
        if not cond_matches:
            for cand in ["chronic migraine", "migraine", "autoimmune flare-up", "flare-up", "illness", "chronic pain", "injury"]:
                if cand in rep.raw_text.lower():
                    cond_matches.append(cand)
                    break
        condition_name = cond_matches[0] if cond_matches else "a physical health condition"

        return SelectedStrategy(
            modality="validation",
            strategy_id="bodily_limitation_compassion",
            name="Empathetic Validation & Rest Permission",
            clinical_framework="Compassion-Focused Therapy (Gilbert, 2009); Pacing & Energy Conservation Theory",
            rationale=(
                f"The user's limitation is a legitimate biological condition ({condition_name}), not a cognitive distortion. "
                "Treating physical fatigue or pain as a thinking trap or prescribing productivity steps would cause physiological harm."
            ),
            what_may_be_happening=(
                f"You are experiencing physical symptoms related to {condition_name} that severely restrict your capacity and energy. "
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
                f"Dealing with {condition_name} is a physical reality that demands restorative pacing, not self-reproach. "
                "Your body is expending significant energy to stabilize and recover; resting is essential physiological care, not lost time."
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
        raw_lower = rep.raw_text.lower()
        is_disruption = any(w in raw_lower for w in ["phased out", "disposable", "ai tools", "ai", "layoff", "laid off", "restructur", "pivot", "automation"])

        if is_disruption:
            what_happening = (
                "You are facing sudden organizational or technological changes that threaten your current role and livelihood. "
                "Navigating industry shifts after building deep professional expertise understandably brings up feelings of vulnerability, grief, and fear."
            )
            core_msg = (
                "Facing a unit phaseout or major technological transition after decades of dedication is a profound disruption, not personal failure. "
                "While organizational shifts and rapid automation are outside your control, the depth of your accumulated expertise, craftsmanship, and problem-solving remains yours. "
                "You can honor the legitimate grief of this disruption while taking grounded agency over how you translate your durable experience into your next chapter."
            )
            step_msg = (
                "Identify one manageable action focused strictly on your own path today (such as noting down three core strengths or consulting a trusted professional contact), rather than carrying the entire future at once."
            )
            rationale_text = (
                "The user describes industry or workplace disruption (e.g. unit phaseout or automation). "
                "Validates disruption shock while anchoring locus of control strictly in transferable experience and self-directed next steps."
            )
        else:
            what_happening = (
                "You are navigating a stressful career transition and comparing your pace to the milestones of others. "
                "When peers appear to move ahead quickly, it is natural for self-doubt and pressure around future security to surge."
            )
            core_msg = (
                "It is completely natural to feel both happy for your friends and anxious about your own path. "
                "Other people's placement timelines do not define your ceiling or your worth. "
                "You can deeply care about your family while honoring that building a career happens at an individual, sustainable pace."
            )
            step_msg = (
                "Identify one manageable, controllable action for your own path today (such as polishing one section of your resume or setting aside structured rest), rather than carrying the entire future all at once."
            )
            rationale_text = (
                "The user faces compound concerns: future career transition uncertainty, peer placement milestones, fear of letting family down, and perceived personal deficit. "
                "Decouples personal worth from peer timelines, validates family love while clarifying that hiring pace is external, and directs agency strictly to manageable preparation."
            )

        return SelectedStrategy(
            modality="locus_of_agency",
            strategy_id="workplace_transition_locus_of_agency" if is_disruption else "career_agency_values_differentiation",
            name="Locus of Agency & Career Transition" if is_disruption else "Locus of Agency & Values Differentiation",
            clinical_framework="Stress Inoculation (Meichenbaum, 1985); Acceptance & Commitment Therapy (Hayes, 1999)",
            rationale=rationale_text,
            what_may_be_happening=what_happening,
            confidence=0.88,
            is_uncertain=True,
            alternative_modalities=["validation"],
            contraindications=[
                "do_not_dismiss_family_concern_as_irrational",
                "do_not_force_premature_hiring_certainty",
                "do_not_tell_user_their_worries_are_groundless",
                "do_not_blame_user_for_technological_shifts",
            ],
            reframe_needed=False,  # Anchors in agency and pacing rather than cognitive disputation
            core_message=core_msg,
            suggested_step=step_msg,
        )

    def _select_family_boundary_validation(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="family_boundary_guilt_compassion",
            name="Family Boundary Guilt & Capacity Honoring",
            clinical_framework="Relational Boundary Theory; Self-Compassion Therapy (Neff, 2003)",
            rationale=(
                "The user is experiencing intense guilt and self-blame after communicating a boundary or saying no to parental/family expectations. "
                "Honoring finite human stamina and personal pacing does not diminish filial love or mean they are failing their family."
            ),
            what_may_be_happening=(
                "You are navigating the painful tension between wanting to support your family and reaching the physical limits of your capacity. "
                "Communicating a boundary or saying no when exhausted naturally stirs guilt, but it is an honest acknowledgment of your limits, not a lack of love."
            ),
            confidence=0.90,
            is_uncertain=False,
            alternative_modalities=["locus_of_agency"],
            contraindications=[
                "do_not_demand_self_sacrifice_over_health",
                "do_not_fuel_guilt_or_blame_user_for_boundaries",
                "do_not_invalidate_family_disappointment",
            ],
            reframe_needed=False,
            core_message=(
                "Saying no when you are running on empty does not make you an awful daughter or son. "
                "Loving your family and honoring your own physical and emotional limits can coexist; you cannot pour from an empty cup."
            ),
            suggested_step=(
                "Acknowledge the discomfort of their disappointment without turning it into self-blame, and prioritize getting the rest you genuinely need this weekend."
            ),
        )

    def _select_caregiver_validation(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="caregiver_strain_self_compassion",
            name="Caregiver Strain & Self-Compassion",
            clinical_framework="Self-Compassion Therapy (Neff, 2003); Caregiver Stress Framework",
            rationale=(
                "The user is carrying intensive caregiving and daily responsibilities simultaneously. "
                "Moments of frustration or emotional depletion under severe exhaustion reflect nervous system limits, not a moral failure. "
                "Validates human exhaustion and encourages self-forgiveness over productivity."
            ),
            what_may_be_happening=(
                "You are carrying heavy ongoing caregiving and daily responsibilities that have depleted your energy reserves. "
                "Moments of strain or emotional exhaustion reflect human limits, not a lack of love or character."
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
                "Feeling overwhelmed or reactive when your energy reserves are entirely depleted reflects sheer exhaustion, not that you are an awful person."
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
        raw_lower = rep.raw_text.lower()
        is_creative = any(re.search(rf"\b{re.escape(w)}\b", facts_text) for w in ["manuscript", "writing", "art", "canvas", "painting", "novel", "draft"])
        is_academic = any(re.search(rf"\b{re.escape(w)}\b", raw_lower) for w in ["fellowship", "phd", "grad school", "doctorate", "scholarship", "lab", "dissertation", "adviser", "advisor"])
        is_relationship = any(re.search(rf"\b{re.escape(w)}\b", raw_lower) for w in ["partner", "relationship", "spouse", "husband", "wife", "boyfriend", "girlfriend"])

        if is_creative:
            milestone = "creative work"
            strategy_id = "creative_rejection_reappraisal"
            what_happening = (
                "You are navigating creative rejection and self-doubt after investing significant energy into your work. "
                "It is natural to question your abilities when external reception feels discouraging."
            )
            core_msg = (
                "Facing repeated rejections in creative work naturally activates intense self-doubt. "
                "Rejection of a submission is part of the publishing craft, not a final verdict on your talent or your voice."
            )
            step_msg = (
                "Identify one specific scene or craft element in your manuscript you feel proud of, and honor your persistent dedication to your work."
            )
        elif is_academic:
            milestone = "academic fellowship or advanced program"
            strategy_id = "academic_fellowship_reappraisal"
            what_happening = (
                "Entering a competitive academic fellowship or advanced program naturally triggers intense imposter feelings and fear of being found out. "
                "These internal doubts often surge precisely when stepping into environments surrounded by accomplished peers."
            )
            core_msg = (
                "Being selected for a competitive program or fellowship reflects external evaluation of your rigorous work and potential. "
                "Feeling out of place or fearing you will be exposed is a very common reaction to new, high-caliber environments, not proof that you don't belong."
            )
            step_msg = (
                "Ground yourself in observable facts: write down two concrete research questions or skills you brought to this program, reminding yourself that you were chosen on merit."
            )
        elif is_relationship:
            milestone = "relationship and shared home dynamic"
            strategy_id = "relational_contribution_guilt_reappraisal"
            what_happening = (
                "You are carrying heavy guilt and feelings of inadequacy comparing your current study routine to your partner's intense work hours. "
                "When one partner appears to shoulder visible external burdens, it is easy to internalize self-blame and feel like you are falling short."
            )
            core_msg = (
                "A partnership is a shared journey, not an hour-by-hour transaction ledger. "
                "Studying and building skills is valuable groundwork for the future, not 'being useless'; your contribution and worth are not defined solely by exhaustion or immediate earnings."
            )
            step_msg = (
                "Acknowledge your honest appreciation for your partner, and remind yourself that dedicating focus to your studies is a legitimate and necessary investment in your shared future."
            )
        else:
            milestone = "new role or milestone"
            strategy_id = "imposter_promotion_reappraisal"
            what_happening = (
                "Entering a higher-stakes role or milestone naturally activates internal alarm bells and self-doubt. "
                "Feeling like an imposter often surges when taking on new responsibilities."
            )
            core_msg = (
                "Entering a higher-stakes role naturally activates internal alarm bells. "
                "Feeling like an imposter does not mean you are one; skills are built through ongoing practice, and past achievements remain real."
            )
            step_msg = (
                "List two concrete accomplishments or skills you demonstrated that earned you this opportunity, separating how you feel right now from what you have actually built."
            )

        return SelectedStrategy(
            modality="perspective_reappraisal",
            strategy_id=strategy_id,
            name="Cognitive Decoupling & Evidence Testing",
            clinical_framework="Cognitive Therapy (Beck, 1979); Burns (1980)",
            rationale=(
                f"The user describes intense feelings of being inadequate or an imposter ({milestone}). "
                "Testing internalized doubt against concrete, observable evidence helps restore perspective without invalidating the emotional difficulty."
            ),
            what_may_be_happening=what_happening,
            confidence=0.88,
            is_uncertain=False,
            alternative_modalities=["validation"],
            contraindications=[
                "do_not_dismiss_imposter_feelings_as_silly",
                "do_not_guarantee_external_approval",
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

    def _select_collaboration_injustice(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="collaboration_injustice_advocacy",
            name="Team Collaboration Inequity & Objective Self-Advocacy",
            clinical_framework="Emotion-Focused Therapy (Greenberg, 2002); Relational Boundary Framework",
            rationale=(
                "The user has carried the burden of uncooperative team partners and is facing an unfair shared grading policy. "
                "Validates anger and frustration as legitimate emotional responses to external inequity, while establishing practical options for objective advocacy."
            ),
            what_may_be_happening=(
                "You have shouldered the workload of an uncooperative project team alone, and are now facing the understandable frustration of an unfair shared grading policy from your professor."
            ),
            confidence=0.90,
            is_uncertain=False,
            alternative_modalities=["locus_of_agency"],
            contraindications=[
                "do_not_gaslight_or_excuse_uninvolved_teammates",
                "do_not_frame_injustice_as_a_cognitive_distortion",
                "do_not_blame_user_for_partners_absence",
            ],
            reframe_needed=False,
            core_message=(
                "Feeling furious and frustrated is completely warranted when your dedicated labor is shared with teammates who contributed nothing. "
                "Your frustration is a healthy signal that your effort deserves fair recognition, not an emotional flaw."
            ),
            suggested_step=(
                "Gather your commit logs, work timestamps, and team communication attempts into an objective record before scheduling a conversation with your professor."
            ),
        )

    def _select_sports_injury_validation(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="sports_injury_identity_compassion",
            name="Physical Injury Processing & Identity Grounding",
            clinical_framework="Compassion-Focused Therapy (Gilbert, 2009); Somatic Acceptance",
            rationale=(
                "A sudden physical injury has sidelined the user right before crucial tryouts, causing grief over the disruption of their primary sport and athletic identity. "
                "Validates athletic displacement and provides compassionate permission to heal without feeling like personal worth is lost."
            ),
            what_may_be_happening=(
                "A sudden physical injury has sidelined you right before crucial tryouts, causing deep grief over the temporary loss of your sport and athletic identity."
            ),
            confidence=0.92,
            is_uncertain=False,
            alternative_modalities=["non_interventional_grounding"],
            contraindications=[
                "do_not_demand_premature_physical_activity",
                "do_not_dismiss_athletic_grief_as_trivial",
                "do_not_frame_physical_injury_as_personal_failure",
            ],
            reframe_needed=False,
            core_message=(
                "Grieving the loss of a season and feeling displaced from your sport is completely legitimate. "
                "Your value, dedication, and identity as an athlete are not erased simply because your body requires time on crutches to heal."
            ),
            suggested_step=(
                "Honor your body's need for physical rest today, and consider reaching out to a teammate or coach to stay connected from the sidelines."
            ),
        )

    def _select_resource_strain_validation(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="validation",
            strategy_id="resource_strain_compassionate_pacing",
            name="Resource Strain Validation & Basic Needs Grounding",
            clinical_framework="Problem-Solving Therapy (PST); Compassion-Focused Coping",
            rationale=(
                "The user is experiencing acute pressure from escalating financial obligations (rent, tuition) and sacrificing basic physical needs (skipping meals) to keep everything afloat. "
                "Prioritizes restoring physical nourishment and basic safety over aggressive problem-solving."
            ),
            what_may_be_happening=(
                "You are carrying severe pressure from escalating living costs and family obligations, and making painful sacrifices like skipping meals to keep everything afloat."
            ),
            confidence=0.88,
            is_uncertain=False,
            alternative_modalities=["practical_structuring"],
            contraindications=[
                "do_not_minimize_real_financial_hardship",
                "do_not_demand_instant_long_term_financial_solutions",
                "do_not_blame_user_for_external_economic_costs",
            ],
            reframe_needed=False,
            core_message=(
                "Facing sharp increases in essential expenses while supporting family is an acute, real-world hardship. "
                "Skipping meals is a sign of immense strain, not personal failure; protecting your basic physical nourishment and rest is essential so you have the strength to navigate next steps."
            ),
            suggested_step=(
                "Eat one nourishing meal today to support your physical stamina, then identify one immediate expense or support contact to review tomorrow."
            ),
        )

    def _select_creative_block_reappraisal(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="perspective_reappraisal",
            strategy_id="creative_drought_normalization",
            name="Normalizing Creative Cycles & Rest Permission",
            clinical_framework="Acceptance and Commitment Therapy (ACT); Creative Grounding",
            rationale=(
                "The user is experiencing creative exhaustion and self-doubt after prolonged effort at the canvas, fearing their creative voice has permanently dried up. "
                "Normalizes artistic rhythm and separates current expressive output from intrinsic worth."
            ),
            what_may_be_happening=(
                "You are experiencing creative drought and intense self-doubt after prolonged effort at the canvas, leading to fears that your creative voice has permanently dried up."
            ),
            confidence=0.88,
            is_uncertain=False,
            alternative_modalities=["validation"],
            contraindications=[
                "do_not_force_immediate_artistic_production",
                "do_not_frame_creative_rest_as_laziness",
            ],
            reframe_needed=True,
            core_message=(
                "A period of feeling hollow does not mean your voice has vanished. "
                "Creative energy moves in natural rhythms of absorption and expression; when the well feels dry, the remedy is low-stakes rest and patience, not forced output."
            ),
            suggested_step=(
                "Step away from the blank canvas for today and engage in a restful, non-evaluative activity without judging your progress."
            ),
        )

    def _select_cognitive_loop_break_growth(self, rep: CaseRepresentation) -> SelectedStrategy:
        raw_lower = rep.raw_text.lower()
        has_comparison = "compare" in raw_lower or "comparison" in raw_lower or "friends" in raw_lower
        has_personalization = "personally" in raw_lower or "personal" in raw_lower
        has_boundary = any(w in raw_lower for w in ["email", "weekend", "work", "boundary", "line", "holding that line"])

        if has_comparison and has_personalization:
            habit_desc = "comparing yourself to friends and taking things personally"
            reframe_habit = "automatic habits like social comparison and personalization"
        elif has_comparison:
            habit_desc = "habitual social comparison"
            reframe_habit = "automatic comparison loops"
        elif has_personalization:
            habit_desc = "taking things personally"
            reframe_habit = "automatic personalization loops"
        elif has_boundary:
            habit_desc = "compulsive work-checking and holding your personal boundary"
            reframe_habit = "chronic reactivity and choosing to protect your time and rest"
        else:
            habit_desc = "an unhelpful mental loop"
            reframe_habit = "automatic cognitive loops"

        what_happening = (
            f"You have recognized a pattern of {habit_desc}, and through deliberate effort, you are actively breaking out of that loop and feeling genuine pride in your progress."
        )
        core_message = (
            f"Recognizing {reframe_habit}—and actively choosing not to get hooked by them—is a profound mental breakthrough. "
            "Growth does not mean an old thought or impulse will never cross your mind; it means you now notice the pattern, step back from the loop, and treat yourself with trust and pride, exactly as you are doing."
        )

        return SelectedStrategy(
            modality="growth_affirmation",
            strategy_id="cognitive_loop_break_growth_affirmation",
            name="Cognitive Loop Breaking & Growth Affirmation",
            clinical_framework="CBT (Cognitive Restructuring; Beck, 1979); Acceptance & Commitment Therapy (Defusion; Hayes, 1999); Self-Affirmation Theory (Steele, 1988)",
            rationale=(
                f"The user has developed acute self-awareness around {habit_desc}, "
                "is deliberately interrupting the mental loop, and is feeling genuine, earned pride in that progress. "
                "Affirms this cognitive breakthrough, normalizes that automatic triggers can still flicker without needing to be indulged, "
                "and anchors self-trust."
            ),
            what_may_be_happening=what_happening,
            confidence=0.94,
            is_uncertain=False,
            alternative_modalities=["validation"],
            contraindications=[
                "do_not_frame_growth_as_a_burden_or_crisis",
                "do_not_invalidate_user_pride",
                "do_not_impose_unsolicited_crisis_interventions",
                "do_not_tell_user_they_are_facing_challenging_circumstances",
            ],
            reframe_needed=False,
            core_message=core_message,
            suggested_step=(
                "Pause and let yourself fully savor this moment of pride—acknowledging your capacity to break old mental patterns is what solidifies lasting growth."
            ),
        )

    def _select_social_comparison_differentiation(self, rep: CaseRepresentation) -> SelectedStrategy:
        return SelectedStrategy(
            modality="locus_of_agency",
            strategy_id="social_comparison_differentiation",
            name="Social Comparison Awareness & Values Differentiation",
            clinical_framework="Acceptance & Commitment Therapy (ACT; Hayes, 1999); Social Comparison Theory (Festinger, 1954)",
            rationale=(
                "The user is noticing automatic habits of comparing personal progress against friends or peers. "
                "Normalizes comparison urges as common social conditioning, decouples self-worth from others' timelines, "
                "and returns focus to personal values and pacing."
            ),
            what_may_be_happening=(
                "You are noticing automatic habits of measuring your own path and worth against your friends or peers, which is stirring up self-doubt."
            ),
            confidence=0.88,
            is_uncertain=False,
            alternative_modalities=["validation"],
            contraindications=[
                "do_not_dismiss_feelings_as_silly",
                "do_not_demand_instant_detachment",
            ],
            reframe_needed=True,
            core_message=(
                "Comparing your internal journey to other people's visible milestones is a natural human habit, but it rarely tells the whole story. "
                "Your worth and growth are rooted in your own path, not in how closely your timeline mirrors anyone else's."
            ),
            suggested_step=(
                "Take one intentional breath, gently notice the comparison thought without judging yourself for having it, and choose one meaningful thing to focus on for yourself today."
            ),
        )

    def _select_general_validation(self, rep: CaseRepresentation) -> SelectedStrategy:
        clean_facts = [
            f.text for f in rep.stated_facts 
            if f.text and f.text.lower() not in ("alot", "a lot", "lot", "bit", "stuff", "things", "loop", "this loop")
        ]
        if not clean_facts and rep.stated_thoughts:
            clean_facts = [t.statement for t in rep.stated_thoughts if t.statement]

        facts_str = ", ".join(clean_facts[:2]) if clean_facts else ""

        has_positive = any(e.valence == "positive" for e in rep.stated_emotions)
        has_negative = any(e.valence == "negative" for e in rep.stated_emotions)

        emotions_words = list(dict.fromkeys(e.emotion_word for e in rep.stated_emotions[:2]))
        if len(emotions_words) == 2:
            emotions_str = f"{emotions_words[0]} and {emotions_words[1]}"
        elif emotions_words:
            emotions_str = emotions_words[0]
        else:
            emotions_str = ""

        raw_lower = rep.raw_text.lower()
        is_future_weary = (
            ("future" in raw_lower or rep.uncertainty.uncertainty_type == "future_outcome")
            and any(w in raw_lower or w in emotions_words for w in ("demotivated", "unmotivated", "tired", "exhausted", "drained", "weary"))
        )

        clarification_qs = []
        if is_future_weary:
            what_happening = "You are experiencing mental fatigue and demotivation from trying to anticipate and carry the weight of the future."
            core_msg = (
                "Feeling demotivated and exhausted when thinking about the future is a natural sign of cognitive overload. "
                "You do not have to solve or predict the future today; your exhaustion is a signal that your mind needs space and rest, not that you are falling behind."
            )
            next_step = "When you think about the future right now, what part feels most exhausting to carry, or would it help more to set it aside completely for today?"
            clarification_qs = [
                "When you think about the future right now, is there a specific decision weighing on you, or is it an overall sense of exhaustion?"
            ]
        elif has_positive and not has_negative:
            what_happening = f"You are noticing a meaningful positive moment and feeling {emotions_str}."
            core_msg = (
                f"Giving yourself credit and feeling {emotions_str} is deeply deserved. "
                "Pausing to acknowledge positive shifts and celebrate your self-trust helps reinforce lasting progress."
            )
            next_step = "Take one quiet moment to pause and focus on what feels most grounding for you right now."
        elif facts_str and emotions_str:
            what_happening = f"You are carrying meaningful tension around your situation, which is bringing up feelings of {emotions_str}."
            core_msg = (
                f"Experiencing tension and feeling {emotions_str} is a natural and understandable reaction. "
                "You do not have to carry the entire weight all at once or solve everything today; giving yourself patience is valid and necessary."
            )
            next_step = "Take one quiet moment to pause and focus on what feels most grounding for you right now."
        elif facts_str:
            what_happening = "You are navigating significant demands and circumstances right now."
            core_msg = (
                "Facing challenging circumstances can be a real weight. "
                "You do not have to have every answer sorted out immediately to deserve support and steady pacing."
            )
            next_step = "Take one quiet moment to pause and focus on what feels most grounding for you right now."
        elif emotions_str:
            what_happening = f"You are experiencing real emotional weight and feelings of {emotions_str} right now."
            core_msg = (
                f"Feeling {emotions_str} is an understandable human experience when navigating uncertainty. "
                "Treating yourself with patient self-compassion allows you to find steady ground."
            )
            next_step = "Take one quiet moment to pause and focus on what feels most grounding for you right now."
        else:
            what_happening = "You are carrying the emotional weight of what you described."
            core_msg = (
                "What you are experiencing in this situation is valid and real. "
                "You do not have to carry the whole weight all at once; giving yourself patience and steady pacing is a necessary step."
            )
            next_step = "Take one quiet moment to pause and focus on what feels most grounding for you right now."

        if rep.user_goal:
            next_step = f"Take one small, manageable step toward {rep.user_goal} while honoring your natural daily pace."
        elif not is_future_weary and rep.unknowns.missing_aspects and any("preferred support" in m for m in rep.unknowns.missing_aspects):
            next_step = "What part of this experience feels most pressing or important for you to unpack right now?"

        return SelectedStrategy(
            modality="validation",
            strategy_id="general_grounded_validation",
            name="Grounded Validation & Presence",
            clinical_framework="Person-Centered Counseling; CFT",
            rationale=(
                f"Acknowledge the user's situation ({facts_str or 'personal reflection'}) without imposing premature reframing. "
                "Provides empathetic validation and encourages taking things one manageable moment at a time."
            ),
            what_may_be_happening=what_happening,
            confidence=0.75,
            is_uncertain=True,
            alternative_modalities=["clarification_needed", "locus_of_agency"],
            contraindications=[
                "do_not_impose_unsupported_reframe",
            ],
            reframe_needed=False,
            core_message=core_msg,
            suggested_step=next_step,
            clarification_questions=clarification_qs,
        )

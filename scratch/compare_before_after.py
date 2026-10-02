import sys
import json
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
from app.psychology import parse_situation, select_strategy
from app.context_engine.case_frame import CaseFrame
from app.psychology.case_representation import extract_case_representation
from app.psychology.case_strategy_selector import CaseStrategySelector
from app.narrative_gen.grounded_generator import GroundedNarrativeGenerator
from app.narrative_gen.scene_planner import ScenePlanner

text = "My supervisor gave credit for my three-month project to another colleague in the team meeting. I'm furious and feeling completely powerless."

# --- BEFORE (CURRENT CODE IN app/main.py) ---
psych_profile = parse_situation(text)
strategy = select_strategy(psych_profile)
reasoning_dict_before = psych_profile.structured_reasoning()
reasoning_dict_before.update({
    "what_may_be_happening": f"{psych_profile.trigger} → {psych_profile.pattern}",
    "what_you_can_control": strategy.what_is_controllable,
    "what_is_not_controllable": strategy.what_is_not_controllable,
    "reframe": strategy.reframe,
    "next_step": strategy.concrete_action,
})
technique_before = {
    "name": strategy.name,
    "citation": strategy.citation,
}

# --- AFTER (WITH GROUNDED MAPPING) ---
case_rep = extract_case_representation(text)
strategy_selector = CaseStrategySelector()
selected_strategy = strategy_selector.select(case_rep)

controllable_str = (
    "; ".join(case_rep.controllability.potentially_controllable)
    if case_rep.controllability.potentially_controllable
    else "Focus on personal boundaries, self-advocacy, and emotional safety."
)
uncontrollable_str = (
    "; ".join(case_rep.controllability.potentially_uncontrollable)
    if case_rep.controllability.potentially_uncontrollable
    else "Past events and external actions of others."
)
reasoning_dict_after = dict(reasoning_dict_before)
reasoning_dict_after.update({
    "what_may_be_happening": selected_strategy.rationale,
    "what_you_can_control": controllable_str,
    "what_is_not_controllable": uncontrollable_str,
    "reframe": selected_strategy.core_message,
    "next_step": selected_strategy.suggested_step,
})
technique_after = {
    "name": selected_strategy.modality.replace("_", " ").title(),
    "citation": selected_strategy.clinical_framework,
}

print("=" * 80)
print("BEFORE (WHAT FRONTEND CURRENTLY RENDERS):")
print("=" * 80)
print("What may be happening :", reasoning_dict_before["what_may_be_happening"])
print("What you can control  :", reasoning_dict_before["what_you_can_control"])
print("Reframe               :", f'“{reasoning_dict_before["reframe"]}”')
print("Next step             :", reasoning_dict_before["next_step"])
print("Citation              :", f"{technique_before['name']} — {technique_before['citation']}")

print("\n" + "=" * 80)
print("AFTER (WHAT FRONTEND WILL RENDER WITH GROUNDED FIX):")
print("=" * 80)
print("What may be happening :", reasoning_dict_after["what_may_be_happening"])
print("What you can control  :", reasoning_dict_after["what_you_can_control"])
print("Reframe               :", f'“{reasoning_dict_after["reframe"]}”')
print("Next step             :", reasoning_dict_after["next_step"])
print("Citation              :", f"{technique_after['name']} — {technique_after['citation']}")

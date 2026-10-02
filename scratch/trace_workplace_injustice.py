import sys
import json

sys.path.insert(0, ".")
from app.psychology import parse_situation, select_strategy
from app.context_engine.case_frame import CaseFrame
from app.psychology.case_representation import extract_case_representation
from app.psychology.case_strategy_selector import CaseStrategySelector
from app.narrative_gen.grounded_generator import GroundedNarrativeGenerator
from app.narrative_gen.scene_planner import ScenePlanner

text = "My supervisor gave credit for my three-month project to another colleague in the team meeting. I'm furious and feeling completely powerless."

print("=" * 80)
print("1. RUNNING LEGACY PIPELINE AS EXECUTED IN app/main.py lines 96-140")
print("=" * 80)
psych_profile = parse_situation(text)
strategy = select_strategy(psych_profile)
reasoning_dict = psych_profile.structured_reasoning()
reasoning_dict.update({
    "what_may_be_happening": f"{psych_profile.trigger} -> {psych_profile.pattern}",
    "what_you_can_control": strategy.what_is_controllable,
    "what_is_not_controllable": strategy.what_is_not_controllable,
    "reframe": strategy.reframe,
    "next_step": strategy.concrete_action,
})

case_frame = CaseFrame(
    raw_text=text,
    distortions=[],
    emotion_scores={},
    core_emotion="anger",
    situation_type=psych_profile.situation_id,
    trigger=psych_profile.trigger,
    pattern=psych_profile.pattern,
    is_external_threat=psych_profile.is_external_threat,
    controllable=strategy.what_is_controllable,
    uncontrollable=strategy.what_is_not_controllable,
    strategy=strategy.to_dict(),
    specific_reframe=strategy.reframe,
    concrete_action=strategy.concrete_action,
    reasoning=reasoning_dict,
)
case_frame.build_summary()

print(f"Situation ID detected: {psych_profile.situation_id}")
print(f"Trigger: {psych_profile.trigger}")
print(f"Pattern: {psych_profile.pattern}")
print(f"Strategy Name: {strategy.name}")
print("\nLegacy case_frame.reasoning (RETURNED IN API RESPONSE TO FRONTEND):")
for k, v in reasoning_dict.items():
    print(f"  [{k}]: {v}")

print("\n" + "=" * 80)
print("2. RUNNING GROUNDED PIPELINE (STAGE 1-4) AS EXECUTED IN app/main.py lines 150-167")
print("=" * 80)
case_rep = extract_case_representation(text)
strategy_selector = CaseStrategySelector()
selected_strategy = strategy_selector.select(case_rep)
grounded_gen = GroundedNarrativeGenerator()
raw_story = grounded_gen.generate(case_rep, selected_strategy)
planner = ScenePlanner()
enriched_scenes = planner.plan_grounded_scenes(raw_story, case_rep, selected_strategy)

print(f"Grounded Facts: {[f.text for f in case_rep.stated_facts]}")
print(f"Grounded Stated Emotions: {[(e.emotion_word, e.valence) for e in case_rep.stated_emotions]}")
print(f"Grounded descriptive_summary: {case_rep.descriptive_summary}")
print(f"Grounded Controllability: controllable={case_rep.controllability.potentially_controllable}, uncontrollable={case_rep.controllability.potentially_uncontrollable}")
print(f"Grounded Strategy Modality: {selected_strategy.modality}")
print(f"Clinical Framework: {selected_strategy.clinical_framework}")
print(f"Reframe Needed: {selected_strategy.reframe_needed}")
print(f"Grounded Rationale: {selected_strategy.rationale}")
print(f"Grounded Core Message: {selected_strategy.core_message}")
print(f"Grounded Suggested Step: {selected_strategy.suggested_step}")
print(f"Contraindications respected: {case_rep.support_needs.contraindications}")

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.psychology import parse_situation, select_strategy
from app.context_engine.case_frame import CaseFrame
from app.narrative_gen.generator import NarrativeGenerator
from app.image_gen.prompt_builder import build_scene_prompts

text = "I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful"
profile = parse_situation(text)
strategy = select_strategy(profile)

reasoning_dict = profile.structured_reasoning()
reasoning_dict.update({
    "what_may_be_happening": f"{profile.trigger} → {profile.pattern}",
    "what_you_can_control": strategy.what_is_controllable,
    "what_is_not_controllable": strategy.what_is_not_controllable,
    "reframe": strategy.reframe,
    "next_step": strategy.concrete_action,
})

case_frame = CaseFrame(
    raw_text=text,
    situation_type=profile.situation_id,
    trigger=profile.trigger,
    pattern=profile.pattern,
    is_external_threat=profile.is_external_threat,
    controllable=strategy.what_is_controllable,
    uncontrollable=strategy.what_is_not_controllable,
    strategy=strategy.to_dict(),
    specific_reframe=strategy.reframe,
    concrete_action=strategy.concrete_action,
    reasoning=reasoning_dict,
)
case_frame.build_summary()

gen = NarrativeGenerator(use_small=True)
story = gen.generate(case_frame, {"name": strategy.name, "citation": strategy.citation, "description": strategy.mechanism})
gen.unload()

prompts = build_scene_prompts(story)

print(f"Total scenes: {len(story['scenes'])}")
print(f"Total prompts: {len(prompts)}\n")

for i, (scene, p) in enumerate(zip(story["scenes"], prompts)):
    print(f"=== SCENE {i+1} [{scene.get('stage_category')}] ===")
    print(f"FINAL PROMPT ({len(p.split())} words, {len(p)} chars):")
    print(f"\"{p}\"")
    print(f"\nRAW SCENE STATE PIECES:")
    print(f"  character_action: {scene.get('character_action')}")
    print(f"  posture: {scene.get('posture')}")
    print(f"  expression: {scene.get('expression')}")
    print(f"  gaze: {scene.get('gaze')}")
    print(f"  important_objects: {scene.get('important_objects')}")
    print(f"  environment: {scene.get('environment')}")
    print(f"  composition: {scene.get('composition')}")
    print(f"  camera_distance: {scene.get('camera_distance')}")
    print(f"  camera_angle: {scene.get('camera_angle')}")
    print(f"  lighting: {scene.get('lighting')}")
    print(f"  visual_mood: {scene.get('visual_mood')}")
    print()

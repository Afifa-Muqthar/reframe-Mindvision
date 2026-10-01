import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.psychology import parse_situation, select_strategy
from app.context_engine.case_frame import CaseFrame
from app.narrative_gen.generator import NarrativeGenerator

text = "I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful"
prof = parse_situation(text)
strat = select_strategy(prof)
reasoning = prof.structured_reasoning()

print("=" * 60)
print("1. 8 STRUCTURED REASONING FIELDS:")
print("=" * 60)
print(json.dumps(reasoning, indent=2))

cf = CaseFrame(
    raw_text=text,
    situation_type=prof.situation_id,
    trigger=prof.trigger,
    pattern=prof.pattern,
    is_external_threat=prof.is_external_threat,
    controllable=strat.what_is_controllable,
    uncontrollable=strat.what_is_not_controllable,
    strategy=strat.to_dict(),
    specific_reframe=strat.reframe,
    concrete_action=strat.concrete_action,
    reasoning=reasoning,
)
cf.build_summary()

gen = NarrativeGenerator(use_small=True)
story = gen.generate(cf, {"name": strat.name, "citation": strat.citation, "description": strat.mechanism})
gen.unload()

print("\n" + "=" * 60)
print("2. GENERATED COMIC STORY:")
print("=" * 60)
print(f"Title: {story['title']}")
print(f"Context: {story['context']}")
print(f"Visual Style: {story['visual_style']}")
print(f"Number of scenes: {len(story['scenes'])}")

print("\n" + "=" * 60)
print("3. INDIVIDUAL SCENES BREAKDOWN:")
print("=" * 60)
for idx, s in enumerate(story["scenes"], start=1):
    print(f"\n--- Panel {idx} [{s.get('stage', s.get('stage_category'))}] ---")
    print(f"Dialogue: {s.get('dialogue')}")
    print(f"Caption:  {s.get('caption')}")
    print(f"Visual:   {s.get('visual_description')}")
    print(f"Narrative: {s.get('narrative')}")

print("\n" + "=" * 60)
print("4. CONNECTED VOICE SCRIPT:")
print("=" * 60)
print(story.get("voice_script"))

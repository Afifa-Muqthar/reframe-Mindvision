import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.context_engine.case_frame import CaseFrame
from app.nlp.distortions import detect_distortions
from app.principle_selector.selector import select_techniques
from app.narrative_gen.generator import NarrativeGenerator, format_narrative_text
from app.image_gen.prompt_builder import build_scene_prompts
from app.image_gen.storyboard import compose_comic_strip


def test_core_exam_scenario():
    input_text = "I have an exam tomorrow and I keep thinking I'm going to fail."
    print("=" * 60)
    print("INPUT TEXT:", input_text)
    print("=" * 60)

    # 1. Distortion detection
    distortions = detect_distortions(input_text)
    print("Detected Distortions:", distortions)

    # 2. Case Frame
    case_frame = CaseFrame(
        raw_text=input_text,
        distortions=distortions,
        emotion_scores={"fear": 0.82, "sadness": 0.15},
        core_emotion="fear"
    )
    case_frame.build_summary()
    print("Case Frame Summary:", case_frame.summary)

    # 3. Principle Selector
    techniques = select_techniques(case_frame)
    primary = techniques[0]
    print(f"Selected Technique: {primary['name']} ({primary['citation']})")

    # 4. Structured 3-Stage Narrative
    gen = object.__new__(NarrativeGenerator)
    story = gen._build_structured_story("", case_frame, primary)

    print("\n--- STRUCTURED STORY OBJECT ---")
    print("Title:", story["title"])
    print("Context Archetype:", story["context"])
    print("Character Anchor:")
    print("  Description:", story["character"]["description"])
    print("  Appearance :", story["character"]["appearance"])
    print("  Clothing   :", story["character"]["clothing"])

    print("\n--- 3 COMIC SCENES ---")
    for i, s in enumerate(story["scenes"], start=1):
        print(f"\n[PANEL {i}: {s['stage'].upper()}]")
        print(f"  Bubble ({s['bubble_type'].title()}): \"{s['dialogue']}\"")
        print(f"  Narrative: {s['narrative']}")
        print(f"  Visual Scene: {s['visual_description']}")

    # 5. Image Prompts for SD-Turbo
    prompts = build_scene_prompts(story)
    print("\n--- GENERATED SD-TURBO PROMPTS ---")
    for i, p in enumerate(prompts, start=1):
        print(f"\nPanel {i} Prompt:")
        print(f"  {p}")

    # 6. Connected Voice Script
    print("\n--- CONNECTED VOICE SCRIPT FOR KOKORO TTS ---")
    print(story["voice_script"])

    # 7. Comic Strip Composition Verification
    out_comic_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions", "demo_exam_comic.png")
    compose_comic_strip([], story["scenes"], out_comic_path, title=f"REFRAME COMIC: {story['context'].upper()}")
    print("\n--- COMIC COMPOSITION ---")
    print(f"Composed comic strip saved to: {out_comic_path} (exists={os.path.exists(out_comic_path)})")
    print("=" * 60)


if __name__ == "__main__":
    test_core_exam_scenario()

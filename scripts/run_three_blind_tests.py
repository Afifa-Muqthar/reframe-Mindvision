"""
Blind End-to-End Tests for 3 New Psychological Scenarios:
1. Task Overwhelm
2. Social Comparison
3. Future Uncertainty

Configuration (Strictly production settings):
- Model: stabilityai/sd-turbo
- Resolution: 512x512 native generation
- Dtype: FP16 (torch.float16) on CUDA
- Steps: 2 inference steps
- Guidance scale: 0.0
- Negative prompt: None
- Prompt Compiler: 38-48 word observable format
- Per-panel seed: base_seed + idx * 17
- Lanczos downsampling during comic strip composition
"""

import os
import sys
import time
import json
import subprocess
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ssl_workaround import configure_hf_ssl_verify
configure_hf_ssl_verify()

from app.psychology import parse_situation, select_strategy
from app.context_engine.case_frame import CaseFrame
from app.narrative_gen.generator import NarrativeGenerator
from app.image_gen.prompt_builder import build_scene_prompts
from app.image_gen.generator import ImageGenerator
from app.image_gen.storyboard import compose_comic_strip

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "experiments")
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIOS = [
    {
        "id": "task_overwhelm",
        "name": "Task Overwhelm",
        "input_text": "I have 20 things to do today and I feel completely overwhelmed and paralyzed, I don't know where to start.",
        "base_seed": 101,
    },
    {
        "id": "social_comparison",
        "name": "Social Comparison",
        "input_text": "Everyone my age seems to be succeeding and doing so much better than me, while I feel like a total failure and stuck.",
        "base_seed": 202,
    },
    {
        "id": "future_uncertainty",
        "name": "Future Uncertainty",
        "input_text": "I don't know what will happen with my future and career, and the constant uncertainty is making me anxious and terrified.",
        "base_seed": 303,
    },
]

def get_gpu_telemetry():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True
        )
        parts = [p.strip() for p in out.strip().split(",")]
        return {
            "temp_c": int(parts[0]),
            "mem_used_mb": int(parts[1]),
            "mem_total_mb": int(parts[2]),
        }
    except Exception as e:
        return {"error": str(e)}

def run_scenario(scenario_def, image_gen):
    s_id = scenario_def["id"]
    s_name = scenario_def["name"]
    text = scenario_def["input_text"]
    base_seed = scenario_def["base_seed"]
    comic_out = os.path.join(OUT_DIR, f"comic_{s_id}.png")

    print(f"\n{'='*60}")
    print(f"RUNNING SCENARIO: {s_name.upper()}")
    print(f"Input: \"{text}\"")
    print(f"{'='*60}")

    # 1. Psychological Reasoning
    profile = parse_situation(text)
    strategy = select_strategy(profile)
    print(f"  Situation ID: {profile.situation_id}")
    print(f"  Pattern: {profile.pattern}")
    print(f"  Strategy: {strategy.name}")

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

    # 2. Adaptive Narrative Planning
    gen_narrative = NarrativeGenerator(use_small=True)
    story = gen_narrative.generate(case_frame, {"name": strategy.name, "citation": strategy.citation, "description": strategy.mechanism})
    gen_narrative.unload()

    scenes = story.get("scenes", [])
    num_scenes = len(scenes)
    print(f"  Planned Scenes: {num_scenes}")

    # 3. Prompt Compilation
    prompts = build_scene_prompts(story)
    for idx, (sc, pr) in enumerate(zip(scenes, prompts)):
        words = len(pr.split())
        print(f"    [Panel {idx+1}] ({sc.get('stage_category')}): {pr} ({words} words)")

    # 4. Image Generation
    panel_paths = []
    panel_timings = []
    temps = []

    t_img_start = time.perf_counter()
    for idx, prompt in enumerate(prompts):
        panel_seed = base_seed + idx * 17
        panel_file = os.path.join(OUT_DIR, f"{s_id}_panel_{idx+1}.png")
        panel_paths.append(panel_file)

        t_p_start = time.perf_counter()
        image_gen.generate(
            prompt=prompt,
            output_path=panel_file,
            num_inference_steps=2,
            size=512,
            seed=panel_seed,
        )
        t_p_end = time.perf_counter()
        elapsed = t_p_end - t_p_start
        panel_timings.append(round(elapsed, 3))

        telem = get_gpu_telemetry()
        temps.append(telem.get("temp_c", 0))
        cur_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        print(f"    Panel {idx+1} generated in {elapsed:.3f}s | Temp: {telem.get('temp_c')}°C | Peak VRAM: {cur_vram} MB")

    t_img_end = time.perf_counter()
    total_img_time = t_img_end - t_img_start

    # 5. Composite Comic Strip
    compose_comic_strip(
        panel_paths=panel_paths,
        scenes=scenes,
        out_path=comic_out,
        title=f"REFRAMING: {strategy.name.upper()}",
    )
    print(f"  Comic assembled -> {comic_out}")

    # Copy to brain artifact directory for instant viewing
    brain_dir = r"C:\Users\afifa\.gemini\antigravity\brain\ca2b663d-45c2-47da-8930-6644069a3ea3"
    if os.path.exists(brain_dir):
        import shutil
        brain_comic = os.path.join(brain_dir, f"comic_{s_id}.png")
        shutil.copyfile(comic_out, brain_comic)
        print(f"  Copied to brain artifact -> {brain_comic}")

    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

    return {
        "scenario_id": s_id,
        "scenario_name": s_name,
        "input_text": text,
        "situation_id": profile.situation_id,
        "pattern": profile.pattern,
        "strategy": strategy.name,
        "num_panels": num_scenes,
        "total_image_time_s": round(total_img_time, 3),
        "per_panel_timings_s": panel_timings,
        "peak_vram_mb": peak_vram,
        "peak_gpu_temp_c": max(temps) if temps else 0,
        "comic_output": comic_out,
        "panel_paths": panel_paths,
        "prompts": prompts,
        "scenes": [
            {
                "stage": sc.get("stage_category"),
                "dialogue": sc.get("dialogue"),
                "caption": sc.get("caption"),
            }
            for sc in scenes
        ]
    }

def main():
    print("=== STARTING 3 BLIND END-TO-END SCENARIO TESTS ===")
    telem_init = get_gpu_telemetry()
    print(f"Initial GPU State: {telem_init}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    image_gen = ImageGenerator()

    all_results = []
    t_suite_start = time.perf_counter()

    for sc in SCENARIOS:
        res = run_scenario(sc, image_gen)
        all_results.append(res)

    t_suite_end = time.perf_counter()
    suite_time = t_suite_end - t_suite_start

    image_gen.unload()

    summary_file = os.path.join(OUT_DIR, "three_blind_tests_results.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL 3 SCENARIOS COMPLETED SUCCESSFULLY!")
    print(f"Total Suite Time: {suite_time:.2f}s")
    for r in all_results:
        print(f" - {r['scenario_name']}: {r['num_panels']} panels in {r['total_image_time_s']}s (Peak Temp: {r['peak_gpu_temp_c']}°C, Peak VRAM: {r['peak_vram_mb']} MB)")
    print(f"Summary JSON: {summary_file}")

if __name__ == "__main__":
    main()

"""
End-to-End 5-Panel Comic Generation for the Breakup + Uncertainty Scenario:
- Model: stabilityai/sd-turbo
- Resolution: 512x512 native generation
- Precision: FP16 (torch.float16) on CUDA
- Inference steps: 2
- Guidance scale: 0.0
- No negative prompt
- Compact 38-48 word observable prompt compiler
- Per-panel seed: base_seed + i * 17 (base_seed=42)
- Lanczos downsampling during comic strip composition
- Full telemetry: per-panel times, total time, peak VRAM, temperatures
"""

import os
import sys
import time
import json
import subprocess
import torch
from PIL import Image

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
COMIC_OUT = os.path.join(OUT_DIR, "breakup_5panel_comic.png")

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

def main():
    print("=======================================================")
    print("RUNNING END-TO-END 5-PANEL BREAKUP COMIC TEST (SD-TURBO)")
    print("=======================================================")
    
    text = "I had a breakup because of my uncertainty and also I wasn't sure I think I did not love him it's kind of painful"
    print(f"Input Situation: \"{text}\"\n")

    # Step 1: Psychological Parsing & Strategy Selection
    print("Step 1: Parsing situation & selecting psychological strategy...")
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

    # Step 2: Narrative Generation
    print("\nStep 2: Generating adaptive multi-panel narrative...")
    gen_narrative = NarrativeGenerator(use_small=True)
    story = gen_narrative.generate(case_frame, {"name": strategy.name, "citation": strategy.citation, "description": strategy.mechanism})
    gen_narrative.unload()

    scenes = story.get("scenes", [])
    print(f"  Total planned scenes: {len(scenes)}")

    # Step 3: Prompt Compilation
    print("\nStep 3: Compiling compact observable prompts...")
    prompts = build_scene_prompts(story)
    for idx, (sc, pr) in enumerate(zip(scenes, prompts)):
        words = len(pr.split())
        print(f"  [Panel {idx+1}] ({sc.get('stage_category')}): {pr} ({words} words)")

    # Step 4: Sequential 512x512 FP16 Panel Generation
    print("\nStep 4: Initializing ImageGenerator (512x512, FP16, CUDA)...")
    telem_start = get_gpu_telemetry()
    print(f"Initial GPU Telemetry: {telem_start}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    image_gen = ImageGenerator()
    
    base_seed = 42
    panel_paths = []
    panel_timings = []
    temps = [telem_start.get("temp_c", 0)]

    t_total_start = time.perf_counter()

    for idx, (scene, prompt) in enumerate(zip(scenes, prompts)):
        panel_seed = base_seed + idx * 17
        panel_file = os.path.join(OUT_DIR, f"breakup_panel_{idx+1}.png")
        panel_paths.append(panel_file)

        print(f"\n--- Generating Panel {idx+1}/{len(scenes)} (Seed: {panel_seed}) ---")
        t_panel_start = time.perf_counter()
        
        image_gen.generate(
            prompt=prompt,
            output_path=panel_file,
            num_inference_steps=2,
            size=512,
            seed=panel_seed,
        )
        t_panel_end = time.perf_counter()
        elapsed = t_panel_end - t_panel_start
        panel_timings.append(round(elapsed, 3))
        
        cur_telem = get_gpu_telemetry()
        cur_temp = cur_telem.get("temp_c", 0)
        temps.append(cur_temp)
        cur_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        print(f"  Panel {idx+1} completed in {elapsed:.3f}s | Temp: {cur_temp}°C | Peak PyTorch VRAM: {cur_vram} MB")

    t_total_end = time.perf_counter()
    total_image_time = t_total_end - t_total_start

    image_gen.unload()
    telem_end = get_gpu_telemetry()
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)

    # Step 5: Comic Strip Assembly with Lanczos Downsampling
    print("\nStep 5: Compositing adaptive comic strip with Lanczos downsampling...")
    t_comp_start = time.perf_counter()
    compose_comic_strip(
        panel_paths=panel_paths,
        scenes=scenes,
        out_path=COMIC_OUT,
        title=f"REFRAMING: {strategy.name.upper()}",
    )
    t_comp_end = time.perf_counter()
    comp_time = t_comp_end - t_comp_start
    print(f"Comic strip assembled in {comp_time:.3f}s -> Saved to: {COMIC_OUT}")

    total_pipeline_time = total_image_time + comp_time

    results = {
        "scenario": "breakup_uncertainty",
        "total_panels": len(scenes),
        "total_generation_time_seconds": round(total_image_time, 3),
        "total_pipeline_time_seconds": round(total_pipeline_time, 3),
        "per_panel_timings_seconds": panel_timings,
        "peak_vram_mb": peak_vram_mb,
        "peak_gpu_temp_c": max(temps),
        "start_gpu_temp_c": telem_start.get("temp_c", 0),
        "end_gpu_temp_c": telem_end.get("temp_c", 0),
        "comic_output_path": COMIC_OUT,
        "panel_paths": panel_paths,
        "prompts": prompts,
    }

    res_path = os.path.join(OUT_DIR, "breakup_e2e_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== FINAL TEST SUMMARY ===")
    print(f"Total Panels: {len(scenes)}")
    print(f"Total Image Generation Time: {total_image_time:.3f}s (Average: {total_image_time/len(scenes):.3f}s / panel)")
    print(f"Per-Panel Times: {panel_timings}")
    print(f"Peak PyTorch VRAM: {peak_vram_mb} MB")
    print(f"Peak GPU Temperature: {max(temps)}°C")
    print(f"Comic Output: {COMIC_OUT}")
    print("\nTEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

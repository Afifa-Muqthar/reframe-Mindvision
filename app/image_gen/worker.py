"""
Dedicated process-isolated worker for Stable Diffusion image generation.

Runs in an isolated child process to safeguard the parent Flask server from
heavy CPU memory allocation and VAE-decode native crashes.

Reads structured job arguments from a JSON configuration file rather than
string-interpolated code execution.
"""

import argparse
import json
import os
import sys

from app.image_gen.generator import ImageGenerator
from app.image_gen.storyboard import compose_comic_strip


def run_worker(config_path: str):
    if not os.path.exists(config_path):
        sys.stderr.write(f"Worker config not found: {config_path}\n")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    session_id = job.get("session_id", 0)
    prompts = job.get("prompts", [])
    scenes = job.get("scenes", [])
    output_path = job.get("output_path", "")
    seed = job.get("seed", session_id)
    steps = job.get("num_inference_steps", 2)
    title = job.get("title", "REFRAME COMIC")

    if not prompts or not output_path:
        sys.stderr.write("Invalid job configuration: prompts or output_path missing.\n")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    panel_paths = [
        os.path.join(out_dir, f"temp_session_{session_id}_panel{i + 1}.png")
        for i in range(len(prompts))
    ]

    try:
        # 1. Generate SD-Turbo panel artwork sequentially
        gen = ImageGenerator()
        base_seed = seed
        for i, (prompt, path) in enumerate(zip(prompts, panel_paths)):
            panel_seed = (base_seed + i * 17) if base_seed is not None else None
            gen.generate(prompt, path, seed=panel_seed, num_inference_steps=steps)
        gen.unload()

        # 2. Composite comic strip with borders, bubbles, and captions
        compose_comic_strip(panel_paths, scenes, output_path, title=title)
        print("SUCCESS")
    except Exception as e:
        sys.stderr.write(f"Image worker failed: {e}\n")
        sys.exit(1)
    finally:
        # 3. Clean up intermediate panel files
        for p in panel_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def main():
    parser = argparse.ArgumentParser(description="SD-Turbo Comic Worker")
    parser.add_argument("config", help="Path to JSON configuration file")
    args = parser.parse_args()
    run_worker(args.config)


if __name__ == "__main__":
    main()

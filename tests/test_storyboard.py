import sys
import os
import tempfile
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.image_gen.storyboard import compose_comic_strip


def test_compose_comic_strip_generates_valid_image():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create 3 dummy panel images
        panel_paths = []
        for i in range(3):
            p = os.path.join(tmp_dir, f"panel_{i+1}.png")
            img = Image.new("RGB", (256, 256), (200 + i * 20, 200, 200))
            img.save(p)
            panel_paths.append(p)

        scenes = [
            {
                "stage": "problem",
                "title": "1. PROBLEM",
                "narrative": "Student feeling overwhelmed by tomorrow's exam.",
                "dialogue": "I keep thinking I'm going to fail.",
                "bubble_type": "thought",
            },
            {
                "stage": "reframing",
                "title": "2. REFRAMING",
                "narrative": "Remembering that anxiety is normal and does not guarantee failure.",
                "dialogue": "Being nervous doesn't mean I will fail.",
                "bubble_type": "thought",
            },
            {
                "stage": "resolution",
                "title": "3. RESOLUTION",
                "narrative": "Focusing on a manageable 20-minute review session.",
                "dialogue": "I'll do one chapter now.",
                "bubble_type": "speech",
            },
        ]

        out_path = os.path.join(tmp_dir, "test_comic.png")
        result = compose_comic_strip(panel_paths, scenes, out_path, title="TEST COMIC")

        assert result == out_path
        assert os.path.exists(out_path)

        # Verify image properties
        with Image.open(out_path) as comic_img:
            assert comic_img.format == "PNG"
            w, h = comic_img.size
            # 3 panels of 256 + borders + margins + gutters
            assert w >= 800
            assert h >= 450

"""
Comic Strip Composition Engine.

Composes 3 to 8 generated panel images into an authentic adaptive comic strip.

Features:
- Adaptive grid layout (1x3, 2x2, 2x3, 2x4) preserving exact 256x256 panel resolution without squeezing.
- Solid dark comic panel borders and clean gutters.
- Styled scene headers with color-coded psychological stage themes.
- Genuine speech bubbles (with directional pointer tail) and thought bubbles
  (with trailing circular thought dots).
- Readable comic typography with automatic text wrapping and centering.
- Grounded narrative caption boxes below each panel.
- Overall comic title banner with dynamic psychological progression indicator.
"""

import os
import textwrap
from PIL import Image, ImageDraw, ImageFont

_PANEL_W = 256
_PANEL_H = 256
_MARGIN = 20
_GAP = 18
_HEADER_H = 34
_CAPTION_H = 92
_TOP_BANNER_H = 48

_CANVAS_BG = (248, 250, 252)        # Slate-50 background
_PANEL_BORDER = (30, 41, 59)        # Slate-800 crisp border
_CAPTION_BG = (255, 255, 255)       # White caption box
_CAPTION_FG = (51, 65, 85)          # Slate-700 caption text
_BUBBLE_BG = (255, 255, 255)        # Pure white bubble
_BUBBLE_BORDER = (30, 41, 59)       # Dark comic outline
_BUBBLE_FG = (15, 23, 42)           # Near black text


def _get_scene_theme(stage_category: str, index: int, total: int) -> dict:
    """Returns soft psychological stage colors for panel headers."""
    cat = (stage_category or "").lower()
    if any(k in cat for k in ["problem", "trigger", "incident"]):
        return {"bg": (254, 226, 226), "fg": (153, 27, 27)}      # Soft Crimson
    elif any(k in cat for k in ["spiral", "disorientation"]):
        return {"bg": (254, 243, 199), "fg": (180, 83, 9)}       # Soft Amber
    elif any(k in cat for k in ["examination", "validation"]):
        return {"bg": (224, 231, 255), "fg": (55, 48, 163)}      # Soft Indigo
    elif any(k in cat for k in ["reframing", "externalizing"]):
        return {"bg": (224, 242, 254), "fg": (7, 89, 133)}       # Soft Sky Blue
    elif any(k in cat for k in ["boundary", "action", "support", "resolution"]):
        return {"bg": (220, 252, 231), "fg": (22, 101, 52)}      # Soft Emerald

    # Default 3-color palette
    palettes = [
        {"bg": (254, 226, 226), "fg": (153, 27, 27)},
        {"bg": (224, 242, 254), "fg": (7, 89, 133)},
        {"bg": (220, 252, 231), "fg": (22, 101, 52)},
    ]
    return palettes[index % len(palettes)]


def _calculate_grid(n: int) -> tuple:
    """Calculates optimal (cols, rows) for n panels (3 <= n <= 8)."""
    if n <= 3:
        return 3, 1
    elif n == 4:
        return 2, 2
    elif n <= 6:
        return 3, 2
    else:
        return 4, 2


def _load_font(size: int = 12, bold: bool = False, comic: bool = False) -> ImageFont.FreeTypeFont:
    """Attempts to load preferred system fonts with robust fallbacks."""
    if comic:
        candidates = ["comicbd.ttf", "comic.ttf"] if bold else ["comic.ttf", "segoeui.ttf", "arial.ttf"]
    elif bold:
        candidates = ["segoeuib.ttf", "arialbd.ttf", "comicbd.ttf"]
    else:
        candidates = ["segoeui.ttf", "arial.ttf", "comic.ttf"]

    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_thought_bubble(draw: ImageDraw.ImageDraw, box: tuple, text: str, font: ImageFont.ImageFont):
    """
    Draws a comic thought bubble with rounded cloud-like body and
    descending trailing thought circles.
    """
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0

    # Main bubble body (rounded rectangle with thick border)
    radius = min(14, h // 3)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=_BUBBLE_BG, outline=_BUBBLE_BORDER, width=2)

    # Trailing thought dots pointing towards the lower-middle
    cx = x0 + int(w * 0.55)
    dots = [
        (cx, y1 + 5, 4),
        (cx - 5, y1 + 13, 3),
        (cx - 10, y1 + 20, 2),
    ]
    for dx, dy, r in dots:
        draw.ellipse([dx - r, dy - r, dx + r, dy + r], fill=_BUBBLE_BG, outline=_BUBBLE_BORDER, width=2)

    _draw_centered_text(draw, text, font, (x0 + 6, y0 + 4, x1 - 6, y1 - 4), fill=_BUBBLE_FG)


def _draw_speech_bubble(draw: ImageDraw.ImageDraw, box: tuple, text: str, font: ImageFont.ImageFont):
    """
    Draws a comic speech bubble with a triangular tail pointing to the character.
    """
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0

    radius = min(10, h // 3)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=_BUBBLE_BG, outline=_BUBBLE_BORDER, width=2)

    tx = x0 + int(w * 0.5)
    tail_pts = [(tx - 8, y1), (tx + 8, y1), (tx - 3, y1 + 14)]
    draw.polygon(tail_pts, fill=_BUBBLE_BG)
    draw.line([tail_pts[0], tail_pts[2]], fill=_BUBBLE_BORDER, width=2)
    draw.line([tail_pts[1], tail_pts[2]], fill=_BUBBLE_BORDER, width=2)

    _draw_centered_text(draw, text, font, (x0 + 6, y0 + 4, x1 - 6, y1 - 4), fill=_BUBBLE_FG)


def _draw_centered_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, box: tuple, fill):
    """Wraps text and vertically + horizontally centers it within box with dynamic font scaling."""
    x0, y0, x1, y1 = box
    max_w = x1 - x0
    max_h = y1 - y0

    init_size = getattr(font, "size", 11)
    best_font = font
    best_lines = []

    # Iteratively scale font size down until all lines fit within the box height
    for sz in range(init_size, 7, -1):
        test_font = _load_font(size=sz, bold=True, comic=True)
        avg_char_w = test_font.getlength("M") / 1.5 if hasattr(test_font, "getlength") else 6
        wrap_width = max(10, int(max_w / avg_char_w))
        lines = []
        for paragraph in (text or "").split("\n"):
            lines.extend(textwrap.wrap(paragraph, width=wrap_width) or [""])
        line_h = sz + 3
        if len(lines) * line_h <= max_h + 2:
            best_font = test_font
            best_lines = lines
            break
        best_font = test_font
        best_lines = lines

    line_h = getattr(best_font, "size", 9) + 3
    total_text_h = len(best_lines) * line_h
    start_y = max(y0, y0 + (max_h - total_text_h) // 2)

    for i, line in enumerate(best_lines):
        line_y = start_y + i * line_h
        line_w = best_font.getlength(line) if hasattr(best_font, "getlength") else len(line) * 6
        line_x = max(x0, x0 + (max_w - line_w) // 2)
        draw.text((line_x, line_y), line, font=best_font, fill=fill)


def _draw_caption_box(draw: ImageDraw.ImageDraw, box: tuple, label: str, text: str,
                      label_font: ImageFont.ImageFont, text_font: ImageFont.ImageFont):
    """Draws comic narrative caption box at the bottom of a panel with dynamic font auto-scaling."""
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], fill=_CAPTION_BG, outline=_PANEL_BORDER, width=2)

    draw.text((x0 + 8, y0 + 5), label.upper(), font=label_font, fill=(100, 116, 139))

    content_box = (x0 + 8, y0 + 20, x1 - 8, y1 - 4)
    avail_w = content_box[2] - content_box[0]
    avail_h = content_box[3] - content_box[1]

    init_size = getattr(text_font, "size", 11)
    best_font = text_font
    best_lines = []

    # Iteratively scale font down until caption text fits
    for sz in range(init_size, 7, -1):
        test_font = _load_font(size=sz, bold=False, comic=False)
        avg_w = test_font.getlength("n") if hasattr(test_font, "getlength") else 5.5
        wrap_chars = max(12, int(avail_w / avg_w))
        lines = []
        for p in (text or "").split("\n"):
            lines.extend(textwrap.wrap(p, width=wrap_chars) or [""])
        lh = sz + 3
        if len(lines) * lh <= avail_h + 2:
            best_font = test_font
            best_lines = lines
            break
        best_font = test_font
        best_lines = lines

    lh = getattr(best_font, "size", 9) + 3
    cur_y = content_box[1]
    for line in best_lines:
        if cur_y + lh > content_box[3] + 6:
            break
        draw.text((content_box[0], cur_y), line, font=best_font, fill=_CAPTION_FG)
        cur_y += lh


def compose_comic_strip(panel_paths: list, scenes: list, out_path: str, title: str = "REFRAME COMIC") -> str:
    """
    Composites 3 to 8 scene images into an adaptive comic strip layout.

    Args:
        panel_paths: list of image file paths generated by SD-Turbo
        scenes: list of scene dictionaries
        out_path: destination file path for composite image
        title: comic banner title
    """
    num_panels = max(3, min(len(scenes) if scenes else len(panel_paths), 8))
    cols, rows = _calculate_grid(num_panels)

    panel_block_h = _HEADER_H + _PANEL_H + 6 + _CAPTION_H
    total_w = _MARGIN * 2 + _PANEL_W * cols + _GAP * (cols - 1)
    total_h = _MARGIN + _TOP_BANNER_H + 12 + rows * panel_block_h + (rows - 1) * _GAP + _MARGIN

    canvas = Image.new("RGB", (total_w, total_h), _CANVAS_BG)
    draw = ImageDraw.Draw(canvas)

    # Fonts
    banner_title_font = _load_font(size=16, bold=True, comic=True)
    banner_sub_font = _load_font(size=11, bold=False, comic=False)
    header_font = _load_font(size=12, bold=True, comic=True)
    bubble_font = _load_font(size=11, bold=True, comic=True)
    caption_label_font = _load_font(size=9, bold=True, comic=False)
    caption_text_font = _load_font(size=11, bold=False, comic=False)

    # 1. Top Comic Title Banner
    banner_x0 = _MARGIN
    banner_y0 = _MARGIN
    banner_w = total_w - _MARGIN * 2
    draw.rectangle([banner_x0, banner_y0, banner_x0 + banner_w, banner_y0 + _TOP_BANNER_H],
                   fill=(241, 245, 249), outline=_PANEL_BORDER, width=2)

    subtitle = "TRIGGER  ->  REFRAMING  ->  RESOLUTION" if num_panels == 3 else f"ADAPTIVE PSYCHOLOGICAL NARRATIVE ({num_panels} PANELS)"
    sub_w = banner_sub_font.getlength(subtitle) if hasattr(banner_sub_font, "getlength") else len(subtitle) * 7
    title_w = banner_title_font.getlength(title) if hasattr(banner_title_font, "getlength") else len(title) * 9

    if title_w + sub_w + 40 > banner_w:
        # Two-tier layout to prevent text overlap
        title_font_compact = _load_font(size=13, bold=True, comic=True)
        draw.text((banner_x0 + 14, banner_y0 + 7), title, font=title_font_compact, fill=(15, 23, 42))
        draw.text((banner_x0 + 14, banner_y0 + 28), subtitle, font=banner_sub_font, fill=(100, 116, 139))
    else:
        draw.text((banner_x0 + 14, banner_y0 + 14), title, font=banner_title_font, fill=(15, 23, 42))
        draw.text((banner_x0 + banner_w - sub_w - 16, banner_y0 + 16), subtitle,
                  font=banner_sub_font, fill=(100, 116, 139))

    # 2. Render each Comic Panel in Adaptive Grid
    for i in range(num_panels):
        scene = scenes[i] if i < len(scenes) else {}
        stage_category = scene.get("stage_category") or scene.get("stage", f"scene_{i+1}")
        theme = _get_scene_theme(stage_category, i, num_panels)

        col = i % cols
        row = i // cols

        cur_x = _MARGIN + col * (_PANEL_W + _GAP)
        y_header = _MARGIN + _TOP_BANNER_H + 12 + row * (panel_block_h + _GAP)
        y_art = y_header + _HEADER_H
        y_caption = y_art + _PANEL_H + 6

        # --- A. Panel Header Banner ---
        panel_title = scene.get("title", f"{i+1}. {stage_category.upper()}").upper()
        draw.rectangle([cur_x, y_header, cur_x + _PANEL_W, y_header + _HEADER_H],
                       fill=theme["bg"], outline=_PANEL_BORDER, width=2)
        draw.text((cur_x + 10, y_header + 7), panel_title, font=header_font, fill=theme["fg"])

        # --- B. Panel Image Artwork with Speech/Thought Bubble Overlay ---
        if i < len(panel_paths) and os.path.exists(panel_paths[i]):
            panel_img = Image.open(panel_paths[i]).convert("RGBA").resize((_PANEL_W, _PANEL_H), Image.Resampling.LANCZOS)
        else:
            panel_img = Image.new("RGBA", (_PANEL_W, _PANEL_H), (226, 232, 240))

        bubble_draw = ImageDraw.Draw(panel_img)
        dialogue = scene.get("dialogue", "").strip()
        bubble_type = scene.get("bubble_type", "thought").lower()

        if dialogue:
            b_h = 66 if len(dialogue) <= 45 else (76 if len(dialogue) <= 85 else 86)
            bx0, by0, bx1, by1 = 10, 8, _PANEL_W - 10, 8 + b_h
            if "thought" in bubble_type:
                _draw_thought_bubble(bubble_draw, (bx0, by0, bx1, by1), dialogue, bubble_font)
            else:
                _draw_speech_bubble(bubble_draw, (bx0, by0, bx1, by1), dialogue, bubble_font)

        canvas.paste(panel_img.convert("RGB"), (cur_x, y_art))
        draw.rectangle([cur_x, y_art, cur_x + _PANEL_W, y_art + _PANEL_H], outline=_PANEL_BORDER, width=2)

        # --- C. Narrative Caption Box Below Artwork ---
        caption_text = scene.get("caption") or scene.get("narrative", "")
        _draw_caption_box(
            draw,
            (cur_x, y_caption, cur_x + _PANEL_W, y_caption + _CAPTION_H),
            label=f"SCENE {i+1} INSIGHT",
            text=caption_text,
            label_font=caption_label_font,
            text_font=caption_text_font,
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


# Backwards compatibility alias
compose_storyboard = compose_comic_strip

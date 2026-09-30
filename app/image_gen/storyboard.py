"""
Comic Strip Composition Engine.

Composes three generated panel images into an authentic 3-panel comic strip:
  SCENE 1: PROBLEM  ->  SCENE 2: REFRAMING  ->  SCENE 3: RESOLUTION

Features:
- Solid dark comic panel borders and clean gutters.
- Styled scene headers with color-coded psychological stage themes.
- Genuine speech bubbles (with directional pointer tail) and thought bubbles
  (with trailing circular thought dots).
- Readable comic typography with automatic text wrapping and centering.
- Grounded narrative caption boxes at the bottom of each panel.
- Overall comic title banner.
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
_TOP_BANNER_H = 42

# Color palettes per scene: (header_bg, header_fg, border_color)
_SCENE_THEMES = [
    {"bg": (254, 226, 226), "fg": (153, 27, 27), "title": "1. PROBLEM"},      # Soft Crimson
    {"bg": (224, 242, 254), "fg": (7, 89, 133), "title": "2. REFRAMING"},     # Soft Sky Blue
    {"bg": (220, 252, 231), "fg": (22, 101, 52), "title": "3. RESOLUTION"},   # Soft Emerald
]

_CANVAS_BG = (248, 250, 252)        # Slate-50 background
_PANEL_BORDER = (30, 41, 59)        # Slate-800 crisp border
_CAPTION_BG = (255, 255, 255)       # White caption box
_CAPTION_FG = (51, 65, 85)          # Slate-700 caption text
_BUBBLE_BG = (255, 255, 255)        # Pure white bubble
_BUBBLE_BORDER = (30, 41, 59)       # Dark comic outline
_BUBBLE_FG = (15, 23, 42)           # Near black text


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

    # Wrap and center text inside bubble
    _draw_centered_text(draw, text, font, (x0 + 6, y0 + 4, x1 - 6, y1 - 4), fill=_BUBBLE_FG)


def _draw_speech_bubble(draw: ImageDraw.ImageDraw, box: tuple, text: str, font: ImageFont.ImageFont):
    """
    Draws a comic speech bubble with a triangular tail pointing to the character.
    """
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0

    # Main bubble body
    radius = min(10, h // 3)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=_BUBBLE_BG, outline=_BUBBLE_BORDER, width=2)

    # Triangular pointer tail on bottom edge
    tx = x0 + int(w * 0.5)
    tail_pts = [(tx - 8, y1), (tx + 8, y1), (tx - 3, y1 + 14)]
    # Fill tail with white
    draw.polygon(tail_pts, fill=_BUBBLE_BG)
    # Outline the two extending edges of the tail
    draw.line([tail_pts[0], tail_pts[2]], fill=_BUBBLE_BORDER, width=2)
    draw.line([tail_pts[1], tail_pts[2]], fill=_BUBBLE_BORDER, width=2)

    # Wrap and center text inside bubble
    _draw_centered_text(draw, text, font, (x0 + 6, y0 + 4, x1 - 6, y1 - 4), fill=_BUBBLE_FG)


def _draw_centered_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, box: tuple, fill):
    """Wraps text and vertically + horizontally centers it within box."""
    x0, y0, x1, y1 = box
    max_w = x1 - x0
    avg_char_w = font.getlength("M") / 1.5 if hasattr(font, "getlength") else 7
    wrap_width = max(10, int(max_w / avg_char_w))

    lines = []
    for paragraph in (text or "").split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=wrap_width) or [""])

    line_h = getattr(font, "size", 12) + 3
    total_text_h = len(lines) * line_h
    start_y = max(y0, y0 + (y1 - y0 - total_text_h) // 2)

    for i, line in enumerate(lines):
        line_y = start_y + i * line_h
        if line_y + line_h > y1 + 5:
            break
        line_w = font.getlength(line) if hasattr(font, "getlength") else len(line) * avg_char_w
        line_x = max(x0, x0 + (max_w - line_w) // 2)
        draw.text((line_x, line_y), line, font=font, fill=fill)


def _draw_caption_box(draw: ImageDraw.ImageDraw, box: tuple, label: str, text: str,
                      label_font: ImageFont.ImageFont, text_font: ImageFont.ImageFont):
    """Draws comic narrative caption box at the bottom of a panel."""
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], fill=_CAPTION_BG, outline=_PANEL_BORDER, width=2)

    # Small stage label
    draw.text((x0 + 8, y0 + 5), label.upper(), font=label_font, fill=(100, 116, 139))

    # Narrative body text
    content_box = (x0 + 8, y0 + 20, x1 - 8, y1 - 4)
    avg_w = text_font.getlength("n") if hasattr(text_font, "getlength") else 6
    wrap_chars = max(12, int((content_box[2] - content_box[0]) / avg_w))

    lines = []
    for p in (text or "").split("\n"):
        lines.extend(textwrap.wrap(p, width=wrap_chars) or [""])

    cur_y = content_box[1]
    line_spacing = 3
    lh = getattr(text_font, "size", 11) + line_spacing
    for line in lines:
        if cur_y + lh > content_box[3]:
            break
        draw.text((content_box[0], cur_y), line, font=text_font, fill=_CAPTION_FG)
        cur_y += lh


def compose_comic_strip(panel_paths: list, scenes: list, out_path: str, title: str = "REFRAME COMIC") -> str:
    """
    Composites 3 scene images into a single unified 3-panel comic strip.

    Args:
        panel_paths: list of 3 image file paths generated by SD-Turbo
        scenes: list of 3 scene dictionaries containing:
                - title, narrative, dialogue, bubble_type
        out_path: destination file path for composite image
        title: comic banner title
    """
    total_w = _MARGIN * 2 + _PANEL_W * 3 + _GAP * 2
    total_h = _MARGIN + _TOP_BANNER_H + _HEADER_H + _PANEL_H + 8 + _CAPTION_H + _MARGIN

    canvas = Image.new("RGB", (total_w, total_h), _CANVAS_BG)
    draw = ImageDraw.Draw(canvas)

    # Fonts
    banner_title_font = _load_font(size=17, bold=True, comic=True)
    banner_sub_font = _load_font(size=12, bold=False, comic=False)
    header_font = _load_font(size=13, bold=True, comic=True)
    bubble_font = _load_font(size=11, bold=True, comic=True)
    caption_label_font = _load_font(size=9, bold=True, comic=False)
    caption_text_font = _load_font(size=11, bold=False, comic=False)

    # 1. Top Comic Title Banner
    banner_x0 = _MARGIN
    banner_y0 = _MARGIN
    banner_w = total_w - _MARGIN * 2
    draw.rectangle([banner_x0, banner_y0, banner_x0 + banner_w, banner_y0 + _TOP_BANNER_H],
                   fill=(241, 245, 249), outline=_PANEL_BORDER, width=2)
    draw.text((banner_x0 + 14, banner_y0 + 10), title, font=banner_title_font, fill=(15, 23, 42))
    draw.text((banner_x0 + banner_w - 260, banner_y0 + 13), "PROBLEM  ->  REFRAMING  ->  RESOLUTION",
              font=banner_sub_font, fill=(100, 116, 139))

    # 2. Render each of the 3 Comic Panels
    cur_x = _MARGIN
    y_header = _MARGIN + _TOP_BANNER_H + 10
    y_art = y_header + _HEADER_H
    y_caption = y_art + _PANEL_H + 6

    for i in range(3):
        theme = _SCENE_THEMES[i]
        scene = scenes[i] if i < len(scenes) else {}

        # --- A. Panel Header Banner ---
        panel_title = scene.get("title", theme["title"]).upper()
        draw.rectangle([cur_x, y_header, cur_x + _PANEL_W, y_header + _HEADER_H],
                       fill=theme["bg"], outline=_PANEL_BORDER, width=2)
        draw.text((cur_x + 10, y_header + 7), panel_title, font=header_font, fill=theme["fg"])

        # --- B. Panel Image Artwork with Speech/Thought Bubble Overlay ---
        if i < len(panel_paths) and os.path.exists(panel_paths[i]):
            panel_img = Image.open(panel_paths[i]).convert("RGBA").resize((_PANEL_W, _PANEL_H))
        else:
            # Placeholder canvas if image missing
            panel_img = Image.new("RGBA", (_PANEL_W, _PANEL_H), (226, 232, 240))

        # Overlay speech or thought bubble onto the artwork
        bubble_draw = ImageDraw.Draw(panel_img)
        dialogue = scene.get("dialogue", "").strip()
        bubble_type = scene.get("bubble_type", "thought").lower()

        if dialogue:
            # Bubble bounding box in upper portion of the panel
            bx0, by0, bx1, by1 = 12, 10, _PANEL_W - 12, 70
            if "thought" in bubble_type:
                _draw_thought_bubble(bubble_draw, (bx0, by0, bx1, by1), dialogue, bubble_font)
            else:
                _draw_speech_bubble(bubble_draw, (bx0, by0, bx1, by1), dialogue, bubble_font)

        # Paste artwork into canvas
        canvas.paste(panel_img.convert("RGB"), (cur_x, y_art))
        # Draw outer panel border around artwork
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

        cur_x += _PANEL_W + _GAP

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


# Backwards compatibility alias
compose_storyboard = compose_comic_strip

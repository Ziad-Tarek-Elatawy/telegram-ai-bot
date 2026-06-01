"""
Image Editor Service
Handles printing Arabic text over generated images.
"""

import logging
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

from config import (
    FONTS_DIR,
    DEFAULT_ARABIC_FONT,
    FONT_SIZE_DEFAULT,
    FONT_COLOR,
    FONT_OUTLINE_COLOR,
    FONT_OUTLINE_WIDTH,
)

logger = logging.getLogger("ai_image_bot.image_editor")

def _get_font(font_name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Retrieve font, falling back to default if not found."""
    font_path = FONTS_DIR / font_name
    if not font_path.exists():
        logger.warning(f"Font {font_name} not found at {font_path}, attempting fallback.")
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default()
    
    return ImageFont.truetype(str(font_path), size)


def add_arabic_text_to_image(image_path: Path, text: str, save_path: Path | None = None) -> Path:
    """
    Overlays Arabic text onto an image.
    Handles right-to-left (RTL) formatting and letter reshaping.
    """
    if not text.strip():
        return image_path

    logger.info(f"Adding text to {image_path.name}")
    
    # 1. Reshape Arabic letters (connect them properly)
    reshaped_text = arabic_reshaper.reshape(text)
    
    # 2. Fix RTL direction
    bidi_text = get_display(reshaped_text)
    
    # 3. Open Image
    try:
        with Image.open(image_path) as img:
            # Convert to RGBA for drawing if needed
            img = img.convert("RGBA")
            
            # Create Draw object
            draw = ImageDraw.Draw(img)
            
            # Load Font
            font = _get_font(DEFAULT_ARABIC_FONT, FONT_SIZE_DEFAULT)
            
            # Calculate text size and position (bottom center)
            # Use textbbox (textsize is deprecated in newer Pillow versions)
            bbox = draw.textbbox((0, 0), bidi_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            width, height = img.size
            x = (width - text_width) / 2
            y = height - text_height - 60  # 60 pixels padding from bottom
            
            # Draw Outline/Stroke (thicker for better visibility on all backgrounds)
            outline_range = FONT_OUTLINE_WIDTH
            for adj_x in range(-outline_range, outline_range + 1):
                for adj_y in range(-outline_range, outline_range + 1):
                    if adj_x == 0 and adj_y == 0:
                        continue
                    draw.text(
                        (x + adj_x, y + adj_y), 
                        bidi_text, 
                        font=font, 
                        fill=FONT_OUTLINE_COLOR
                    )
            
            # Draw Main Text
            draw.text((x, y), bidi_text, font=font, fill=FONT_COLOR)
            
            # Convert back to RGB for saving (JPEG doesn't support RGBA)
            img = img.convert("RGB")
            
            # Save Image
            if save_path is None:
                save_path = image_path.parent / f"text_{image_path.name}"
                
            img.save(save_path)
            logger.info(f"Image with text saved to {save_path}")
            return save_path
            
    except Exception as e:
        logger.error(f"Error adding text to image: {e}")
        return image_path

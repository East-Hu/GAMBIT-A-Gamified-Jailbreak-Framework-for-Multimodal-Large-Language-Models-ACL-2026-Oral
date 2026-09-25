"""Image helpers shared across the attack runners."""

import base64
import io

from PIL import Image


def pil_image_to_base64(image, image_format="PNG"):
    """Encode a PIL image (or a path to one) as a base64 string.

    Parameters
    ----------
    image : PIL.Image.Image or str
        Either an in-memory PIL image or a path to an image file.
    image_format : str
        Encoding format expected by the target endpoint ("PNG" or "JPEG").
    """
    if isinstance(image, str):
        image = Image.open(image)
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def as_data_url(image, image_format="PNG"):
    """Return an OpenAI-style ``data:`` URL for a PIL image or path."""
    b64 = pil_image_to_base64(image, image_format=image_format)
    mime = "jpeg" if image_format.upper() in ("JPEG", "JPG") else image_format.lower()
    return f"data:image/{mime};base64,{b64}"

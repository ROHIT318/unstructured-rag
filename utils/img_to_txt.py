import os
import io

from dotenv import load_dotenv
import pytesseract
from PIL import Image

from google import genai
from google.genai import types

load_dotenv()

# The tesseract executable ships in the repo's tesseract/ subfolder, not on
# PATH. Resolve it against this file, not the working directory — the app may
# be launched from anywhere, and a relative path here makes pytesseract fail
# silently (OCR comes back empty) from any other cwd.
pytesseract.pytesseract.tesseract_cmd = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tesseract", "tesseract.exe")
)

GEMINI_EMBEDDING_API_KEY = os.getenv("GEMINI_EMBEDDING_API_KEY")
GEMINI_CAPTION_MODEL = os.getenv("GEMINI_CAPTION_MODEL")
BASE_URL = os.getenv("BASE_URL")

client_kwargs = {"api_key": GEMINI_EMBEDDING_API_KEY}
if BASE_URL:
    client_kwargs["http_options"] = {"base_url": BASE_URL}
client = genai.Client(**client_kwargs)

SEMANTIC_FALLBACK = "Semantic meaning unavailable"


def ocr_image(image_bytes: bytes) -> str:
    """
    Read the text visible in the image with pytesseract.

    Failures are not fatal to ingestion — an unreadable image yields an empty
    string (photos, charts and logos legitimately contain no text).
    """
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes))).strip()
    except Exception:
        return ""


def describe_image(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """
    Ask the genai client (GEMINI_CAPTION_MODEL) for a short description of what
    the image shows. A failure yields the fallback string, never an exception —
    ingestion continues either way.
    """
    try:
        response = client.models.generate_content(
            model=GEMINI_CAPTION_MODEL,
            contents=[
                "Describe the image in detail. Include objects, people, charts or text.",
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
        return (response.text or "").strip() or SEMANTIC_FALLBACK
    except Exception:
        return SEMANTIC_FALLBACK


def extract_text_from_image(image_bytes: bytes, mime_type: str = "image/png") -> tuple:
    """
    Extract both texts of an image: the OCR text actually present in it and the
    genai-generated semantic description. Both underlying functions take the
    image bytes, so the pipeline calls this one step uniformly per image.

    Returns:
        (ocr_text, semantic_meaning): ocr_text is "" when the image has no
        readable text; semantic_meaning is the fallback string when the
        description call fails.
    """
    return ocr_image(image_bytes), describe_image(image_bytes, mime_type)


if __name__ == "__main__":
    image_file = f"{os.getcwd()}/data/images/resume_rohit_sharma.jpg"
    with open(image_file, "rb") as f:
        print(extract_text_from_image(f.read()))

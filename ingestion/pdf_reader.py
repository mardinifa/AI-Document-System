import pymupdf
import pytesseract
from PIL import Image
import io


def extract_text(pdf_path: str) -> list[dict]:
    """Opens a PDF and returns the text on every page."""
    doc = pymupdf.open(pdf_path)

    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()

        if not text:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)

        pages.append({
            "page_number": i + 1,
            "text": text
        })

    return pages


if __name__ == "__main__":
    result = extract_text("data/benchmarks/rfq_sample.pdf")

    print(f"Extracted {len(result)} pages.")
    print(result[0]["text"][:500])
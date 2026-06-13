import fitz  # this is pymupdf
import pytesseract
from PIL import Image
import io

# Open the image-only PDF and render its page to a picture
doc = fitz.open("data/raw/Sequence-rules-of-the-game-EN.pdf")
page = doc[0]
pix = page.get_pixmap(dpi=200)              # turn the page into an image
img = Image.open(io.BytesIO(pix.tobytes("png")))

# Read text out of the image
text = pytesseract.image_to_string(img)
print(f"OCR extracted {len(text)} characters:\n")
print(text)
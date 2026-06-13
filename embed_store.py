# embed_store.py — turn chunks into vectors and store them in a searchable database
import fitz
import pytesseract
from PIL import Image
import io
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nebius import NebiusEmbeddings
from langchain_chroma import Chroma
import os, shutil
# Wipe the old database first, so re-running doesn't pile up duplicates
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

load_dotenv()  # reads your .env so the Nebius key is available

DATA_DIR = Path("data/raw")

def game_name(filename):
    name = filename.lower()
    if "catan" in name: return "catan"
    if "saboteur" in name: return "saboteur"
    if "evolution" in name: return "evolution"
    if "sequence" in name: return "sequence"
    if "spot" in name or "dobble" in name: return "spot-it"
    return "unknown"

# Load pages and give each a CLEAN set of tags (only simple values, or Chroma complains)
all_pages = []
def ocr_page(pdf_path, page_number):
    """Read text out of an image-only page using OCR."""
    doc = fitz.open(str(pdf_path))
    pix = doc[page_number].get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)

for pdf_path in DATA_DIR.glob("*.pdf"):
    pages = PyMuPDFLoader(str(pdf_path)).load()
    for page in pages:
        # If a page has almost no text, it's probably an image — OCR it instead
        if len(page.page_content.strip()) < 50:
            page.page_content = ocr_page(pdf_path, page.metadata.get("page", 0))
            print(f"  OCR used for {pdf_path.name} page {page.metadata.get('page', 0)}")
        page.metadata = {
            "game": game_name(pdf_path.name),
            "page": page.metadata.get("page", 0),
            "source": pdf_path.name,
        }
    all_pages.extend(pages)

chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(all_pages)
print(f"Prepared {len(chunks)} chunks")

# Turn each chunk into a vector using Nebius, and save them all into a local database
embeddings = NebiusEmbeddings(model="Qwen/Qwen3-Embedding-8B")  # uses the key from your .env
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",  # saves to this folder so you don't re-embed every run
)
print("Stored! Vectors saved in the chroma_db folder.\n")

# Quick test: ask a real question and see which chunks come back
results = vectorstore.similarity_search("How do saboteurs win?", k=3)
for i, doc in enumerate(results, 1):
    print(f"--- result {i} | {doc.metadata['game']} | page {doc.metadata['page']} ---")
    print(doc.page_content[:200], "\n")
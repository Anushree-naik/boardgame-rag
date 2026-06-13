# ingest.py — loads the PDFs and splits them into chunks

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Where your 5 PDFs live
DATA_DIR = Path("data/raw")

# Turn a filename into a clean game name (this becomes a "tag" on each chunk)
def game_name(filename):
    name = filename.lower()
    if "catan" in name: return "catan"
    if "saboteur" in name: return "saboteur"
    if "evolution" in name: return "evolution"
    if "sequence" in name: return "sequence"
    if "spot" in name or "dobble" in name: return "spot-it"
    return "unknown"

# Load every PDF, one document per page, tagging which game each page is from
all_pages = []
for pdf_path in DATA_DIR.glob("*.pdf"):
    pages = PyMuPDFLoader(str(pdf_path)).load()
    for page in pages:
        page.metadata["game"] = game_name(pdf_path.name)
    all_pages.extend(pages)
    print(f"Loaded {len(pages):>3} pages from {pdf_path.name}  ->  {game_name(pdf_path.name)}")

print(f"\nTotal pages: {len(all_pages)}")

# Cut the pages into smaller chunks (800 characters each, slightly overlapping)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(all_pages)

print(f"Total chunks: {len(chunks)}\n")
print("----- example chunk -----")
print("game:", chunks[0].metadata.get("game"), "| page:", chunks[0].metadata.get("page"))
print(chunks[0].page_content[:300])
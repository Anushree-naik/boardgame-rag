from langchain_community.document_loaders import PyMuPDFLoader

pages = PyMuPDFLoader("data/raw/Sequence-rules-of-the-game-EN.pdf").load()
print("Pages loaded:", len(pages))
for i, p in enumerate(pages):
    print(f"--- page {i} | {len(p.page_content)} characters of text ---")
    print(repr(p.page_content[:500]))
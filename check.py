from dotenv import load_dotenv
from langchain_nebius import NebiusEmbeddings
from langchain_chroma import Chroma
load_dotenv()

embeddings = NebiusEmbeddings(model="Qwen/Qwen3-Embedding-8B")
vs = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

q = "how many sequences do I need to win"
chunks = vs.similarity_search(q, k=4, filter={"game": "sequence"})
for i, c in enumerate(chunks, 1):
    print(f"--- chunk {i} (page {c.metadata['page']}) ---")
    print(c.page_content)
    print()
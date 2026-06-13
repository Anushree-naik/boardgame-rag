from dotenv import load_dotenv
from langchain_nebius import NebiusEmbeddings
from langchain_chroma import Chroma
from collections import Counter
load_dotenv()

embeddings = NebiusEmbeddings(model="Qwen/Qwen3-Embedding-8B")
vs = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

data = vs.get()                      # pull everything in the database
games = [m["game"] for m in data["metadatas"]]
print("Total chunks in DB:", len(games))
print("Chunks per game:", Counter(games))
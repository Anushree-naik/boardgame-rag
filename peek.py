from dotenv import load_dotenv
from langchain_nebius import NebiusEmbeddings
load_dotenv()

emb = NebiusEmbeddings(model="Qwen/Qwen3-Embedding-8B")
vector = emb.embed_query("Can the robber go on the desert?")
print("This one question became a list of", len(vector), "numbers.")
print("First 10 of them:", vector[:10])
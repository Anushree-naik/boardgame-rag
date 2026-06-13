# evaluate.py — run all 15 golden questions, score retrieval, show answers to judge

from collections import Counter
from dotenv import load_dotenv
from langchain_nebius import NebiusEmbeddings, ChatNebius
from langchain_chroma import Chroma
from golden_set import GOLDEN_SET

load_dotenv()

embeddings = NebiusEmbeddings(model="Qwen/Qwen3-Embedding-8B")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
llm = ChatNebius(model="openai/gpt-oss-120b")   # whichever chat model you settled on

GAME_ALIASES = {
    "catan": ["catan"], "saboteur": ["saboteur"], "evolution": ["evolution"],
    "sequence": ["sequence"], "spot-it": ["spot it", "spot-it", "dobble"],
}

def games_named_in(text):
    t = text.lower()
    return [g for g, a in GAME_ALIASES.items() if any(x in t for x in a)]

def generate(question, chunks, note=""):
    context = "\n\n".join(
        f"[{c.metadata['game']}, page {c.metadata['page']}]\n{c.page_content}"
        for c in chunks
    )
    prompt = f"""You are a board game rules assistant. Answer using ONLY the rulebook excerpts below.
If the answer is not in the excerpts, say you don't know. Keep it short, and say which game it's about.

Rulebook excerpts:
{context}

Question: {question}
Answer:"""
    return (note + " " if note else "") + llm.invoke(prompt).content

def run_bot(question):
    """Returns (bot_behavior, answer_text, games_of_retrieved_chunks)."""
    named = games_named_in(question)

    if len(named) == 1:
        chunks = vectorstore.similarity_search(question, k=4, filter={"game": named[0]})
        return "ANSWER", generate(question, chunks), [c.metadata["game"] for c in chunks]

    if len(named) == 0:
        chunks = vectorstore.similarity_search(question, k=4)
        retrieved_games = [c.metadata["game"] for c in chunks]
        counts = Counter(retrieved_games)
        top_game, top_count = counts.most_common(1)[0]
        if top_count > len(chunks) / 2:
            return "ASSUME", generate(question, chunks, note=f"(Assuming {top_game}.)"), retrieved_games
        else:
            return "ASK", f"Which game? Found rules in: {', '.join(sorted(counts))}", retrieved_games

    return "ASK", f"You named several games: {', '.join(named)} — which one?", []

# ---- run all 15 ----
retrieval_hits = 0
retrieval_total = 0

for i, row in enumerate(GOLDEN_SET, 1):
    bot_behavior, answer_text, retrieved_games = run_bot(row["q"])

    # Retrieval score: only meaningful when we expect a specific game
    retrieval_mark = "n/a"
    if row["game"] is not None:
        retrieval_total += 1
        hit = row["game"] in retrieved_games
        retrieval_hits += hit
        retrieval_mark = "HIT " if hit else "MISS"

    behavior_match = "OK " if bot_behavior == row["behavior"] else "XX "

    print("=" * 70)
    print(f"Q{i}: {row['q']}")
    print(f"  expected behavior: {row['behavior']:<7}  | bot did: {bot_behavior:<7} [{behavior_match}]")
    print(f"  retrieval (expected {row['game']}): [{retrieval_mark}]")
    print(f"  expected answer: {row['expected']}")
    print(f"  BOT ANSWER: {answer_text}")

print("=" * 70)
print(f"\nRETRIEVAL SCORE (recall@4): {retrieval_hits}/{retrieval_total}")
print("Generation: read each BOT ANSWER above and mark pass/fail by hand.")
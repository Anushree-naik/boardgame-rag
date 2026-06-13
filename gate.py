# gate.py — decide whether to answer or ask "which game?" before responding

from collections import Counter
from dotenv import load_dotenv
from langchain_nebius import NebiusEmbeddings, ChatNebius
from langchain_chroma import Chroma

load_dotenv()

embeddings = NebiusEmbeddings(model="Qwen/Qwen3-Embedding-8B")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
llm = ChatNebius(model="openai/gpt-oss-120b")

# Words that signal each game
GAME_ALIASES = {
    "catan": ["catan"],
    "saboteur": ["saboteur"],
    "evolution": ["evolution"],
    "sequence": ["sequence"],
    "spot-it": ["spot it", "spot-it", "dobble"],
}

def games_named_in(question):
    q = question.lower()
    return [g for g, aliases in GAME_ALIASES.items() if any(a in q for a in aliases)]

def answer(question, chunks, note=""):
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
    result = llm.invoke(prompt).content
    print("A:", (note + " " if note else "") + result)

def handle(question):
    print("\n" + "=" * 60)
    print("Q:", question)

    named = games_named_in(question)

    # Case 1: question names exactly one game -> search only that game's pages
    if len(named) == 1:
        game = named[0]
        chunks = vectorstore.similarity_search(question, k=4, filter={"game": game})
        answer(question, chunks)

    # Case 2: no game named -> retrieve, then see which game the evidence points to
    elif len(named) == 0:
        chunks = vectorstore.similarity_search(question, k=4)
        counts = Counter(c.metadata["game"] for c in chunks)
        top_game, top_count = counts.most_common(1)[0]

        if top_count > len(chunks) / 2:
            # One game is the clear majority -> answer, but state the assumption
            answer(question, chunks, note=f"(Assuming {top_game}.)")
        else:
            # Evidence is split across games -> don't guess, ASK
            options = ", ".join(sorted(counts))
            print(f"A: Which game do you mean? I found relevant rules in: {options}")

    # Case 3: named more than one game -> ask them to pick
    else:
        print("A: You named a few games — which one:", ", ".join(named), "?")

questions = [
    "How many victory points do you need to win Catan?",
    "Can the robber go on the desert?",
    "How do saboteurs win?",
    "How many cards do I start with?",
    "What happens if I knock over the table?",
    "In Spot It, do I want the most cards or the fewest?",
]
for q in questions:
    handle(q)
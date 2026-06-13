# demo.py — interactive board game rules bot

from collections import Counter
from dotenv import load_dotenv
from langchain_nebius import NebiusEmbeddings, ChatNebius
from langchain_chroma import Chroma

load_dotenv()

embeddings = NebiusEmbeddings(model="Qwen/Qwen3-Embedding-8B")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
llm = ChatNebius(model="openai/gpt-oss-120b")   # use whichever chat model you settled on

GAME_ALIASES = {
    "catan": ["catan"],
    "saboteur": ["saboteur"],
    "evolution": ["evolution"],
    "sequence": ["sequence"],
    "spot-it": ["spot it", "spot-it", "dobble"],
}

def games_named_in(text):
    t = text.lower()
    return [g for g, aliases in GAME_ALIASES.items() if any(a in t for a in aliases)]

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
    print("\nBot:", (note + " " if note else "") + llm.invoke(prompt).content)

def respond(question, forced_game=None):
    # If we already know the game (because the user just told us), filter to it
    if forced_game:
        chunks = vectorstore.similarity_search(question, k=4, filter={"game": forced_game})
        answer(question, chunks)
        return

    named = games_named_in(question)

    if len(named) == 1:
        chunks = vectorstore.similarity_search(question, k=4, filter={"game": named[0]})
        answer(question, chunks)

    elif len(named) == 0:
        chunks = vectorstore.similarity_search(question, k=4)
        counts = Counter(c.metadata["game"] for c in chunks)
        top_game, top_count = counts.most_common(1)[0]

        if top_count > len(chunks) / 2:
            answer(question, chunks, note=f"(Assuming {top_game}.)")
        else:
            # Ambiguous: ask which game, then answer the SAME question for that game
            options = sorted(counts)
            print(f"\nBot: Which game do you mean? I found rules in: {', '.join(options)}")
            choice = input("You (pick a game): ").strip().lower()
            picked = games_named_in(choice) or [choice]
            if picked[0] in GAME_ALIASES:
                respond(question, forced_game=picked[0])
            else:
                print("Bot: I don't have that game's rules.")

    else:
        print(f"\nBot: You mentioned a few games — which one: {', '.join(named)}?")
        choice = input("You (pick a game): ").strip().lower()
        picked = games_named_in(choice) or [choice]
        if picked[0] in GAME_ALIASES:
            respond(question, forced_game=picked[0])

# ---- the chat loop ----
print("Board game rules bot. Ask a question, or type 'quit' to exit.")
while True:
    q = input("\nYou: ").strip()
    if q.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break
    respond(q)
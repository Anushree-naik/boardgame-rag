# ask.py — ask a question and get a real answer from your rulebooks

from dotenv import load_dotenv
from langchain_nebius import NebiusEmbeddings, ChatNebius
from langchain_chroma import Chroma

load_dotenv()

# 1. Re-open the database you already built (no re-embedding — it's saved on disk)
embeddings = NebiusEmbeddings(model="Qwen/Qwen3-Embedding-8B")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# 2. The chat model that will write the answer
llm = ChatNebius(model="openai/gpt-oss-120b")

# A list of test questions covering different scenarios
questions = [
    "How many victory points do you need to win Catan?",   # clear fact -> should answer "10"
    "Can the robber go on the desert?",                    # tricky edge case
    "How do saboteurs win?",                               # a different game
    "How many cards do I start with?",                     # ambiguous! true answer differs per game
    "What happens if I knock over the table?",             # not in any rulebook -> should say "I don't know"
    "In Spot It, do I want the most cards or the fewest?", # depends on which mini-game -> should be unsure
]

# Ask each question one at a time
for question in questions:
    chunks = vectorstore.similarity_search(question, k=4)

    context = "\n\n".join(
        f"[{c.metadata['game']}, page {c.metadata['page']}]\n{c.page_content}"
        for c in chunks
    )

    prompt = f"""You are a board game rules assistant. Answer the question using ONLY the rulebook excerpts below.
If the answer is not in the excerpts, say you don't know. Keep it short, and say which game it's about.

Rulebook excerpts:
{context}

Question: {question}
Answer:"""

    answer = llm.invoke(prompt)

    print("\n" + "=" * 60)
    print("Q:", question)
    print("A:", answer.content)
    print("sources:", ", ".join(f"{c.metadata['game']} p{c.metadata['page']}" for c in chunks))
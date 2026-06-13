# app.py — browser chat interface for the board game rules bot

import streamlit as st
import os
from collections import Counter
from dotenv import load_dotenv
from langchain_nebius import NebiusEmbeddings, ChatNebius
from langchain_chroma import Chroma

load_dotenv()

# Make the Nebius key work locally (.env) and on the cloud (secrets)
if "NEBIUS_API_KEY" in st.secrets:
    os.environ["NEBIUS_API_KEY"] = st.secrets["NEBIUS_API_KEY"]

# --- password gate ---
def check_password():
    def entered():
        if st.session_state["password"] == st.secrets.get("APP_PASSWORD", ""):
            st.session_state["auth_ok"] = True
            del st.session_state["password"]
        else:
            st.session_state["auth_ok"] = False
    if st.session_state.get("auth_ok", False):
        return True
    st.text_input("Enter password:", type="password", on_change=entered, key="password")
    if "auth_ok" in st.session_state and not st.session_state["auth_ok"]:
        st.error("Incorrect password.")
    return False

if not check_password():
    st.stop()

# --- set up the bot once and reuse it (cached so it doesn't reload every click) ---
@st.cache_resource
def load_bot():
    embeddings = NebiusEmbeddings(model="Qwen/Qwen3-Embedding-8B")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    llm = ChatNebius(model="openai/gpt-oss-120b")
    return vectorstore, llm

vectorstore, llm = load_bot()

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
    sources = ", ".join(f"{c.metadata['game']} p{c.metadata['page']}" for c in chunks)
    return (note + " " if note else "") + llm.invoke(prompt).content, sources

def respond(question, forced_game=None):
    if forced_game:
        chunks = vectorstore.similarity_search(question, k=4, filter={"game": forced_game})
        return generate(question, chunks)

    named = games_named_in(question)
    if len(named) == 1:
        chunks = vectorstore.similarity_search(question, k=4, filter={"game": named[0]})
        return generate(question, chunks)
    if len(named) == 0:
        chunks = vectorstore.similarity_search(question, k=4)
        counts = Counter(c.metadata["game"] for c in chunks)
        top_game, top_count = counts.most_common(1)[0]
        if top_count > len(chunks) / 2:
            return generate(question, chunks, note=f"(Assuming {top_game}.)")
        return f"Which game do you mean? Options: {', '.join(sorted(GAME_ALIASES))}", ""
    return f"You named several games: {', '.join(named)} — which one?", ""

# --- the browser UI ---
st.title("🎲 Board Game Rules Bot")
st.caption("Ask a rules question about Catan, Saboteur, Evolution, Sequence, or Spot It. "
           "If it's ambiguous, I'll ask which game.")

# keep the conversation on screen
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# remember a question that's waiting for the user to name a game
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if user_input := st.chat_input("Ask a rules question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # If the bot was waiting for a game name, stitch it onto the earlier question
    if st.session_state.pending_question and user_input.lower().strip() in GAME_ALIASES:
        full_question = f"{st.session_state.pending_question} (in {user_input.strip()})"
        picked = games_named_in(user_input)[0]
        chunks = vectorstore.similarity_search(full_question, k=4, filter={"game": picked})
        answer, sources = generate(full_question, chunks)
        st.session_state.pending_question = None      # clear the memory
    else:
        answer, sources = respond(user_input)
        # If the bot just asked "which game?", remember this question for next turn
        if answer.startswith("Which game"):
            st.session_state.pending_question = user_input

    display = answer + (f"\n\n*sources: {sources}*" if sources else "")
    st.session_state.messages.append({"role": "assistant", "content": display})
    with st.chat_message("assistant"):
        st.markdown(display)
# app.py — browser chat interface for the board game rules bot

import os
import re
from collections import defaultdict
from difflib import get_close_matches

import streamlit as st
from langchain_nebius import NebiusEmbeddings, ChatNebius
from langchain_chroma import Chroma

# Load local .env if present (harmless on the cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Make the Nebius key work locally (.env) and on the cloud (secrets)
try:
    if "NEBIUS_API_KEY" in st.secrets:
        os.environ["NEBIUS_API_KEY"] = st.secrets["NEBIUS_API_KEY"]
except Exception:
    pass


# --- set up the bot once and reuse it (cached so it doesn't reload every click) ---
@st.cache_resource
def load_bot():
    embeddings = NebiusEmbeddings(model="Qwen/Qwen3-Embedding-8B")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    llm = ChatNebius(model="openai/gpt-oss-120b")
    return vectorstore, llm

vectorstore, llm = load_bot()


# ---------- game name resolution ----------

# Each game maps to the words/spellings people actually type
GAME_ALIASES = {
    "catan": ["catan", "settlers of catan", "settlers"],
    "saboteur": ["saboteur", "sabateur", "sabotour"],
    "evolution": ["evolution", "evo"],
    "sequence": ["sequence", "sequance"],
    "spot-it": ["spot it", "spotit", "spot-it", "dobble", "doble"],
}

# Distinctive content words that strongly imply one game (no game name needed)
GAME_KEYWORDS = {
    "catan": ["robber", "hex", "settlement", "longest road", "victory point", "ore", "wheat", "brick", "desert"],
    "saboteur": ["dwarf", "tunnel", "nugget", "gold digger", "gold-digger", "mining", "path card", "pickaxe"],
    "evolution": ["species", "trait", "carnivore", "watering hole", "body size", "population", "food token", "extinct"],
    "sequence": ["chip", "jack", "two-eyed", "one-eyed", "joker space", "five in a row"],
    "spot-it": ["symbol", "matching symbol", "mini-game", "tower", "well", "hot potato"],
}

def _normalize(text):
    # lowercase, strip punctuation/hyphens, collapse spaces
    text = text.lower()
    text = re.sub(r"[-_]", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def games_named_in(text):
    """Find game names in text, tolerant of spacing, hyphens, and small typos."""
    norm = _normalize(text)
    words = norm.split()
    found = []
    for game, aliases in GAME_ALIASES.items():
        for alias in aliases:
            a = _normalize(alias)
            if a in norm:                                   # direct match
                found.append(game); break
            if get_close_matches(a, words, n=1, cutoff=0.8):  # fuzzy near-miss
                found.append(game); break
    return found

def games_by_keyword(text):
    """Find games implied by distinctive content words (no game name needed)."""
    norm = _normalize(text)
    hits = []
    for game, words in GAME_KEYWORDS.items():
        if any(_normalize(w) in norm for w in words):
            hits.append(game)
    return hits

def game_from_llm(question):
    """Last resort: ask the LLM which game a vague question is about."""
    games = ", ".join(GAME_ALIASES.keys())
    prompt = f"""The user asked a board game rules question but didn't clearly name the game.
Based on the description, which ONE of these games is it most likely about?
Games: {games}
Reply with EXACTLY one game id from that list, or the word "unclear" if you can't tell.

Question: {question}
Answer (one word):"""
    guess = _normalize(llm.invoke(prompt).content).replace(" ", "-")
    return guess if guess in GAME_ALIASES else None


# ---------- answer generation ----------

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
    answer = (note + " " if note else "") + llm.invoke(prompt).content

    # Group pages by game, dedupe + sort -> "Catan (pages 4, 5, 8)"
    pages_by_game = defaultdict(set)
    for c in chunks:
        pages_by_game[c.metadata["game"]].add(c.metadata["page"])
    parts = []
    for game in sorted(pages_by_game):
        pgs = sorted(pages_by_game[game])
        label = "page" if len(pgs) == 1 else "pages"
        parts.append(f"{game.title()} ({label} {', '.join(str(p) for p in pgs)})")
    sources = "  •  ".join(parts)
    return answer, sources


# ---------- the layered resolver ----------

def respond(question, forced_game=None):
    # 0. game already known (named or confirmed) -> answer it
    if forced_game:
        chunks = vectorstore.similarity_search(question, k=4, filter={"game": forced_game})
        return generate(question, chunks)

    # 1. exact or fuzzy game NAME in the question
    named = games_named_in(question)
    if len(set(named)) == 1:
        chunks = vectorstore.similarity_search(question, k=4, filter={"game": named[0]})
        return generate(question, chunks)
    if len(set(named)) > 1:
        return f"You mentioned several games: {', '.join(sorted(set(named)))} — which one?", ""

    # 2. distinctive KEYWORDS point to exactly one game -> confirm
    kw = set(games_by_keyword(question))
    if len(kw) == 1:
        return f"__CONFIRM__{next(iter(kw))}", ""

    # 3. LLM guesses the game from a vague description -> confirm
    guess = game_from_llm(question)
    if guess:
        return f"__CONFIRM__{guess}", ""

    # 4. nothing resolves -> ask which game
    return f"Which game do you mean? Options: {', '.join(sorted(GAME_ALIASES))}", ""


# ---------- turn handling (manages the back-and-forth) ----------

def route_fresh(user_input):
    answer, sources = respond(user_input)
    if answer.startswith("__CONFIRM__"):
        g = answer.replace("__CONFIRM__", "")
        st.session_state.pending_question = user_input
        st.session_state.pending_game = g
        return f"Do you mean **{g.title()}**? (reply *yes*, or name the correct game)", ""
    if answer.startswith("Which game") or answer.startswith("You mentioned"):
        st.session_state.pending_question = user_input
        return answer, ""
    return answer, sources

def handle_turn(user_input):
    pend_q = st.session_state.get("pending_question")
    pend_g = st.session_state.get("pending_game")
    reply = user_input.lower().strip()

    # We previously asked "Do you mean X?"
    if pend_g:
        st.session_state.pending_game = None
        st.session_state.pending_question = None
        if reply in ("yes", "y", "yeah", "yep", "yup", "correct", "right"):
            return respond(pend_q, forced_game=pend_g)
        named = games_named_in(user_input)
        if named:
            return respond(pend_q, forced_game=named[0])
        if reply in ("no", "n", "nope", "nah"):
            st.session_state.pending_question = pend_q
            return f"Okay — which game is it? Options: {', '.join(sorted(GAME_ALIASES))}", ""
        return route_fresh(user_input)   # treat as a new question

    # We previously asked "Which game?"
    if pend_q:
        st.session_state.pending_question = None
        named = games_named_in(user_input)
        if named:
            return respond(pend_q, forced_game=named[0])
        return route_fresh(user_input)   # no game named -> treat as new question

    # Normal new question
    return route_fresh(user_input)


# ---------- the browser UI ----------

st.title("🎲 Board Game Rules Bot")
st.caption("Ask a rules question about Catan, Saboteur, Evolution, Sequence, or Spot It. "
           "If it's unclear, I'll ask which game.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "pending_game" not in st.session_state:
    st.session_state.pending_game = None

def show_assistant(answer, sources):
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("📖 Sources"):
                st.markdown(f"Answer drawn from: **{sources}**")

# replay the conversation so far
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📖 Sources"):
                st.markdown(f"Answer drawn from: **{m['sources']}**")

# handle a new message
if user_input := st.chat_input("Ask a rules question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer, sources = handle_turn(user_input)
    show_assistant(answer, sources)
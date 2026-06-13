# golden_set.py — the ground truth for evaluation. Written BEFORE scoring.
# behavior is what the bot SHOULD do:
#   ANSWER = give the fact   ASSUME = infer game + state assumption
#   ASK    = ask which game  REFUSE = say it doesn't know
# game = where the correct answer lives (for scoring retrieval); None when N/A

GOLDEN_SET = [
    # --- ANSWER: clear questions, correct fact expected ---
    {"q": "How many victory points do you need to win Catan?",
     "behavior": "ANSWER", "game": "catan", "expected": "10"},

    {"q": "How do saboteurs win a round?",
     "behavior": "ANSWER", "game": "saboteur", "expected": "When the treasure finish card is never reached / path to treasure fails"},

    {"q": "In Evolution, what can a Carnivore eat?",
     "behavior": "ANSWER", "game": "evolution", "expected": "Only meat by attacking other species; never plant food"},

    {"q": "What does a two-eyed Jack do in Sequence?",
     "behavior": "ANSWER", "game": "sequence", "expected": "It's wild — play a chip on any open space"},

    {"q": "How many cards are in a Spot It deck?",
     "behavior": "ANSWER", "game": "spot-it", "expected": "55 cards"},

    {"q": "What happens when you roll a 7 in Catan?",
     "behavior": "ANSWER", "game": "catan", "expected": "Discard if >7 cards, move the robber, steal a card"},

    # --- ASSUME: no game named, but term is unique to one game ---
    {"q": "What happens when a species goes extinct?",
     "behavior": "ASSUME", "game": "evolution", "expected": "Discard its trait cards & board, draw replacement cards (Evolution)"},

    {"q": "Can the robber be moved onto the desert?",
     "behavior": "ASSUME", "game": "catan", "expected": "Edge case — rulebook places robber on desert at setup but doesn't clearly say you may move it there later"},

    # --- ASK: ambiguous, term spans several games ---
    {"q": "How many cards do I start with?",
     "behavior": "ASK", "game": None, "expected": "Should ask which game (true for Catan/Saboteur/Sequence/Evolution)"},

    {"q": "How many players can play?",
     "behavior": "ASK", "game": None, "expected": "Should ask which game"},

    {"q": "How do I win?",
     "behavior": "ASK", "game": None, "expected": "Should ask which game"},

    {"q": "Who takes the first turn?",
     "behavior": "ASK", "game": None, "expected": "Should ask which game"},

    # --- REFUSE: not answerable from the rulebooks ---
    {"q": "What happens if I knock over the table?",
     "behavior": "REFUSE", "game": None, "expected": "Not in any rulebook — should say it doesn't know"},

    {"q": "Can I play Catan using a standard deck of playing cards?",
     "behavior": "REFUSE", "game": None, "expected": "Not covered by the rules — should say it doesn't know"},

    {"q": "What's the best opening strategy in Catan?",
     "behavior": "REFUSE", "game": "catan", "expected": "Names a game, but strategy isn't in the rulebook — should not invent one"},
]
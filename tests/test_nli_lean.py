from sentence_transformers import CrossEncoder
from scipy.special import softmax
import numpy as np

nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base')

claims_and_evidence = [
    (
        "A new species of frog was discovered in the Amazon.",
        "Scientists announce the discovery of a tiny new frog species dwelling in the Amazon rainforest canopy."
    ),
    (
        "The Federal Reserve announced a quarter-point interest rate increase.",
        "Fed hikes rates by 0.25%, matching expectations in latest policy announcement."
    ),
    (
        "Eating 50 pounds of sugar a day is healthy.",
        "Doctors warn that excessive sugar intake leads to diabetes and severe health complications."
    )
]

for claim, ev in claims_and_evidence:
    raw = nli_model.predict([claim, ev])
    scores = softmax(raw)
    print(f"Claim: {claim}")
    print(f"Con: {scores[0]:.4f}, Ent: {scores[1]:.4f}, Neu: {scores[2]:.4f}")
    
    # Testing lean
    if scores[1] > scores[0] * 1.2:
        print("  -> Lean Supports")
    elif scores[0] > scores[1] * 1.2:
        print("  -> Lean Refutes")
    else:
        print("  -> Neutral")
    print("-" * 40)

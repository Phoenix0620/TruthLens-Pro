import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from sentence_transformers import CrossEncoder
import numpy as np

model = CrossEncoder('cross-encoder/nli-distilroberta-base')

claim = "NFL receiver Rondale Moore found dead in his Indiana hometown"
evidence = "NFL wide receiver Rondale Moore dies at age of 25. The man was found dead in his Indiana hometown."

pairs_wrong = [(claim, evidence)]
pairs_correct = [(evidence, claim)]

scores_wrong = model.predict(pairs_wrong)
probs_wrong = np.exp(scores_wrong) / np.sum(np.exp(scores_wrong), axis=1, keepdims=True)

scores_correct = model.predict(pairs_correct)
probs_correct = np.exp(scores_correct) / np.sum(np.exp(scores_correct), axis=1, keepdims=True)

print("Original (Claim -> Evidence):")
print(f"Contra: {probs_wrong[0][0]:.3f}, Entail: {probs_wrong[0][1]:.3f}, Neutral: {probs_wrong[0][2]:.3f}")

print("\nFixed (Evidence -> Claim):")
print(f"Contra: {probs_correct[0][0]:.3f}, Entail: {probs_correct[0][1]:.3f}, Neutral: {probs_correct[0][2]:.3f}")

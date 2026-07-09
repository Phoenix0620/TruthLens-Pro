from sentence_transformers import CrossEncoder
from scipy.special import softmax
import numpy as np

model = CrossEncoder("cross-encoder/nli-distilroberta-base")
claim = "pyramids were build by egyptians"
evidence = "Ancient Egypt was a cradle of civilization concentrated along the lower reaches of the Nile River in Northeast Africa. It emerged from prehistoric Egy..."

raw_logits = model.predict([claim, evidence])
scores = softmax(raw_logits)

print("Contradict (0):", scores[0])
print("Entailment (1):", scores[1])
print("Neutral    (2):", scores[2])
print("Argmax:", np.argmax(scores))

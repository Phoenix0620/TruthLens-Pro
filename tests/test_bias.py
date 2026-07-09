import sys
import os
from core.linguistics import LinguisticAnalyzer

analyzer = LinguisticAnalyzer(model_path="force_fallback")

real_news = [
    "The Federal Reserve announced a quarter-point interest rate increase on Wednesday.",
    "Scientists have discovered a new species of frog in the Amazon rainforest.",
    "The local city council voted 5-2 to approve the new zoning laws for the downtown district."
]

fake_news = [
    "Government secretly hides aliens in Area 51.",
    "Drinking bleach cures all known viruses instantly according to top doctors!",
    "The moon landing was entirely faked in a Hollywood studio by Stanley Kubrick."
]

print("--- REAL NEWS ---")
for text in real_news:
    res = analyzer.analyze(text)
    print(f"[{res['label'].upper()}] ({res['confidence']:.2f}) - {text}")

print("\n--- FAKE NEWS ---")
for text in fake_news:
    res = analyzer.analyze(text)
    print(f"[{res['label'].upper()}] ({res['confidence']:.2f}) - {text}")

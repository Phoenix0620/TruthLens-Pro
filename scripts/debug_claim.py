import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.orchestrator import TruthOrchestrator

orchestrator = TruthOrchestrator(use_rag=False, use_web=True)

test_claims = [
    "Ratan Tata is dead.", # A basic declarative claim that is true
    "A news article says Ratan Tata is dead." # longer
]

for claim in test_claims:
    print(f"\n--- Testing Claim: {claim} ---")
    res = orchestrator.analyze_claim(claim)
    print(f"Verdict: {res['verdict']}")
    print(f"Confidence: {res['confidence_percent']}")
    print(f"Source: {res['primary_source']}")
    print(f"Web Queries: {res['web_queries_generated']}")
    print(f"Linguistic: {res['linguistic_data']}")
    print("Web Evidence details:")
    for ev in res['web_evidence']:
        print(f"  - {ev['relation']} | {ev['credibility']['credibility_score']} | {ev['url']}")

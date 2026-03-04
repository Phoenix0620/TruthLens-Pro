import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.orchestrator import TruthOrchestrator

claim = "India won against West Indies and gained a chance to qualify for semi finals"

print("Initializing Orchestrator...")
orchestrator = TruthOrchestrator()
print(f"Analyzing claim: {claim}")
res = orchestrator.analyze_claim(claim)

print(f"\nFINAL VERDICT: {res['verdict']}")
print(f"Confidence: {res['confidence_percent']}")
print(f"Primary Source: {res['primary_source']}")
print(f"Generated Queries: {res.get('web_queries_generated', [])}")

print("\nWEB EVIDENCE:")
for w in res.get('web_evidence', []):
    print(f"\nTitle: {w['title']}")
    print(f"URL: {w['url']}")
    print(f"Relation: {w['relation']}")
    print(f"Weight: {w['weight']}")

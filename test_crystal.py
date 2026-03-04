import sys
import traceback
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.orchestrator import TruthOrchestrator

claim = "Scientists at Google created an AI system that discovered over 2 million new crystal structures in a single research project."

try:
    print("Initializing Orchestrator...")
    orchestrator = TruthOrchestrator()
    print(f"Analyzing claim: {claim}")
    res = orchestrator.analyze_claim(claim)

    print(f"\nFINAL VERDICT: {res['verdict']}")
    print(f"Confidence: {res['confidence_percent']}")
    print(f"Primary Source: {res['primary_source']}")
    print(f"RAG Matches: {res['rag_matches']}")
    print(f"Linguistic Data: {res['linguistic_data']}")
    print("\nWEB EVIDENCE:")
    for w in res.get('web_evidence', []):
        print(f"Title: {w['title']}")
        print(f"Relation: {w['relation']}")
        print(f"Weight: {w['weight']}")
except Exception as e:
    print(f"CRASH OCCURRED: {e}")
    traceback.print_exc()

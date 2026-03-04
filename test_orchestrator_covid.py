import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.orchestrator import TruthOrchestrator

claim = "Drinking hot water every 15 minutes can kill the COVID-19 virus inside your body."

orchestrator = TruthOrchestrator()
res = orchestrator.analyze_claim(claim)

print(f"\nFINAL VERDICT: {res['verdict']}")
print(f"Confidence: {res['confidence_percent']}")
print(f"Primary Source: {res['primary_source']}")
print(f"RAG Matches: {res['rag_matches']}")
print(f"Linguistic Data: {res['linguistic_data']}")
print(f"Web Verdict Extracted: Web_v={res.get('web_evidence')}")

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import TruthOrchestrator
from core.web_validator import WebValidator
from core.query_planner import QueryPlanner

if __name__ == "__main__":
    claim = "Water is composed of two hydrogen atoms and one oxygen atom."
    
    print("\n--- 1. Query Planner ---")
    qp = QueryPlanner()
    queries = qp.generate_queries(claim)
    print(f"Generated Queries: {queries}")
    
    print("\n--- 2. Web Validator ---")
    wv = WebValidator()
    web_res = wv.verify(claim, queries)
    print(f"Web Verdict: {web_res['verdict']}")
    print(f"Web Confidence: {web_res['confidence']}")
    for i, ev in enumerate(web_res.get('details', [])):
        print(f"  Ev {i+1} [{ev['relation']}] (Wt: {ev['weight']}): {ev['title']} ({ev['url']})")
        print(f"      Snippet: {ev['snippet'][:150]}...")
        
    print("\n--- 3. Orchestrator End-to-End ---")
    orch = TruthOrchestrator(use_rag=False, use_web=True)
    final_res = orch.analyze_claim(claim)
    print(f"FINAL VERDICT: {final_res['verdict']}")
    print(f"FINAL CONF: {final_res['confidence_percent']}%")
    print(f"Source: {final_res['primary_source']}")
    print(f"Linguistics: {final_res['linguistic_data']['label']} (Conf: {final_res['linguistic_data']['confidence']:.2f})")

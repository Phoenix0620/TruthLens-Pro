import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def log(msg):
    print(msg, flush=True)

try:
    log("T1: Importing Orchestrator...")
    from core.orchestrator import TruthOrchestrator

    claim = "Scientists at Google created an AI system that discovered over 2 million new crystal structures in a single research project."

    log("T2: Initializing Orchestrator...")
    orchestrator = TruthOrchestrator()
    
    log(f"T3: Analyzing claim: {claim}")
    
    log("T4: Getting Linguistic Data...")
    ling_res = orchestrator.ling.analyze(claim)
    log(f"T4 Done. Ling output: {ling_res['label']}")

    log("T5: Getting RAG Matches...")
    rag_res = orchestrator.rag.search(claim)
    log(f"T5 Done. RAG matches: {len(rag_res)}")
    
    log("T6: Generating Web Queries...")
    queries = orchestrator.query_planner.generate_queries(claim)
    log(f"T6 Done. Queries: {queries}")
    
    log("T7: Running Web Validator verification...")
    # This is probably where it crashes.
    web_res = orchestrator.web.verify(claim, queries)
    log(f"T7 Done. Web verdict: {web_res.get('verdict')}")
    
    log("SUCCESS: All pieces ran natively.")
    
except Exception as e:
    log(f"EXCEPTION CAUGHT: {e}")
    traceback.print_exc()
except BaseException as e: # Catches SystemExit
    log(f"BASE EXCEPTION CAUGHT: {e}")
    traceback.print_exc()

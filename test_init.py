import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

print("STEP 1: Importing Modules")
from core.rag_engine import RagEngine
from core.linguistics import LinguisticAnalyzer
from core.query_planner import QueryPlanner
from core.web_validator import WebValidator

print("STEP 2: RAG Engine")
rag = RagEngine('data/liar/train.tsv')

print("STEP 3: Linguistic Analyzer")
ling = LinguisticAnalyzer()

print("STEP 4: Query Planner")
qp = QueryPlanner()

print("STEP 5: Web Validator")
wv = WebValidator()

print("SUCCESS: All modules initialized successfully.")

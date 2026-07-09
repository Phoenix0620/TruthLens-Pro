import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.web_validator import WebValidator
from core.query_planner import QueryPlanner

claim = "NFL receiver Rondale Moore found dead in his Indiana hometown"

planner = QueryPlanner()
queries = planner.generate_queries(claim)
print(f"Generated Queries: {queries}")

validator = WebValidator()
res = validator.verify(claim, queries)

print("\n=== VALIDATOR VERDICT ===")
print(res['verdict'])
print(f"Confidence: {res['confidence']}")
print("\n=== DETAILED EVIDENCE ===")
for detail in res['details']:
    print(f"\nTitle: {detail['title']}")
    print(f"URL: {detail['url']}")
    print(f"Relation: {detail['relation']}")
    print(f"Weight: {detail['weight']}")
    print(f"Raw NLI Probs: {detail.get('raw_probs', 'Not stored')}")

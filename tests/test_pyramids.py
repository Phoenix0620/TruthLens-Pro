from core.orchestrator import TruthOrchestrator
import json

orch = TruthOrchestrator(use_rag=False, use_web=True)
claim = 'pyramids were build by egyptians'
res = orch.analyze_claim(claim)

print('--- FINAL RESULT ---')
print(f'Verdict: {res["verdict"]}')
print(f'Confidence: {res["confidence_percent"]}%')

print(f'\nLinguistics Output:')
print(json.dumps(res['linguistic_data'], indent=2))

print(f'\nWeb Output:')
for w in res.get('web_evidence', []):
    print(f"[{w['relation']}] {w['title']} (Weight: {w['weight']})")
    print(f"Snippet: {w['snippet'][:150]}...")

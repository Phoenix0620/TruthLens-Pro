from core.orchestrator import TruthOrchestrator
import json

orch = TruthOrchestrator(use_rag=False, use_web=True)
claim = 'Water is composed of two hydrogen atoms and one oxygen atom.'
res = orch.analyze_claim(claim)

print('--- FINAL RESULT ---')
print(f'Verdict: {res["verdict"]}')
print(f'Confidence: {res["confidence_percent"]}%')

print(f'\nLinguistics Output:')
print(json.dumps(res['linguistic_data'], indent=2))

print(f'\nWeb Output:')
print(res['web_evidence'][0]['relation'] if res['web_evidence'] else 'No evidence')

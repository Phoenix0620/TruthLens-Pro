import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.llm_judge import LLMJudge

claim = "India won against West Indies and gained a chance to qualify for semi finals"
evidence_texts = [
    "T20 World Cup: India set up semi-final against England after edging West Indies by five wickets - as it happened. 10 hours ago - That's a wrap on the Super 8s at the T20 World Cup. India have beaten West Indies by five wickets to book their spot in the semi-finals alongside England, New Zealand, and South Africa.",
    "Sanju Samson's heroics lead India to semi-finals - Sanju Samson scored an unbeaten 97 as India beat West Indies by five wickets to set up a semi-final meeting with England at the T20 World Cup."
]

judge = LLMJudge()

results = judge.evaluate(claim, evidence_texts)

for i, (verdict, conf) in enumerate(results):
    print(f"Evidence {i+1}:")
    print(f"Verdict: {verdict} - Conf: {conf}\n")

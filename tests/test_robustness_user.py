from core.orchestrator import TruthOrchestrator
import time

def run_tests():
    orch = TruthOrchestrator(use_rag=False, use_web=True)
    
    tests = [
        {"claim": "Scientists at MIT confirmed that drinking coffee cures all types of cancer.", "expected_category": "False"},
        {"claim": "The Earth does not revolve around the Sun.", "expected_category": "False"},
        {"claim": "In 2023, India achieved a historic lunar landing near the Moon’s south pole with its Chandrayaan-3 mission.", "expected_category": "True"},
        {"claim": "Water boils at 150°C at standard atmospheric pressure.", "expected_category": "False"},
        {"claim": "Twitter banned Donald Trump in 2024.", "expected_category": "False"},
        {"claim": "SHOCKING: NASA secretly confirmed alien life was discovered on Mars.", "expected_category": "False"},
        {"claim": "A massive earthquake has just struck Japan.", "expected_category": "False"}
    ]
    
    passed = 0
    
    for i, test in enumerate(tests):
        print(f"\n[{i+1}/7] Testing: {test['claim']}")
        res = orch.analyze_claim(test['claim'])
        verdict = res["verdict"]
        conf = res["confidence_percent"]
        ling = res["linguistic_data"]["label"]
        web_ev = res.get("web_evidence", [])
        
        # Check pass/fail logic
        is_pass = False
        if test["expected_category"] == "False" and "False" in verdict:
            is_pass = True
        elif test["expected_category"] == "True" and "True" in verdict:
            is_pass = True
            
        if is_pass:
            passed += 1
            print(f"  [PASS] -> Verdict: {verdict} ({conf}%)")
        else:
            print(f"  [FAIL] -> Expected {test['expected_category']}, got {verdict} ({conf}%)")
            print(f"     Linguistics: {ling}")
            web_supports = sum(1 for e in web_ev if e['relation'] == 'Supports')
            web_refutes = sum(1 for e in web_ev if e['relation'] == 'Refutes')
            web_neutral = sum(1 for e in web_ev if e['relation'] == 'Neutral')
            print(f"     Web Evidence: {web_supports} Supports, {web_refutes} Refutes, {web_neutral} Neutral")
            for w in web_ev[:2]:
                print(f"       - [{w['relation']}] {w['title']}")
                print(f"         {w['snippet'][:100]}...")
                
    print(f"\n--- Accuracy: {passed}/{len(tests)} ({(passed/len(tests))*100:.1f}%) ---")

if __name__ == "__main__":
    run_tests()

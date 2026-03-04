import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.orchestrator import TruthOrchestrator

test_claims = [
    # FAKE CLAIMS (Should be False)
    ("Drinking hot water every 15 minutes can kill the COVID-19 virus inside your body.", False),
    ("NASA has confirmed that the Earth is actually flat.", False),
    ("Donald Trump resigned from his presidential post in 2026.", False),
    ("Eating garlic cures all forms of cancer.", False),
    ("Bill Gates put microchips in the COVID-19 vaccines to track people.", False),
    ("The moon landing in 1969 was completely faked in a Hollywood studio.", False),
    ("A new law legally requires everyone to give up their pets to the government.", False),
    ("Sharks have been found swimming in the flooded subway stations of New York.", False),
    ("Drinking bleach is a proven medical treatment for viral infections.", False),
    ("The Great Wall of China is the only man-made object visible from space with the naked eye.", False),
    
    # TRUE CLAIMS (Should be True)
    ("Water is composed of two hydrogen atoms and one oxygen atom.", True),
    ("The Eiffel Tower is located in Paris, France.", True),
    ("NFL receiver Rondale Moore died at the age of 25 in his Indiana hometown.", True),
    ("Joe Biden served as the 46th President of the United States.", True),
    ("The speed of light in a vacuum is approximately 299,792 kilometers per second.", True),
    ("Japan is an island nation located in East Asia.", True),
    ("Apple Inc. was co-founded by Steve Jobs and Steve Wozniak.", True),
    ("Mount Everest is the highest mountain above sea level.", True),
    ("Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients.", True),
    ("The human skeleton of an adult consists of 206 bones.", True)
]

print("Initializing Orchestrator...")
orchestrator = TruthOrchestrator()

correct_count = 0
total = len(test_claims)

print("Running 20 Test Claims...\n")

for claim, expected_truth in test_claims:
    res = orchestrator.analyze_claim(claim)
    verdict_text = res['verdict']
    
    # Map verdict to binary for testing
    is_true = "True" in verdict_text or "Plausible" in verdict_text
    is_false = "False" in verdict_text or "Suspicious" in verdict_text or "Unsupported" in verdict_text
    
    actual_truth = True if is_true else (False if is_false else None)
    
    status = "✅ PASS" if actual_truth == expected_truth else "❌ FAIL"
    if actual_truth == expected_truth:
        correct_count += 1
        
    print(f"Claim: {claim}")
    print(f"Expected: {expected_truth} | Actual Verdict: {verdict_text} ({res['confidence_percent']}%) [{res['primary_source']}]")
    print(f"Status: {status}\n")

print(f"Accuracy: {correct_count}/{total} ({(correct_count/total)*100}%)")

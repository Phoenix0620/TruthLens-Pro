import sys
import os
from core.linguistics import LinguisticAnalyzer

analyzer = LinguisticAnalyzer(model_path="force_fallback")
res = analyzer.analyze("Government secretly hides aliens in Area 51.")
print("Test Result:", res)

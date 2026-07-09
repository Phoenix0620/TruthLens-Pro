import sys
import os
import shap
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.linguistics import LinguisticAnalyzer

analyzer = LinguisticAnalyzer()

if analyzer.pipeline:
    text = "NFL receiver Rondale Moore found dead in his Indiana hometown"
    explainer = shap.Explainer(analyzer.pipeline)
    shap_values = explainer([text])
    
    print("\nSHAP Values Object:")
    print("Values Shape:", shap_values.values.shape)
    print("Data:", shap_values.data)
    
    for i in range(len(shap_values.data[0])):
        val = shap_values.values[0][i]
        print(f"Token: {shap_values.data[0][i]} | Value: {val}")
else:
    print("No fine-tuned pipeline found.")

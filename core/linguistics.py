import os
import torch
import shap
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class LinguisticAnalyzer:
    def __init__(self, model_path=None):
        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), "models/linguistic_model")
        else:
            self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.fallback_pipeline = None
        
        try:
            if os.path.exists(self.model_path):
                print("Loading custom fine-tuned linguistic model...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.pipeline = pipeline('text-classification', model=self.model, tokenizer=self.tokenizer, top_k=None)
            else:
                print("Custom fine-tuned model not found. Using Hugging Face's Fake News BERT model as fallback.")
                self.fallback_pipeline = pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect", top_k=None)
        except Exception as e:
            print(f"Error loading linguistic model: {e}")

    def analyze(self, text):
        active_pipeline = self.pipeline if self.pipeline else self.fallback_pipeline
        
        if active_pipeline:
            # Shared logic for both text-classification models
            result_list = active_pipeline(text)
            # Handle potential nested list return
            if isinstance(result_list[0], list):
                outputs = result_list[0]
            else:
                outputs = result_list
                
            probs = {d['label']: d['score'] for d in outputs}
            
            if self.pipeline:
                # Custom Finetuned: LABEL_0 = True, LABEL_1 = False
                true_prob = probs.get('LABEL_0', 0.0)
                false_prob = probs.get('LABEL_1', 0.0)
                source_name = "Finetuned DistilBERT"
            else:
                # HF Fake-News-Bert-Detect: LABEL_0 = Fake, LABEL_1 = Real
                false_prob = probs.get('LABEL_0', 0.0)
                true_prob = probs.get('LABEL_1', 0.0)
                source_name = "HF Fake News BERT (Fallback)"
            
            label = "likely_false" if false_prob > true_prob else "likely_true"
            conf = max(true_prob, false_prob)
            
            # Explainable AI (XAI) Token Extraction via SHAP
            top_words = []
            try:
                # Use pre-compiled explainer for speed
                if getattr(self, 'explainer', None) is None:
                    self.explainer = shap.Explainer(active_pipeline)
                
                # Truncate text to 40 words max for SHAP to strictly prevent CPU 10-minute exponential freezes 
                text_for_xai = " ".join(text.split()[:40])
                shap_values = self.explainer([text_for_xai])
                
                # SHAP tensor shape resolution depending on the pipeline
                if len(shap_values.values.shape) == 3:
                    class_idx = 1 if label == "likely_false" else 0
                    word_impacts = shap_values.values[0, :, class_idx]
                else:
                    word_impacts = shap_values.values[0]
                    
                tokens = shap_values.data[0]
                
                # Sort token indices by absolute mathematical impact on the model's decision
                top_indices = sorted(range(len(word_impacts)), key=lambda i: abs(word_impacts[i]), reverse=True)
                for i in top_indices:
                    token = str(tokens[i]).strip()
                    if len(token) > 2 and token.isalpha(): # Skip punctuation and subword markers
                        top_words.append({
                            "word": token,
                            "impact": float(word_impacts[i])
                        })
                    if len(top_words) >= 5:
                        break
            except Exception as e:
                print(f"SHAP XAI Error: {e}")
            
            return {
                "label": label,
                "confidence": conf,
                "source": source_name,
                "xai_words": top_words
            }
        else:
            return {"label": "error", "confidence": 0.0, "source": "None", "xai_words": []}

if __name__ == "__main__":
    analyzer = LinguisticAnalyzer()
    res = analyzer.analyze("Government secretly hides aliens in Area 51.")
    print("Test Result:", res)

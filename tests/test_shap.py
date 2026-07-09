import shap
from transformers import pipeline

print("Loading pipeline...")
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define labels you want the zero-shot model to classify against
labels = ["real news", "fake news"]

# Create a custom prediction wrapper that always passes the labels
def predict(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif hasattr(texts, "tolist"):
        texts = texts.tolist()
    else:
        texts = list(texts)
        
    results = pipe(texts, candidate_labels=labels)
    if not isinstance(results, list):
        results = [results]
        
    # SHAP expects output probabilities as a 2D array: [num_samples, num_classes]
    scores = []
    for r in results:
        # Sort scores to match label order
        label_scores = {label: score for label, score in zip(r['labels'], r['scores'])}
        scores.append([label_scores[l] for l in labels])
    return scores

print("Initializing explainer...")
explainer = shap.Explainer(predict, shap.maskers.Text(pipe.tokenizer))

text = "NFL receiver Rondale Moore found dead in his Indiana hometown"
print("Running shap...")
shap_values = explainer([text])

print(shap_values)

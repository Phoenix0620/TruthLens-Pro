# TruthLens Pro

The ultimate, full-fledged hybrid architecture for fake news mitigation. Based strongly on the "FNAD Architectures Core Principles":
- **Dataset (LIAR) + RAG**
- **Linguistic Sentiment Engine**
- **Autonomous Query Expanded Live Web Verification**
- **Credibility Scoring System**

## Core Modules breakdown:
1. `core/rag_engine.py`: Loads the massive LIAR dataset, embeds it using `sentence-transformers`, limits to FAISS indexing for ultra-fast vector math. Represents the Ground Truth engine. 
2. `core/linguistics.py`: Determines if the text 'reads' like deceptive or true media. Uses `zero-shot-classification` for instantaneous use, but fully integrates with custom `distilbert-base-uncased` checkpoints when you train it.
3. `core/query_planner.py`: Rather than searching a massive abstract claim, it mimics LLMs (ChatGPT) by breaking the claim into related, targeted queries using `spaCy` NLP extraction and NLTK RAKE.
4. `core/credibility.py`: An autonomous scoring metric using `Whois` domain age and a dataset of known blacklist/whitelist domain registries.
5. `core/web_validator.py`: A `DuckDuckGo` API search runner backed internally by `nli-distilroberta-base`. It scales NLI relation evidence based on the source's `Credibility Profile`.
6. `app.py`: The massive Glassmorphism premium dashboard orchestrating all the components.

## Setup Instructions

Make sure you have all required tooling:
```powershell
pip install python-whois faiss-cpu spacy tldextract rake-nltk duckduckgo-search sentence-transformers streamlit torch transformers datasets scikit-learn pandas numpy
python -m spacy download en_core_web_sm
```

## Running the Dashboard
Simply execute:
```powershell
.\run_app.bat
```

## Want maximum performance? Train your AI locally!
If you do not want to use the Zero-Shot Linguistical Fallback engine, run `train_model.py`.
```powershell
python train_model.py
```
*Note: Depending on your hardware, running finetuning might take time. Once completed, `TruthLens Pro` will seamlessly integrate it into the UI logic!*

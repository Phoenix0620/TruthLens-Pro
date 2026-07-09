# TruthLens Pro: System Architecture and Methodology Summary
*A comprehensive technical breakdown for technical documentation preparation.*

## 1. System Overview
TruthLens Pro is a **Hybrid Fake News Detection and Fact-Checking System**. Unlike standard fact-checkers that rely on a single database or a single AI model, TruthLens uses a multi-modal ensemble approach. It combines **offline Linguistic Stylometry**, **Retrieval-Augmented Generation (RAG)** on historical datasets, and **Live Contextual Web Search** powered by Natural Language Inference (NLI) to mathematically aggregate a final truth verdict.

## 2. Core Modules and Methodologies

### A. Linguistic Analysis & Explainable AI (XAI)
**Purpose:** To detect manipulative, sensationalist, or statistically deceptive language patterns even before a claim is fact-checked online.
*   **Primary Model:** A locally hosted, fine-tuned transformer model (based on `DistilBERT`), trained natively on fake news datasets to classify text as `likely_true` or `likely_false`.
*   **Fallback Model:** Hugging Face's `jy46604790/Fake-News-Bert-Detect` via pipeline. If the local model is unavailable, the system instantly routes to this cloud-based classifier.
*   **Explainable AI (SHAP):** The system integrates SHapley Additive exPlanations (`shap.Explainer`). It dissects the user's claim word-by-word and mathematically calculates the positive/negative impact of individual tokens on the neural network's decision. This is passed to the UI to highlight exactly *which words* triggered the "fake" or "real" classification.

### B. Query Generation (Query Planner)
**Purpose:** To transform a user's conversational claim into optimized search engine queries.
*   **Methodology:** Uses Natural Language Processing (NLP) rules or lightweight LLMs to extract core entities, stripped of stop words, and synthesizes 3 distinct search permutations (e.g., "[Claim] debunked", "Fact check [Claim]").

### C. Live Web Search & Credibility Profiling
**Purpose:** To fetch real-time evidence from the internet.
*   **Search Engine Integration:** Utilizes the `duckduckgo_search` (DDGS) library to asynchronously scrape top news articles and web results based on the generated queries.
*   **Credibility Engine (`credibility.py`):** Before evidence is accepted, the domain is evaluated.
    *   **Domain Matching:** Checks against a curated database of known highly credible sources (e.g., *bbc.com, reuters.com*) and explicit disinformation/satire domains.
    *   **Scoring Logic:** Assigns a `credibility_score` (1 to 5). High scores apply mathematical multipliers to the evidence's weight later in the pipeline.

### D. The Web Validator (Natural Language Inference)
**Purpose:** To mathematically compare the User's Claim against the scraped Web Articles to see if they agree or disagree.
*   **Model Used:** `cross-encoder/nli-distilroberta-base`.
*   **Methodology (Softmax NLI):** The system passes the `[User Claim]` and the `[Web Article Snippet]` side-by-side into the Cross-Encoder. The model outputs raw logits which are converted to probabilities using a `softmax` activation function.
*   **Stance Extraction (Entailment vs. Contradiction):**
    *   **Argmax Classification:** Directly maps the highest probability to `Supports` (Entailment), `Refutes` (Contradiction), or `Neutral`.
    *   **Semantic Leaning Overrides:** Because search engine snippets are often fragmented, strict argmax sometimes forces valid journalism into a "Neutral" state. The system uses a mathematical "leaning" ratio: *If Entailment > Contradiction * 1.2, it leans to Supports.*
    *   **Credibility Trust Bypass:** If a source has high credibility (score >= 3) and mentions the topic neutrally (contradiction < 0.2), the system implicitly trusts it as verified reporting and upgrades it to `Supports`.
    *   **Noise Filtering:** Implements strict probability floors (e.g., Contradiction must be > 30%) to ensure irrelevant snippets don't falsely trigger a "Refutes" state.

### E. Retrieval-Augmented Generation (RAG Engine)
**Purpose:** To cross-reference claims against previously debunked historical lies.
*   **Methodology:** Loads the `LIAR` dataset (or similar benchmark datasets). When a user submits a claim, it vectorizes the claim and performs semantic similarity matching against thousands of known true/false statements. 

### F. The Truth Orchestrator (Ensemble Logic)
**Purpose:** To act as the central brain, mathematically fusing the offline linguistics, RAG, and live web evidence into a single verdict.
*   **The 80/20 Ensemble Math:** 
    *   **Web Evidence (80% weight):** Ground-truth reporting takes precedence. The system calculates a normalized `support_score` vs `refute_score` across all web documents (factoring in credibility multipliers) to generate a `web_truth_score`.
    *   **Linguistic Analysis (20% weight):** Acts as intuitive backup. If the claim is highly sensationalist, it drags the score down.
    *   **Outward Scaling Math:** Web confidence is scaled outward from a neutral 50% boundary (e.g., a "Likely True" web consensus correctly forces the ensemble higher, rather than acting as a raw literal percentage that accidentally drags the ensemble down).
*   **Final Output:** Based on the aggregated `ensemble_truth_score`, the Orchestrator resolves a distinct verdict: `Likely True`, `Likely False`, `Contested`, or `Unsupported`.

## 3. Frontend / User Interface
*   **Framework:** Streamlit (`app.py`).
*   **Aesthetics:** Engineered with a custom "Premium Glassmorphism" UI using injected CSS. It features an animated gradient background, floating geometric cards, and glowing verdict alerts.
*   **Data Visualization:** Explicitly renders the semantic relationship of web articles (Supports/Refutes), renders SHAP token XAI as highlighted colored pills, and dynamically switches UI states based on the presence of conclusive Web vs. RAG evidence.

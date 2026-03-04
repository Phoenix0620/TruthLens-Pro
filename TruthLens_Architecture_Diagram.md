# TruthLens Pro: System Architecture Block Diagram

Below is the Mermaid-js block diagram detailing the hybrid pipeline of the TruthLens Pro fact-checking system. 

If your markdown viewer supports Mermaid (like GitHub, GitLab, or VS Code), it will render automatically. You can also copy the code block below and paste it into [Mermaid Live Editor](https://mermaid.live/) to generate a high-quality PNG or SVG image.

```mermaid
flowchart TD
    %% Styling
    classDef user fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#fff,font-weight:bold
    classDef engine fill:#334155,stroke:#94a3b8,stroke-width:2px,color:#fff
    classDef api fill:#0ea5e9,stroke:#0284c7,stroke-width:2px,color:#fff
    classDef logic fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    classDef output fill:#f43f5e,stroke:#e11d48,stroke-width:2px,color:#fff
    classDef db fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px,color:#fff

    %% Nodes
    User(("👤 User Input (Claim)")):::user
    
    %% RAG Pathway
    RAG["🏛️ RAG Ground Truth Engine\n(sentence-transformers + FAISS)"]:::engine
    Dataset[("LIAR Dataset\n(12,000+ Verified Statements)")]:::db
    RAGMatch{"Match > 80%?"}:::logic
    
    %% Web Pathway
    QueryGen["🧠 Autonomous Query Planner\n(spaCy + RAKE-NLTK)"]:::engine
    DuckDuckGo(("DuckDuckGo Search API\n(Live Web Scraping)")):::api
    
    Credibility["🛡️ Credibility Engine\n(python-whois Domain Profiling)"]:::engine
    WhoisDB[("Known Domain Whitelist/Blacklist\n+ Live Server Ping")]:::db
    
    NLI["🔍 NLI Verification\n(cross-encoder/nli-distilroberta-base)"]:::engine
    
    Consensus{"Credibility-Weighted\nConsensus"}:::logic
    
    %% Linguistic Pathway
    Linguistic["📝 Linguistic Style Analyzer\n(DistilBERT / Zero-Shot BART)"]:::engine
    Fallback{"Fallback Needed?"}:::logic

    %% Output
    Verdict[["📊 Final Output / UI Dashboard\n(Streamlit orchestrator)"]]:::output

    %% Flow
    User --> RAG
    User --> QueryGen
    User --> Linguistic
    
    %% RAG Logic
    Dataset <--> RAG
    RAG --> |Vector Search| RAGMatch
    
    %% Web Logic
    QueryGen --> |Extract Entities & Keywords| DuckDuckGo
    DuckDuckGo --> |Return Live Articles| NLI
    DuckDuckGo --> |Return URLs| Credibility
    WhoisDB <--> Credibility
    
    User --> |Compare input to Live Articles| NLI
    
    NLI --> |Entailment / Neutral / Contradict| Consensus
    Credibility --> |Domain Trust Score multiplier| Consensus
    
    %% Aggregation
    RAGMatch --> |Yes| Verdict
    RAGMatch --> |No| Fallback
    
    Consensus --> |Valid Web Evidence Found| Verdict
    Consensus --> |No Web Evidence Found| Fallback
    
    Fallback --> |Yes| Linguistic
    Linguistic --> |Analyze Semantic Deception| Verdict
```

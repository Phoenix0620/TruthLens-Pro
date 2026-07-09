# TruthLens Pro Architecture (Updated 2026)

This diagram reflects the latest hybrid architecture incorporating the 80/20 Ensemble Math, Softmax NLI Leaning, High-Credibility Syntactic Bypass, and the Fake-News-BERT Fallback logic.

```mermaid
graph TD
    %% Define Node Styles
    classDef input fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#fff,font-weight:bold,rx:10,ry:10;
    classDef pathway fill:#1e293b,stroke:#334155,stroke-width:2px,color:#fff,font-weight:bold,rx:5,ry:5;
    classDef engine fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff,rx:5,ry:5;
    classDef model fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff,rx:5,ry:5;
    classDef logic fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff,rx:20,ry:20;
    classDef db fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px,color:#fff,rx:5,ry:5;
    classDef final fill:#ef4444,stroke:#dc2626,stroke-width:3px,color:#fff,font-weight:bold,rx:10,ry:10;

    %% Entry Point
    User([User Input: Text Claim]):::input --> Orchy{Truth Orchestrator}:::pathway
    
    %% RAG Pathway (Left)
    Orchy --> |Parallel Execution 1| RAG[RAG Ground Truth Engine]:::engine
    RAG --> LIAR[(LIAR Benchmark DB)]:::db
    LIAR --> RAG_Check{Match > 80%?}:::logic
    RAG_Check -- Yes --> FinalOutput
    
    %% Web Search Pathway (Center)
    RAG_Check -- No --> QP[Autonomous Query Planner]:::engine
    Orchy --> |Parallel Execution 2| QP
    QP --> DDGS[(DDGS Live Search)]:::db
    DDGS --> NLI[Web Validator]:::engine
    
    %% NLI Sub-logic
    NLI --> Softmax[CrossEncoder Softmax]:::model
    Softmax --> CredEngine[Credibility Profiler]:::engine
    CredEngine --> |"Known Domain Whitelist"| LogicOverride{Credibility >= 3\n& Contradict < 0.2?}:::logic
    
    LogicOverride -- Yes --> Support[Force 'Supports' Stance]:::model
    LogicOverride -- No --> SemLean[Semantic Leaning Ratio\n(Entail vs Contradict)]:::logic
    
    SemLean --> AggWeb[Aggregate Web Truth Score]:::engine
    Support --> AggWeb
    
    %% Linguistic Pathway (Right)
    Orchy --> |Parallel Execution 3| Ling[Linguistic Stylometry]:::engine
    Ling --> LocalModel{Load Custom\nFinetuned DistilBERT}:::logic
    
    LocalModel -- Available --> SHAP[SHAP Token XAI Module]:::model
    LocalModel -- Failed/Missing --> Fallback[Fallback to HF\nFake-News-Bert-Detect]:::model
    Fallback --> SHAP
    SHAP --> AggLing[Linguistic Truth Score]:::engine
    
    %% Ensemble Fusion (Bottom)
    AggWeb --> |80% Weight| Ensemble{Mathematical Fusion}:::pathway
    AggLing --> |20% Weight| Ensemble
    
    Ensemble --> Scaling[Outward Neutrality Scaling]:::logic
    Scaling --> FinalOutput[[Final Verdict Evaluated\nLikely True / False / Contested]]:::final
```

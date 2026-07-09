from .rag_engine import RagEngine
from .linguistics import LinguisticAnalyzer
from .query_planner import QueryPlanner
from .web_validator import WebValidator

class TruthOrchestrator:
    def __init__(self, use_rag=True, use_web=True):
        self.rag = RagEngine('../data/liar/train.tsv') if use_rag else None
        self.ling = LinguisticAnalyzer()
        self.query_planner = QueryPlanner() if use_web else None
        self.web = WebValidator() if use_web else None

    def analyze_claim(self, claim):
        # 1. Linguistic Analysis
        ling_res = self.ling.analyze(claim)
        
        # 2. RAG Check (Ground Truth Database Match)
        rag_res = self.rag.search(claim) if self.rag else []
        rag_res = [r for r in rag_res if r['score'] >= 0.50]
        best_match = None
        if rag_res:
            best_match = max(rag_res, key=lambda x: x['score'])
            
        # 3. Query Generation & Web Search
        queries = self.query_planner.generate_queries(claim) if self.query_planner else []
        web_res = self.web.verify(claim, queries) if self.web else {}
        
        # Aggregate logic
        score = 0
        verdict = "Uncertain"
        
        # Rule 1: High RAG match > 0.8 is considered authoritative Ground Truth
        if best_match and best_match['score'] > 0.8:
            verdict = best_match['label']
            source = "LIAR Ground Truth Dataset"
            confidence = int(best_match['score'] * 100)
        else:
            # Rule 2: 80/20 Mathematical Ensemble (Cap Linguistic influence)
            web_verdict = web_res.get('verdict', 'Unverified')
            web_conf = web_res.get('confidence', 0)
            
            ling_label = ling_res['label']
            ling_conf = ling_res['confidence']
            
            # Map Linguistic to a truth score [0, 1]
            if "true" in ling_label.lower() or "credible" in ling_label.lower():
                ling_truth_score = ling_conf
            else:
                ling_truth_score = 1.0 - ling_conf
                
            # Cap Linguistic at 20%
            ling_weight = 0.20
            
            if web_verdict == "Unverified":
                 # RAG was weak, Web was Unverified. 
                 
                 # --- CLICKBAIT OVERRIDE ---
                 # If the text is overwhelmingly deceptive/clickbait (>85% false confidence)
                 # AND the claim is long enough for linguistic patterns to be reliable
                 word_count = len(claim.split())
                 if ling_truth_score < 0.15 and word_count > 5:
                     verdict = "Likely False (Linguistic Override)"
                     confidence = int((1.0 - ling_truth_score) * 100)
                     source = "Linguistic Pattern Matching (High Deception)"
                 else:
                     # Linguistic can only contribute 20% MAX.
                     ensemble_conf = ling_truth_score * ling_weight
                     verdict = "Unsupported (No Factual Evidence Found)"
                     confidence = int(ensemble_conf * 100)
                     source = "Ensemble: Web Failed, Linguistic Cap Hit"
            else:
                 # Web Evidence exists
                 if "False" in web_verdict:
                     # Scale web_conf outward from neutral: A 0.4 refute confidence means 0.3 truth score.
                     web_truth_score = 0.5 - (web_conf * 0.5)
                 elif "True" in web_verdict:
                     # Scale web_conf outward from neutral: A 0.4 true confidence means 0.7 truth score.
                     web_truth_score = 0.5 + (web_conf * 0.5)
                 else:
                     web_truth_score = 0.5
                     
                 web_weight = 0.80
                 
                 # Ensemble Math
                 ensemble_truth_score = (web_truth_score * web_weight) + (ling_truth_score * ling_weight)
                 
                 # Resolve
                 if ensemble_truth_score > 0.6:
                     verdict = "Likely True (Ensemble)"
                     confidence = int(ensemble_truth_score * 100)
                 elif ensemble_truth_score < 0.4:
                     verdict = "Likely False (Ensemble)"
                     confidence = int((1.0 - ensemble_truth_score) * 100)
                 else:
                     # --- CLICKBAIT OVERRIDE ---
                     # Contested zone. If Web is mixed but language is highly deceptive:
                     word_count = len(claim.split())
                     if ling_truth_score < 0.15 and word_count > 5: 
                         verdict = "Likely False (Linguistic Override)"
                         confidence = int((1.0 - ling_truth_score) * 100)
                         source = "Live Web (Inconclusive) + Linguistic Pattern (High Deception)"
                     else:
                         verdict = "Contested (Ensemble)"
                         confidence = int(max(ensemble_truth_score, 1.0 - ensemble_truth_score) * 100)
                 
                 source = f"Live Web (80%) + Linguistic Tone (20%)"

        return {
            "verdict": verdict,
            "confidence_percent": confidence,
            "primary_source": source,
            "linguistic_data": ling_res,
            "rag_matches": rag_res[:1] if rag_res else [], # Just top match
            "web_queries_generated": queries,
            "web_evidence": web_res.get("details", [])
        }

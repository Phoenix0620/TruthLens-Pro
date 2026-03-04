import numpy as np
from ddgs import DDGS
from .credibility import CredibilityEngine
from sentence_transformers import CrossEncoder
from scipy.special import softmax

class WebValidator:
    def __init__(self):
        self.cred_engine = CredibilityEngine()
        self.nli_model = CrossEncoder("cross-encoder/nli-distilroberta-base")

    def verify(self, claim, queries):
        print("Gathering Web Evidence...")
        results = []
        with DDGS() as ddgs:
            for q in queries:
                try:
                    search_results = list(ddgs.text(q, max_results=3))
                    results.extend(search_results)
                except Exception as e:
                    print(f"DuckDuckGo error: {e}")
                    
        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in results:
            if r['href'] not in seen_urls:
                seen_urls.add(r['href'])
                unique_results.append(r)
                
        if not unique_results:
            return {"verdict": "Unverified", "confidence": 0, "details": []}

        # Analyze with NLI + Credibility
        analyzed_evidence = []
        support_score = 0
        refute_score = 0
        total_cred_weight = 0
        
        for res in unique_results:
            url = res['href']
            
            # Prepare text for NLI
            evidence_text = res.get('body', '') + ' ' + res.get('title', '')
            raw_logits = self.nli_model.predict([claim, evidence_text])
            nli_scores = softmax(raw_logits)
            
            # cross-encoder/nli-distilroberta-base maps: 0: Contradiction, 1: Entailment, 2: Neutral
            contradict_prob = nli_scores[0]
            entail_prob = nli_scores[1]
            neutral_prob = nli_scores[2]
            
            # Use Credibility Engine
            cred_assessment = self.cred_engine.assess_source(url)
            base_weight = 1
            
            # Amplifying system: Multiply the weight by credibility factor
            mult = 1 + (cred_assessment['credibility_score'] / 10)
            weight = base_weight * mult
            
            # Semantic leaning and credibility-based stance extraction
            # If a highly credible source covers the topic and DOES NOT refute it, it implies verified coverage
            if cred_assessment['credibility_score'] >= 3 and contradict_prob < 0.3:
                rel_label = "Supports"
                # Boost support mathematically since NLI struggles with incomplete journalism snippets
                support_score += weight 
                total_cred_weight += weight
            elif entail_prob > contradict_prob * 1.2 and entail_prob > 0.05:
                rel_label = "Supports"
                support_score += weight
                total_cred_weight += weight
            elif contradict_prob > entail_prob * 1.2 and contradict_prob > 0.3:
                rel_label = "Refutes"
                refute_score += weight
                total_cred_weight += weight
            else:
                rel_label = "Neutral"
                total_cred_weight += (weight * 0.1)
                
            analyzed_evidence.append({
                'title': res['title'],
                'url': url,
                'snippet': res['body'],
                'relation': rel_label,
                'credibility': cred_assessment,
                'weight': round(weight, 2)
            })

        if total_cred_weight == 0:
             return {"verdict": "Unverified", "confidence": 0, "details": analyzed_evidence}

        normalized_support = support_score / total_cred_weight
        normalized_refute = refute_score / total_cred_weight
        
        if support_score == 0 and refute_score == 0:
            return {"verdict": "Unverified", "confidence": 0.0, "details": analyzed_evidence}
            
        if normalized_support > normalized_refute:
            if normalized_support > 0.3:
                return {"verdict": "Likely True (Web)", "confidence": normalized_support, "details": analyzed_evidence}
            else:
                return {"verdict": "Unverified", "confidence": 0.0, "details": analyzed_evidence}
        else:
            if normalized_refute > 0.3:
                return {"verdict": "Likely False (Web)", "confidence": normalized_refute, "details": analyzed_evidence}
            else:
                return {"verdict": "Unverified", "confidence": 0.0, "details": analyzed_evidence}

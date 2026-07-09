import numpy as np
import re
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
            
            # --- NLP HEURISTICS OVERRIDES ---
            
            # 0. Death / Life Events Heuristic: NLI fails to map date ranges to "is dead".
            death_keywords = {"dead", "died", "death", "passes", "passed", "deceased", "killed"}
            import re
            clean_claim = re.sub(r'[^\w\s]', '', claim.lower())
            claim_death = bool(set(clean_claim.split()).intersection(death_keywords))
            
            # Match formats like (28 December 1937 – 9 October 2024) or explicit statements of passing
            has_death_proof = bool(re.search(r'\(\s*(?:b\.\s*)?[12]\d{3}\s*[-–].+?[12]\d{3}\s*\)', evidence_text)) or "died" in evidence_text.lower() or "passed away" in evidence_text.lower() or "death" in evidence_text.lower() or "is dead" in evidence_text.lower() or "no more" in evidence_text.lower()
            
            if claim_death and has_death_proof:
                 entail_prob = max(float(entail_prob), 0.8)
                 contradict_prob = min(float(contradict_prob), 0.1)
                 
            # 1. Temporal Mismatch: NLI ignores dates.
            claim_years = set(re.findall(r'\b(19\d{2}|20\d{2})\b', claim))
            snippet_years = set(re.findall(r'\b(19\d{2}|20\d{2})\b', evidence_text))
            
            # If claim cites a specific year, and snippet cites DIFFERENT years (but not the claim year), penalize.
            if claim_years:
                if snippet_years and not claim_years.intersection(snippet_years):
                    entail_prob *= 0.2
                    contradict_prob = max(float(contradict_prob), 0.6)
                elif not snippet_years:
                    # Claim specifies a year, but snippet doesn't confirm any date. Cannot fully Support.
                    entail_prob *= 0.5
                    neutral_prob = max(float(neutral_prob), 0.5)
                
            # 2. Negation Mismatch: NLI is easily tricked by "lexical overlap" when negations are used.
            negations = {"not", "never", "no", "false", "fake", "didn't", "doesn't", "hasn't", "won't", "cannot", "can't"}
            claim_neg = bool(set(claim.lower().split()).intersection(negations))
            snippet_neg = bool(set(evidence_text.lower().split()).intersection(negations))
            
            if claim_neg != snippet_neg:
                # One is negated, the other isn't. Penalize entailment.
                if entail_prob > 0.3:  # Only penalize if it actually looked like an entailment
                    entail_prob *= 0.3
                    contradict_prob = max(float(contradict_prob), 0.4)
                
            # 3. Numerical / Quantitative Mismatch
            claim_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', claim)) - claim_years
            snippet_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', evidence_text)) - snippet_years
            
            if claim_nums and snippet_nums and not claim_nums.intersection(snippet_nums):
                # If numbers don't match exactly, the fact is likely altered.
                entail_prob *= 0.2
                contradict_prob = max(float(contradict_prob), 0.6)
                
            # 4. Absolute / Superlative Mismatch
            absolutes = {"all", "cures", "every", "always", "100%", "guaranteed", "definitively"}
            claim_abs = bool(set(claim.lower().split()).intersection(absolutes))
            snippet_abs = bool(set(evidence_text.lower().split()).intersection(absolutes))
            
            if claim_abs and not snippet_abs:
                # Claim makes an absolute statement the web snippet doesn't support
                entail_prob *= 0.5
                contradict_prob = max(float(contradict_prob), 0.4)
                
            # Re-evaluate argmax if probabilities shifted
            nli_scores = [contradict_prob, entail_prob, neutral_prob]
            predicted_class = np.argmax(nli_scores)
            # --- END NLP HEURISTICS OVERRIDES ---
            
            # Semantic leaning and credibility-based stance extraction
            # If a highly credible source covers the topic and DOES NOT refute it, it implies verified coverage
            if cred_assessment['credibility_score'] >= 3 and contradict_prob < 0.3:
                rel_label = "Supports"
                # Boost support mathematically since NLI struggles with incomplete journalism snippets
                support_score += weight 
                total_cred_weight += weight
            elif predicted_class == 1 or entail_prob > 0.4:
                rel_label = "Supports"
                support_score += weight
                total_cred_weight += weight
            elif predicted_class == 0 or contradict_prob > 0.5:
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
            
        if normalized_support > normalized_refute and (normalized_support - normalized_refute) > 0.1:
            if normalized_support > 0.3:
                return {"verdict": "Likely True (Web)", "confidence": normalized_support, "details": analyzed_evidence}
            else:
                return {"verdict": "Unverified", "confidence": 0.0, "details": analyzed_evidence}
        elif normalized_refute > normalized_support and (normalized_refute - normalized_support) > 0.1:
            if normalized_refute > 0.3:
                return {"verdict": "Likely False (Web)", "confidence": normalized_refute, "details": analyzed_evidence}
            else:
                return {"verdict": "Unverified", "confidence": 0.0, "details": analyzed_evidence}
        else:
            # It's a tie or extremely close, web evidence is mixed
            if max(normalized_support, normalized_refute) >= 0.3:
                 return {"verdict": "Contested (Web)", "confidence": float(max(normalized_support, normalized_refute)), "details": analyzed_evidence}
            else:
                 return {"verdict": "Unverified", "confidence": 0.0, "details": analyzed_evidence}

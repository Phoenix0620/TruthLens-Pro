import requests
import json
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class LLMJudge:
    def __init__(self, model_name="llama3:8b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.system_prompt = (
            "You are an expert fact-checking AI. Your task is to analyze the provided Claim "
            "alongside the Evidence snippet and determine if the Evidence SUPPORTS, REFUTES, "
            "or is NEUTRAL towards the Claim.\n"
            "Respond with exactly ONE of the following words on the first line: SUPPORTS, REFUTES, or NEUTRAL. "
            "Do not provide any other explanation."
        )

    def evaluate(self, claim: str, evidence_list: List[str]) -> List[Tuple[str, float]]:
        """
        Evaluate a claim against a list of evidence strings.
        Returns a list of tuples (verdict, confidence).
        verdict is one of "Supports", "Refutes", "Neutral".
        """
        results = []
        for evidence in evidence_list:
            prompt = f"{self.system_prompt}\n\nClaim: {claim}\nEvidence: {evidence}\nVerdict:"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 1024
                }
            }
            try:
                # Add timeout to avoid hanging the UI too long per evidence
                response = requests.post(self.api_url, json=payload, timeout=30)
                if response.status_code == 200:
                    text_response = response.json().get("response", "").strip().upper()
                    if "SUPPORTS" in text_response:
                        results.append(("Supports", 0.9)) # Mocking high confidence
                    elif "REFUTES" in text_response:
                        results.append(("Refutes", 0.9))
                    else:
                        results.append(("Neutral", 0.9))
                else:
                    logger.warning(f"Ollama returned status {response.status_code}: {response.text}")
                    results.append(("Neutral", 0.0))
            except requests.exceptions.Timeout:
                logger.warning("Ollama API timed out.")
                results.append(("Neutral", 0.0))
            except Exception as e:
                logger.error(f"LLM Judge error: {e}")
                results.append(("Neutral", 0.0))
                
        return results

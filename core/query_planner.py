import spacy
from rake_nltk import Rake

class QueryPlanner:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.rake = Rake()
        except Exception as e:
            print(f"Error loading NLP tools: {e}")
            self.nlp = None

    def generate_queries(self, claim):
        queries = [claim] # Always start with full claim
        
        if self.nlp:
            doc = self.nlp(claim)
            
            # Method 1: Extract Noun Phrases
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Method 2: Entity extraction
            entities = [ent.text for ent in doc.ents]
            
            # Assemble semantic queries
            if entities:
                queries.append(" ".join(entities) + " true or false")
            
            # Method 3: Keyword Extraction via RAKE
            self.rake.extract_keywords_from_text(claim)
            keywords = self.rake.get_ranked_phrases()
            if keywords:
                # If the keyword phrase has enough context, use it. Otherwise, combine it.
                if len(keywords[0].split()) >= 2:
                    queries.append(f"{keywords[0]} reality")
                else:
                    queries.append(f"{claim} explained")
                
            # Filter distinct queries
            filtered = list(set(q for q in queries if len(q.split()) >= 3))
            
            # Return top 3 generated queries max
            return sorted(filtered, key=len, reverse=True)[:3]
            
        return queries

if __name__ == "__main__":
    planner = QueryPlanner()
    print(planner.generate_queries("The earth is flat and NASA admitted it in a secret document."))

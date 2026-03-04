import whois
import datetime
from tldextract import extract
import socket

class CredibilityEngine:
    def __init__(self):
        # Mapped known domains from user drawing requirements (Mocked but real structure)
        self.known_misinformation = {
            "infowars.com": -5,
            "theonion.com": -2, # Satire but treated as false,
            "beforeitsnews.com": -5,
            "naturalnews.com": -4
        }
        self.known_credible = {
            # International News Agencies
            "reuters.com": 5,
            "apnews.com": 5,
            "afp.com": 5,
            "bloomberg.com": 5,
            # Fact Checking Organizations
            "snopes.com": 5,
            "factcheck.org": 5,
            "politifact.com": 5,
            "fullfact.org": 5,
            "leadstories.com": 5,
            "poynter.org": 5,
            # Major Broadcasters / Publishers
            "bbc.com": 4,
            "bbc.co.uk": 4,
            "npr.org": 4,
            "pbs.org": 4,
            "nytimes.com": 4,
            "washingtonpost.com": 4,
            "wsj.com": 4,
            "theguardian.com": 4,
            # Science/Tech Journals
            "nature.com": 5,
            "sciencemag.org": 5,
            "scientificamerican.com": 4,
            # Reference
            "wikipedia.org": 4
        }

    def get_domain(self, url):
        # Extract example.com from https://www.example.com/path
        ext = extract(url)
        return f"{ext.domain}.{ext.suffix}"

    def get_domain_age_months(self, domain):
        try:
            w = whois.whois(domain)
            creation_date = w.creation_date
            
            # whois can return lists if multiple registry entries
            if type(creation_date) is list:
                creation_date = creation_date[0]
                
            if isinstance(creation_date, datetime.datetime):
                age = datetime.datetime.now() - creation_date
                # Months approx
                months = age.days // 30
                return months
            return -1 # Age not found
        except Exception:
            return -1

    def assess_source(self, url):
        domain = self.get_domain(url)
        age = self.get_domain_age_months(domain)
        
        score = 0
        status = "Neutral"
        
        # Check explicit blacklists/whitelists
        if domain in self.known_misinformation:
            score += self.known_misinformation[domain]
            status = "Known Misinformation"
        elif domain in self.known_credible:
            score += self.known_credible[domain]
            status = "Highly Credible"
        else:
            # If unknown, base on Domain Age
            # E.g. less than 6 months is highly suspicious for news
            if age != -1:
                if age < 6:
                    score -= 3
                    status = "Suspiciously New Domain"
                elif age > 120: # 10 years
                    score += 2
                    status = "Established Domain"
                    
        return {
            "domain": domain,
            "age_months": age,
            "credibility_score": score,
            "status": status
        }

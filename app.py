import sys
import os
import streamlit as st
import pandas as pd

# Add the directory to the path to import core package 
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.linguistics import LinguisticAnalyzer
from core.query_planner import QueryPlanner
from core.credibility import CredibilityEngine
from core.rag_engine import RagEngine
from core.web_validator import WebValidator

# Page Config
st.set_page_config(
    page_title="TruthLens Pro | Hybrid Fact Checker",
    page_icon="search",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;900&family=Inter:wght@400;500;600&display=swap');
    
    body { font-family: 'Inter', sans-serif; background-color: #020617; color: #f8fafc; }
    
    /* Animated Gradient Background */
    .stApp { 
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #09090b, #170f2e);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Mesmerizing Text Styling */
    .title-text { 
        font-family: 'Outfit', sans-serif;
        font-size: 4.5rem; font-weight: 900; text-align: center; 
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
        margin-bottom: 0.2rem;
        letter-spacing: -0.03em;
        text-shadow: 0px 4px 30px rgba(79, 172, 254, 0.4);
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }

    .subtitle-text { 
        font-family: 'Outfit', sans-serif;
        text-align: center; color: #94a3b8; font-size: 1.25rem; margin-bottom: 3rem; 
        font-weight: 300; letter-spacing: 0.1em; text-transform: uppercase;
    }
    
    /* Premium Glassmorphism Cards */
    .card {
        background: rgba(15, 23, 42, 0.45); 
        backdrop-filter: blur(24px); 
        -webkit-backdrop-filter: blur(24px);
        border-radius: 24px;
        padding: 32px; 
        border: 1px solid rgba(255, 255, 255, 0.05); 
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        border-left: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 1.5rem;
        overflow: hidden;
        position: relative;
    }
    
    .card::before {
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 50%; height: 100%;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.03), transparent);
        transform: skewX(-25deg);
        transition: 0.5s;
    }
    
    .card:hover::before {
        left: 150%;
    }

    .card:hover { 
        transform: translateY(-8px) scale(1.02); 
        box-shadow: 0 20px 50px -10px rgba(56, 189, 248, 0.3); 
        border-color: rgba(56, 189, 248, 0.5); 
    }
    
    /* Glowing Verdict Box */
    .verdict { 
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem; font-weight: 900; text-align: center; 
        padding: 24px; border-radius: 16px; margin: 30px 0; 
        letter-spacing: 0.05em; text-transform: uppercase;
        animation: pulseglow 2s infinite alternate;
    }
    
    @keyframes pulseglow {
        from { box-shadow: 0 0 10px -5px currentColor; }
        to { box-shadow: 0 0 30px 5px currentColor; }
    }
    
    .verdict-True { background: rgba(34, 197, 94, 0.08); border: 2px solid #22c55e; color: #4ade80; }
    .verdict-False { background: rgba(239, 68, 68, 0.08); border: 2px solid #ef4444; color: #f87171; }
    .verdict-Neutral { background: rgba(234, 179, 8, 0.08); border: 2px solid #eab308; color: #facc15; }
    
    /* Soft Inputs */
    .stTextArea textarea { 
        background: rgba(0, 0, 0, 0.4) !important; 
        color: #f8fafc !important; 
        border: 1px solid rgba(255,255,255,0.1) !important; 
        border-radius: 16px; 
        font-size: 1.15rem;
        padding: 20px;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: inset 0 0 0 1px #38bdf8, 0 0 20px rgba(56, 189, 248, 0.2) !important;
    }
    
    /* Vibrant Hover Buttons */
    .stButton button { 
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: #ffffff; 
        font-family: 'Outfit', sans-serif;
        font-weight: 800; font-size: 1.25rem; border-radius: 16px; 
        padding: 1rem 0; width: 100%; border: none; 
        box-shadow: 0 10px 25px -5px rgba(0, 114, 255, 0.6); 
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        text-transform: uppercase; letter-spacing: 0.1em;
        position: relative;
        overflow: hidden;
    }
    .stButton button:hover { 
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
        box-shadow: 0 15px 35px -5px rgba(0, 114, 255, 0.8); 
        transform: translateY(-4px) scale(1.01); 
    }
    .stButton button:active {
        transform: translateY(2px);
    }
</style>
""", unsafe_allow_html=True)

from core.orchestrator import TruthOrchestrator

@st.cache_resource
def load_orchestrator():
    # Cache all heavy models inside the TruthOrchestrator
    return TruthOrchestrator()

# Layout
st.markdown('<div class="title-text">TruthLens Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">LLM-Driven Hybrid Fact Checker (RAG + Credibility System)</div>', unsafe_allow_html=True)

with st.spinner("Warming up Neural Engines..."):
    orchestrator = load_orchestrator()

# Main input layout
claim_input = st.text_area("Analyze Claim:", height=120, placeholder="Paste a news article snippet or claim here...")
analyze_btn = st.button("Check Facts & Verify")

if analyze_btn and claim_input:
    # Centralized Orchestrator Call
    with st.spinner("Running 80/20 Mathematical Ensemble..."):
         orchestrator_res = orchestrator.analyze_claim(claim_input)
         
    verdict_text = orchestrator_res['verdict']
    conf = orchestrator_res['confidence_percent']
    source = orchestrator_res['primary_source']
    
    ling_res = orchestrator_res['linguistic_data']
    rag_res = orchestrator_res['rag_matches']
    generated_queries = orchestrator_res['web_queries_generated']
    web_details = orchestrator_res['web_evidence']

    # Final cleanup to ensure strict "True/False" or "Real/Fake" aesthetics
    v_class = "Neutral"
    if "True" in verdict_text or "Real" in verdict_text: 
        v_class = "True"
    elif "False" in verdict_text or "Fake" in verdict_text: 
        v_class = "False"

    # Present Results
    st.markdown(f"""
    <div class="verdict verdict-{v_class}">
        VERDICT: {verdict_text.upper()} <span style="font-size: 1rem; color: #e2e8f0; opacity: 0.8;">({conf}% Match)</span>
    </div>
    <div style="text-align: center; color: #94a3b8; margin-bottom: 2rem;">Source: {source}</div>
    """, unsafe_allow_html=True)

    # Detailed Tabs
    if not rag_res:
        tab_verification, tab_linguistics = st.tabs(["Live Verification & Credibility", "Linguistic Analysis"])
    else:
        tab_truth, tab_verification, tab_linguistics = st.tabs(["Ground Truth", "Live Verification & Credibility", "Linguistic Analysis"])
        
        with tab_truth:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Closest Dataset Matches")
            for i, match in enumerate(rag_res):
                st.markdown(f"**Match {i+1}**: {match['statement']} *(Score: {match['score']:.2f})*")
                st.caption(f"Annotated Label: `{match['label']}`")
                st.divider()
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_verification:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Credibility Weighted Evidence")
        details = web_details
        if not details:
            st.info("No verifiable web evidence found.")
        else:
             for ev in details:
                 cred = ev['credibility']
                 icon = "[SUPPORT]" if ev['relation'] == 'Supports' else "[REFUTE]" if ev['relation'] == 'Refutes' else "[NEUTRAL]"
                 
                 st.markdown(f"#### {icon} [{ev['title']}]({ev['url']})")
                 st.markdown(f"**Credibility Profiler:** Status `{cred['status']}` | Age:`{cred['age_months']} mos` | Engine Score: `{cred['credibility_score']}`")
                 st.markdown(f"> {ev['snippet'][:300]}")
                 st.caption(f"Amplified Weight: {ev['weight']}")
                 st.divider()
        st.markdown('</div>', unsafe_allow_html=True)

    # Removed Query Planner tab

    with tab_linguistics:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Linguistic / Sentiment Analysis")
        st.metric("Deception/Truth Likelihood", f"{ling_res['label'].replace('_', ' ').title()}")
        st.progress(ling_res['confidence'])
        
        # XAI Rendering
        xai_words = ling_res.get('xai_words', [])
        if xai_words:
             st.markdown("#### 🔍 Explainable AI (SHAP Token Impact)")
             st.write("The AI identified these specific words as having the highest mathematical impact on its deception classification:")
             
             tags_html = ""
             for w in xai_words:
                 word = w['word']
                 # Red gradient pill for high-impact words
                 color = "rgba(239, 68, 68, 0.2)"
                 border = "#ef4444"
                 tags_html += f'<span style="background: {color}; border: 1px solid {border}; padding: 6px 14px; border-radius: 16px; margin-right: 8px; font-weight: 600; font-size: 0.9rem;">{word}</span>'
                 
             st.markdown(f'<div style="margin: 15px 0;">{tags_html}</div>', unsafe_allow_html=True)
             
        st.markdown('</div>', unsafe_allow_html=True)

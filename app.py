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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Base */
    body { font-family: 'Inter', sans-serif; background-color: #000000; color: #ffffff; }
    .stApp { background: #000000; }
    
    /* Hide Streamlit Header & Navigation */
    header[data-testid="stHeader"] { visibility: hidden !important; }
    .stDeployButton { display: none !important; }
    footer { visibility: hidden !important; }

    /* Layout Spacing */
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 1500px !important; }

    /* Sticky Left Content for Scrolling */
    [data-testid="stColumn"]:nth-of-type(1), [data-testid="column"]:nth-of-type(1) {
        position: sticky !important;
        top: 2rem !important;
        align-self: flex-start !important;
        z-index: 100 !important;
    }
    
    /* Minimalist Typography */
    .title-text { 
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem; font-weight: 900; text-align: left; 
        color: #ffffff;
        letter-spacing: -0.05em;
        margin-bottom: 0px;
        line-height: 1;
    }
    
    .subtitle-text { 
        font-family: 'Inter', sans-serif;
        text-align: left; color: #888888; font-size: 1rem; margin-top: 8px; margin-bottom: 2rem; 
        font-weight: 400; letter-spacing: -0.01em; 
    }
    
    /* Stark Minimalist Cards */
    .card {
        background: transparent; 
        padding: 16px 0; 
        border: none; 
        border-top: 1px solid #222222;
        border-radius: 0px; 
        margin-bottom: 1rem;
    }
    
    /* Sharp Verdict Box */
    .verdict { 
        font-family: 'Inter', sans-serif;
        font-size: 2rem; font-weight: 800; text-align: left; 
        padding: 16px; border-radius: 0px; margin: 20px 0; 
        letter-spacing: -0.02em; text-transform: uppercase;
        border-left: 6px solid;
        background: #050505;
        border-top: 1px solid #111;
        border-right: 1px solid #111;
        border-bottom: 1px solid #111;
    }
    
    .verdict-True { border-left-color: #22c55e; color: #ffffff; }
    .verdict-True span { color: #22c55e !important; }
    .verdict-False { border-left-color: #ef4444; color: #ffffff; }
    .verdict-False span { color: #ef4444 !important; }
    .verdict-Neutral { border-left-color: #eab308; color: #ffffff; }
    .verdict-Neutral span { color: #eab308 !important; }
    
    /* Stark Inputs */
    .stTextArea textarea { 
        background: #000000 !important; 
        color: #ffffff !important; 
        border: 1px solid #333333 !important; 
        border-radius: 0px !important; 
        font-size: 1rem;
        padding: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #ffffff !important;
        box-shadow: none !important;
    }
    
    /* Stripped Back Buttons */
    .stButton button, .stButton button:focus { 
        background: #ffffff !important;
        color: #000000 !important; 
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important; font-size: 1rem !important; border-radius: 0px !important; 
        padding: 0.75rem 0 !important; width: 100% !important; border: 1px solid transparent !important; 
        transition: background 0.1s ease !important;
        text-transform: uppercase !important; letter-spacing: 0.02em !important;
        outline: none !important; box-shadow: none !important;
    }
    .stButton button p { color: #000000 !important; }
    
    .stButton button:hover { 
        background: #f0f0f0 !important;
        color: #000000 !important;
        border-color: transparent !important;
    }
    .stButton button:active {
        background: #e0e0e0 !important;
        color: #000000 !important;
    }
    
    /* Styling Streamlit defaults to be more minimal */
    hr { margin: 1rem 0 !important; border-color: #222222 !important; }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid #222222; }
    .stTabs [data-baseweb="tab"] { background: transparent !important; border: none !important; color: #666666 !important; padding-left: 0; padding-right: 0; padding-bottom: 8px;}
    .stTabs [aria-selected="true"] { color: #ffffff !important; font-weight: 600; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #ffffff !important; height: 2px !important; }
    
    p, div, span, h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

from core.orchestrator import TruthOrchestrator

@st.cache_resource
def load_orchestrator():
    # Cache all heavy models inside the TruthOrchestrator
    return TruthOrchestrator()

# Layout
if 'results' not in st.session_state:
    st.session_state.results = None
if 'claim_input_val' not in st.session_state:
    st.session_state.claim_input_val = ""

with st.spinner("Warming up Neural Engines..."):
    orchestrator = load_orchestrator()

def perform_analysis(claim):
    with st.spinner("Running 80/20 Mathematical Ensemble..."):
        st.session_state.results = orchestrator.analyze_claim(claim)
        st.session_state.claim_input_val = claim
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

if st.session_state.results is None:
    # --- CENTERED INITIAL LAYOUT ---
    st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown('<div class="title-text" style="text-align: center;">TruthLens Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-text" style="text-align: center;">LLM-Driven Hybrid Fact Checker (RAG + Credibility System)</div>', unsafe_allow_html=True)
        claim = st.text_area("Analyze Claim:", height=150, placeholder="Paste a news article snippet or claim here...")
        if st.button("Check Facts & Verify"):
            if claim:
                perform_analysis(claim)
else:
    # --- SPLIT LAYOUT (LEFT/RIGHT) ---
    left_col, right_col = st.columns([1, 1.6], gap="large")
    
    with left_col:
        st.markdown('<div class="title-text">TruthLens Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-text">LLM-Driven Hybrid Fact Checker</div>', unsafe_allow_html=True)
        claim = st.text_area("Analyze Claim:", value=st.session_state.claim_input_val, height=150)
        
        if st.button("Check Facts & Verify"):
            if claim:
                perform_analysis(claim)
                
        # Results - Left Side
        orchestrator_res = st.session_state.results
        verdict_text = orchestrator_res['verdict']
        conf = orchestrator_res['confidence_percent']
        source = orchestrator_res['primary_source']
        
        v_class = "Neutral"
        if "True" in verdict_text or "Real" in verdict_text: v_class = "True"
        elif "False" in verdict_text or "Fake" in verdict_text: v_class = "False"
        
        st.markdown(f"""
        <div class="verdict verdict-{v_class}">
            VERDICT: {verdict_text.upper()}
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        orchestrator_res = st.session_state.results
        ling_res = orchestrator_res['linguistic_data']
        rag_res = orchestrator_res['rag_matches']
        web_details = orchestrator_res['web_evidence']
        
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        
        if not rag_res:
            tab_verification, tab_linguistics = st.tabs(["Live Verification & Credibility", "Linguistic Analysis"])
        else:
            tab_truth, tab_verification, tab_linguistics = st.tabs(["Ground Truth", "Live Verification", "Linguistic Analysis"])
            
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
            if not web_details:
                st.info("No verifiable web evidence found.")
            else:
                 sort_map = {'Supports': 0, 'Neutral': 1, 'Refutes': 2}
                 sorted_web_details = sorted(web_details, key=lambda x: sort_map.get(x.get('relation', 'Neutral'), 1))
                 for ev in sorted_web_details:
                     cred = ev['credibility']
                     icon = "[SUPPORT]" if ev['relation'] == 'Supports' else "[REFUTE]" if ev['relation'] == 'Refutes' else "[NEUTRAL]"
                     st.markdown(f"#### {icon} [{ev['title']}]({ev['url']})")
                     st.markdown(f"**Credibility Profiler:** Status `{cred['status']}` | Age:`{cred['age_months']} mos` | Engine Score: `{cred['credibility_score']}`")
                     st.markdown(f"> {ev['snippet'][:300]}")
                     st.caption(f"Amplified Weight: {ev['weight']}")
                     st.divider()
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_linguistics:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Linguistic / Sentiment Analysis")
            st.metric("Deception/Truth Likelihood", f"{ling_res['label'].replace('_', ' ').title()}")
            st.progress(ling_res['confidence'])
            
            xai_words = ling_res.get('xai_words', [])
            if xai_words:
                 st.markdown("#### 🔍 Explainable AI (SHAP Token Impact)")
                 st.write("The AI identified these specific words as having the highest mathematical impact on its deception classification:")
                 tags_html = ""
                 for w in xai_words:
                     word = w['word']
                     color = "#FFFFFF"
                     border = "#67568C"
                     text_color = "#67568C"
                     tags_html += f'<span style="background: {color}; border: 1px solid {border}; padding: 4px 12px; border-radius: 0px; margin-right: 8px; margin-bottom: 8px; display: inline-block; font-weight: 500; font-size: 0.85rem; color: {text_color};">{word}</span>'
                 st.markdown(f'<div style="margin: 15px 0;">{tags_html}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class RagEngine:
    def __init__(self, data_path, index_path="core/models/langchain_faiss"):
        self.data_path = data_path
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vectorstore = None
        
        # Load or build index
        if os.path.exists(self.index_path):
            print("Loading RAG Database (LangChain FAISS)...")
            self.load()
        else:
            print("Building RAG Database...")
            self.build_index()

    def load_data(self):
        try:
            df = pd.read_csv(self.data_path, sep='\t', header=None)
            docs = []
            for _, row in df.iterrows():
                statement = str(row[2]) if not pd.isna(row[2]) else ""
                label = str(row[1]) if not pd.isna(row[1]) else ""
                if statement:
                    docs.append(Document(page_content=statement, metadata={"label": label}))
            return docs
        except Exception as e:
            print(f"Error loading RAG data: {e}")
            return []

    def build_index(self):
        docs = self.load_data()
        if not docs:
            print("Empty statements, skipping index build.")
            return

        print(f"Embedding {len(docs)} statements using LangChain FAISS...")
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.vectorstore.save_local(self.index_path)
        print("RAG Index built successfully.")

    def load(self):
        try:
            self.vectorstore = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print("Error loading index, rebuilding...", e)
            self.build_index()

    def search(self, query, top_k=3):
        if not self.vectorstore:
            return []
            
        try:
            # Use raw distance search instead of strict relevance search to bypass Langchain's normalization crashes
            results_raw = self.vectorstore.similarity_search_with_score(query, k=top_k)
        except Exception as e:
            print(f"RAG Search failed: {e}")
            return []
            
        results = []
        for doc, distance in results_raw:
            # Convert FAISS L2 distance to a 0-1 similarity score
            sim_score = max(0.0, 1.0 - float(distance))
            results.append({
                'score': sim_score,
                'statement': doc.page_content,
                'label': doc.metadata.get('label', 'unknown')
            })
        return results

if __name__ == "__main__":
    # Test RAG build
    engine = RagEngine('../../data/liar/train.tsv', 'models/langchain_faiss')
    res = engine.search("NASA faked moon landing")
    print("Search Result:", res)

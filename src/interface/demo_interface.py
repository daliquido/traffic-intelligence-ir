#!/usr/bin/env python3
"""
Demo script for the Interactive Traffic Event Search Interface.
This demonstrates the key features without requiring interactive input.
"""

import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.tfidf_retrieval import TFIDFRetriever
from retrieval.bm25_retrieval import BM25Retriever
from retrieval.simple_enhanced_retrieval import SimpleEnhancedRetriever
from preprocessing.clean_text import preprocess_documents

class DemoSearchInterface:
    """Demo version of the interactive search interface."""
    
    def __init__(self):
        self.retrievers = {}
        self.documents_df = None
        self.enhanced_documents_df = None
    
    def initialize(self, sample_size: int = 2000):
        """Initialize retrievers with demo data."""
        print("Initializing Traffic Event Search Interface...")
        print("="*60)
        
        # Load documents
        documents_path = "/Users/i/traffic-intelligence-ir/data/processed/traffic_documents.csv"
        self.documents_df = pd.read_csv(documents_path)
        
        # Use sample for demo
        sample_df = self.documents_df.sample(n=sample_size, random_state=42)
        print(f"Loaded {len(sample_df)} traffic events")
        
        # Preprocess for TF-IDF and BM25
        processed_df = preprocess_documents(sample_df)
        
        # Initialize TF-IDF
        print("Setting up TF-IDF retriever...")
        tfidf_retriever = TFIDFRetriever()
        tfidf_retriever.fit(processed_df)
        self.retrievers['TF-IDF'] = tfidf_retriever
        
        # Initialize BM25
        print("Setting up BM25 retriever...")
        bm25_retriever = BM25Retriever()
        bm25_retriever.fit(processed_df)
        self.retrievers['BM25'] = bm25_retriever
        
        # Initialize Enhanced TF-IDF
        print("Setting up Enhanced TF-IDF retriever...")
        enhanced_retriever = SimpleEnhancedRetriever()
        self.enhanced_documents_df = enhanced_retriever.create_enhanced_documents(sample_df)
        enhanced_retriever.fit(self.enhanced_documents_df)
        self.retrievers['Enhanced TF-IDF'] = enhanced_retriever
        
        print(f"All {len(self.retrievers)} retrievers ready!")
        print(f"Available models: {list(self.retrievers.keys())}")
    
    def demo_search(self, query: str, retriever_name: str = 'TF-IDF'):
        """Demonstrate search functionality."""
        print(f"\n{'='*80}")
        print(f"{retriever_name} Search Results for: '{query}'")
        print(f"{'='*80}")
        
        retriever = self.retrievers[retriever_name]
        results = retriever.search(query, top_k=3)
        
        if not results:
            print("No relevant documents found.")
            return
        
        print(f"Found {len(results)} relevant traffic events:\n")
        
        for i, (doc_id, title, score) in enumerate(results, 1):
            # Get document details
            if 'Enhanced' in retriever_name:
                doc = self.enhanced_documents_df[self.enhanced_documents_df['doc_id'] == doc_id].iloc[0]
                print(f"{i}. [{score:.4f}] {title}")
                print(f"   ID: {doc_id}")
                print(f"   Location: {doc['location']}")
                print(f"   Weather: {doc['event_type']}")
                print(f"   Vehicles: {doc['vehicle_count']}")
                print(f"   Congestion: {doc['congestion_level']}")
                print(f"   Time: {doc['timestamp']}")
            else:
                doc = self.documents_df[self.documents_df['doc_id'] == doc_id].iloc[0]
                print(f"{i}. [{score:.4f}] {title}")
                print(f"   ID: {doc_id}")
                print(f"   Location: {doc['location']}")
                print(f"   Weather: {doc['event_type']}")
                print(f"   Vehicles: {doc['vehicle_count']}")
                print(f"   Time: {doc['timestamp']}")
            print()
    
    def demo_comparison(self, query: str):
        """Demonstrate retriever comparison."""
        print(f"\n{'='*80}")
        print(f"🔍 RETRIEVAL MODEL COMPARISON for: '{query}'")
        print(f"{'='*80}")
        
        comparison_results = {}
        
        for retriever_name in self.retrievers.keys():
            retriever = self.retrievers[retriever_name]
            results = retriever.search(query, top_k=3)
            comparison_results[retriever_name] = results
            
            print(f"\n--- {retriever_name} ---")
            if results:
                for i, (doc_id, title, score) in enumerate(results, 1):
                    print(f"  {i}. [{score:.4f}] {title}")
            else:
                print("  No results found")
        
        # Calculate overlap
        doc_id_sets = {}
        for retriever_name, results in comparison_results.items():
            doc_id_sets[retriever_name] = {doc_id for doc_id, _, _ in results}
        
        print(f"\n--- Overlap Analysis ---")
        models = list(comparison_results.keys())
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model1, model2 = models[i], models[j]
                overlap = doc_id_sets[model1].intersection(doc_id_sets[model2])
                total = max(len(doc_id_sets[model1]), len(doc_id_sets[model2]))
                overlap_pct = (len(overlap) / total * 100) if total > 0 else 0
                print(f"📊 {model1} vs {model2}: {len(overlap)} shared documents ({overlap_pct:.1f}%)")
    
    def run_demo(self):
        """Run the complete demo."""
        print("\n" + "="*80)
        print("🎯 TRAFFIC EVENT SEARCH INTERFACE DEMO")
        print("="*80)
        print("This demo shows:")
        print("  • Multiple retrieval models (TF-IDF, BM25, Enhanced TF-IDF)")
        print("  • Search functionality with detailed results")
        print("  • Model comparison and overlap analysis")
        print("  • Professional result formatting")
        print("="*80)
        
        # Demo queries
        demo_queries = [
            ("heavy rain traffic congestion", "TF-IDF"),
            ("rush hour problems", "BM25"),
            ("weekend traffic flow", "Enhanced TF-IDF")
        ]
        
        # Demonstrate individual searches
        for query, model in demo_queries:
            self.demo_search(query, model)
        
        # Demonstrate comparison
        self.demo_comparison("heavy rain traffic congestion")
        
        print(f"\n{'='*80}")
        print("🎉 DEMO COMPLETE!")
        print("="*80)
        print("Features demonstrated:")
        print("✅ Multiple retrieval models working")
        print("✅ Professional search results with metadata")
        print("✅ Model comparison and analysis")
        print("✅ Clean, user-friendly interface")
        print("\nTo use interactively, run: python3 src/interface/interactive_search.py")

def main():
    """Run the demo."""
    demo = DemoSearchInterface()
    
    try:
        demo.initialize(sample_size=2000)
        demo.run_demo()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure all required files and dependencies are available.")

if __name__ == "__main__":
    main()

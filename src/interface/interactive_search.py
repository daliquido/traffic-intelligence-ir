import pandas as pd
import numpy as np
import sys
import os
from typing import List, Tuple, Dict

# Add parent directory to path to import retrieval systems
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.tfidf_retrieval import TFIDFRetriever
from retrieval.bm25_retrieval import BM25Retriever
from retrieval.simple_enhanced_retrieval import SimpleEnhancedRetriever
from preprocessing.clean_text import preprocess_documents

class InteractiveSearchInterface:
    """
    Interactive search interface for traffic event retrieval.
    """
    
    def __init__(self):
        self.retrievers = {}
        self.documents_df = None
        self.enhanced_documents_df = None
        self.initialized = False
    
    def initialize(self, sample_size: int = 10000):
        """Initialize all retrieval systems."""
        print("Initializing Interactive Search Interface...")
        print("="*60)
        
        # Load documents
        documents_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     'data', 'processed', 'traffic_documents.csv')
        self.documents_df = pd.read_csv(documents_path)
        
        # Use sample for faster loading
        sample_df = self.documents_df.sample(n=sample_size, random_state=42)
        print(f"Loaded {len(sample_df)} documents for interactive search")
        
        # Preprocess documents for TF-IDF and BM25
        processed_df = preprocess_documents(sample_df)
        
        # Initialize TF-IDF retriever
        print("\nSetting up TF-IDF retriever...")
        tfidf_retriever = TFIDFRetriever()
        tfidf_retriever.fit(processed_df)
        self.retrievers['TF-IDF'] = tfidf_retriever
        
        # Initialize BM25 retriever
        print("Setting up BM25 retriever...")
        bm25_retriever = BM25Retriever()
        bm25_retriever.fit(processed_df)
        self.retrievers['BM25'] = bm25_retriever
        
        # Initialize Enhanced TF-IDF retriever
        print("Setting up Enhanced TF-IDF retriever...")
        enhanced_retriever = SimpleEnhancedRetriever()
        self.enhanced_documents_df = enhanced_retriever.create_enhanced_documents(sample_df)
        enhanced_retriever.fit(self.enhanced_documents_df)
        self.retrievers['Enhanced TF-IDF'] = enhanced_retriever
        
        self.initialized = True
        print(f"\n✅ All {len(self.retrievers)} retrievers initialized successfully!")
        print("Available models: TF-IDF, BM25, Enhanced TF-IDF")
    
    def search(self, query: str, retriever_name: str = 'TF-IDF', top_k: int = 5) -> List[Dict]:
        """Search for documents using specified retriever."""
        if not self.initialized:
            raise ValueError("Interface not initialized. Call initialize() first.")
        
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not available. Options: {list(self.retrievers.keys())}")
        
        retriever = self.retrievers[retriever_name]
        results = retriever.search(query, top_k)
        
        # Convert to detailed format
        detailed_results = []
        for doc_id, title, score in results:
            # Get document details
            if 'Enhanced' in retriever_name:
                doc = self.enhanced_documents_df[self.enhanced_documents_df['doc_id'] == doc_id].iloc[0]
                details = {
                    'doc_id': doc['doc_id'],
                    'title': doc['title'],
                    'score': score,
                    'timestamp': doc['timestamp'],
                    'event_type': doc['event_type'],
                    'location': doc['location'],
                    'vehicle_count': doc['vehicle_count'],
                    'congestion_level': doc['congestion_level']
                }
            else:
                doc = self.documents_df[self.documents_df['doc_id'] == doc_id].iloc[0]
                details = {
                    'doc_id': doc['doc_id'],
                    'title': doc['title'],
                    'score': score,
                    'timestamp': doc['timestamp'],
                    'event_type': doc['event_type'],
                    'location': doc['location'],
                    'vehicle_count': doc['vehicle_count']
                }
            
            detailed_results.append(details)
        
        return detailed_results
    
    def compare_retrievers(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """Compare results across all retrievers."""
        comparison = {}
        for retriever_name in self.retrievers.keys():
            results = self.search(query, retriever_name, top_k)
            comparison[retriever_name] = results
        return comparison
    
    def print_results(self, query: str, results: List[Dict], retriever_name: str = "TF-IDF"):
        """Print search results in a nice format."""
        print(f"\n{'='*80}")
        print(f"🔍 {retriever_name} Results for: '{query}'")
        print(f"{'='*80}")
        
        if not results:
            print("❌ No relevant documents found.")
            return
        
        print(f"📊 Found {len(results)} relevant documents:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. 📄 [{result['score']:.4f}] {result['title']}")
            print(f"   🆔 ID: {result['doc_id']}")
            print(f"   📍 Location: {result['location']}")
            print(f"   🌤️  Weather: {result['event_type']}")
            print(f"   🚗 Vehicle Count: {result['vehicle_count']}")
            if 'congestion_level' in result:
                print(f"   🚦 Congestion: {result['congestion_level']}")
            print(f"   ⏰ Time: {result['timestamp']}")
            print()
    
    def print_comparison(self, query: str, comparison: Dict[str, List[Dict]]):
        """Print comparison across retrievers."""
        print(f"\n{'='*80}")
        print(f"🔍 RETRIEVAL COMPARISON for: '{query}'")
        print(f"{'='*80}")
        
        for retriever_name, results in comparison.items():
            print(f"\n--- {retriever_name} ---")
            if results:
                print(f"Top {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. [{result['score']:.4f}] {result['title']}")
            else:
                print("  No results found")
        
        # Calculate overlap
        all_doc_ids = {}
        for retriever_name, results in comparison.items():
            all_doc_ids[retriever_name] = {r['doc_id'] for r in results}
        
        print(f"\n--- Overlap Analysis ---")
        retriever_names = list(comparison.keys())
        for i in range(len(retriever_names)):
            for j in range(i+1, len(retriever_names)):
                name1, name2 = retriever_names[i], retriever_names[j]
                overlap = all_doc_ids[name1].intersection(all_doc_ids[name2])
                total = max(len(all_doc_ids[name1]), len(all_doc_ids[name2]))
                overlap_pct = (len(overlap) / total * 100) if total > 0 else 0
                print(f"{name1} vs {name2}: {len(overlap)} shared ({overlap_pct:.1f}%)")
    
    def interactive_mode(self):
        """Run interactive search mode."""
        print("\n" + "="*80)
        print("🚀 INTERACTIVE TRAFFIC EVENT SEARCH")
        print("="*80)
        print("Commands:")
        print("  • Type your search query (e.g., 'heavy rain traffic')")
        print("  • 'compare <query>' - Compare all retrievers")
        print("  • 'models' - Show available retrieval models")
        print("  • 'help' - Show this help")
        print("  • 'quit' or 'exit' - Exit the interface")
        print("="*80)
        
        while True:
            try:
                user_input = input("\n🔍 Enter your search query or command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using Traffic Event Search!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  • <query> - Search using TF-IDF (default)")
                    print("  • 'compare <query>' - Compare all retrievers")
                    print("  • 'models' - Show available retrieval models")
                    print("  • 'help' - Show this help")
                    print("  • 'quit' - Exit the interface")
                    continue
                
                if user_input.lower() == 'models':
                    print(f"\n🤖 Available retrieval models:")
                    for i, model in enumerate(self.retrievers.keys(), 1):
                        print(f"  {i}. {model}")
                    continue
                
                if user_input.lower().startswith('compare '):
                    query = user_input[8:].strip()
                    if not query:
                        print("❌ Please provide a query after 'compare'")
                        continue
                    
                    comparison = self.compare_retrievers(query, top_k=3)
                    self.print_comparison(query, comparison)
                    continue
                
                # Regular search
                if not user_input:
                    continue
                
                # Default search with TF-IDF
                results = self.search(user_input, 'TF-IDF', top_k=5)
                self.print_results(user_input, results, 'TF-IDF')
                
            except KeyboardInterrupt:
                print("\n\n👋 Search interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

def main():
    """Main function to run the interactive interface."""
    interface = InteractiveSearchInterface()
    
    try:
        # Initialize the interface
        interface.initialize(sample_size=5000)  # Smaller sample for faster loading
        
        # Start interactive mode
        interface.interactive_mode()
        
    except Exception as e:
        print(f"❌ Failed to initialize interface: {e}")
        print("Make sure all required files are in place and dependencies are installed.")

if __name__ == "__main__":
    main()

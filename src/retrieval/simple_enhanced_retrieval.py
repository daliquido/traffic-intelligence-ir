import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import sys
import os

# Add parent directory to path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.clean_text import preprocess_documents

class SimpleEnhancedRetriever:
    """
    Simple enhanced TF-IDF retrieval with better document wording.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents_df = None
        self.fitted = False
    
    def create_enhanced_documents(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced documents with better wording."""
        enhanced_docs = []
        
        print("Creating enhanced documents with better wording...")
        
        for idx, row in original_df.iterrows():
            # Extract original information
            vehicle_count = row.get('vehicle_count', 0)
            event_type = row.get('event_type', 'Unknown')
            location = row.get('location', 'road')
            timestamp = row.get('timestamp', '')
            
            # Enhanced congestion labeling
            if vehicle_count > 500:
                congestion = "Severe Traffic Congestion Heavy Traffic Backup"
            elif vehicle_count > 200:
                congestion = "Moderate Traffic Congestion Traffic Delays"
            elif vehicle_count > 50:
                congestion = "Light Traffic Conditions Minor Congestion"
            else:
                congestion = "Free Traffic Flow Clear Traffic Normal"
            
            # Enhanced weather labeling
            if event_type == "Heavy Rain":
                weather = "Heavy Rainfall Storm Severe Rain Downpour"
            elif event_type == "Rain":
                weather = "Rain Weather Precipitation Wet Conditions"
            elif event_type == "Clear":
                weather = "Clear Weather Sunny Dry Good Conditions"
            elif event_type == "Clouds":
                weather = "Cloudy Overcast Cloud Cover Gray Sky"
            else:
                weather = "Weather Conditions Atmospheric"
            
            # Enhanced location labeling
            if "secondary" in location.lower():
                location_enhanced = "Secondary Road Local Street Neighborhood Road"
            elif "residential" in location.lower():
                location_enhanced = "Residential Road Housing Area Home Street"
            elif "primary" in location.lower():
                location_enhanced = "Primary Road Main Street Major Arterial"
            elif "service" in location.lower():
                location_enhanced = "Service Road Access Utility Road"
            else:
                location_enhanced = f"{location} Road Street"
            
            # Create enhanced searchable text
            enhanced_text = f"{congestion} {weather} {location_enhanced} traffic event"
            
            # Create enhanced title
            enhanced_title = f"{congestion} - {weather}"
            
            doc = {
                'doc_id': f"enhanced_{idx}",
                'title': enhanced_title,
                'text': enhanced_text,
                'timestamp': timestamp,
                'event_type': event_type,
                'location': location,
                'vehicle_count': vehicle_count,
                'congestion_level': self._get_congestion_level(vehicle_count)
            }
            
            enhanced_docs.append(doc)
        
        enhanced_df = pd.DataFrame(enhanced_docs)
        print(f"Created {len(enhanced_df)} enhanced documents")
        
        return enhanced_df
    
    def _get_congestion_level(self, vehicle_count: int) -> str:
        """Get congestion level."""
        if vehicle_count > 500:
            return "severe"
        elif vehicle_count > 200:
            return "moderate"
        elif vehicle_count > 50:
            return "light"
        else:
            return "free"
    
    def fit(self, documents_df: pd.DataFrame):
        """Fit the enhanced TF-IDF model."""
        print("Fitting Simple Enhanced TF-IDF model...")
        self.documents_df = documents_df.copy()
        
        # Create TF-IDF vectorizer with enhanced parameters
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            lowercase=True
        )
        
        # Fit on enhanced text
        self.tfidf_matrix = self.vectorizer.fit_transform(documents_df['text'])
        
        self.fitted = True
        print(f"Enhanced TF-IDF model fitted:")
        print(f"  Vocabulary: {len(self.vectorizer.vocabulary_)} terms")
        print(f"  Document matrix: {self.tfidf_matrix.shape}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Search for relevant documents."""
        if not self.fitted:
            raise ValueError("Model must be fitted before searching.")
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc_id = self.documents_df.iloc[idx]['doc_id']
                title = self.documents_df.iloc[idx]['title']
                score = similarities[idx]
                
                results.append((doc_id, title, score))
        
        return results
    
    def print_results(self, query: str, results: List[Tuple[str, str, float]]):
        """Print search results."""
        print(f"\n=== Enhanced TF-IDF Results for: '{query}' ===")
        print(f"Found {len(results)} relevant documents\n")
        
        for i, (doc_id, title, score) in enumerate(results, 1):
            print(f"{i}. [{score:.4f}] {title}")
            print(f"   ID: {doc_id}")
            
            # Get document details
            doc = self.documents_df[self.documents_df['doc_id'] == doc_id].iloc[0]
            print(f"   Location: {doc['location']}")
            print(f"   Congestion: {doc['congestion_level']}")
            print(f"   Weather: {doc['event_type']}")
            
            # Show text preview
            text_preview = doc['text'][:120] + "..." if len(doc['text']) > 120 else doc['text']
            print(f"   Preview: {text_preview}")
            print()

def compare_baseline_vs_enhanced(baseline_retriever, enhanced_retriever, queries: List[str], top_k: int = 5):
    """Compare baseline with enhanced system."""
    print("=" * 80)
    print("BASELINE vs ENHANCED TF-IDF COMPARISON")
    print("=" * 80)
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: '{query}'")
        print(f"{'='*60}")
        
        # Get baseline results
        baseline_results = baseline_retriever.search(query, top_k)
        print(f"\n--- Baseline TF-IDF Results ---")
        if baseline_results:
            for i, (doc_id, title, score) in enumerate(baseline_results, 1):
                print(f"{i}. [{score:.4f}] {title}")
        else:
            print("No results found")
        
        # Get enhanced results
        enhanced_results = enhanced_retriever.search(query, top_k)
        print(f"\n--- Enhanced TF-IDF Results ---")
        if enhanced_results:
            for i, (doc_id, title, score) in enumerate(enhanced_results, 1):
                print(f"{i}. [{score:.4f}] {title}")
        else:
            print("No results found")
        
        # Calculate overlap
        baseline_ids = {doc_id for doc_id, _, _ in baseline_results}
        enhanced_ids = {doc_id for doc_id, _, _ in enhanced_results}
        overlap = baseline_ids.intersection(enhanced_ids)
        
        print(f"\n--- Comparison ---")
        print(f"Baseline unique: {len(baseline_ids - enhanced_ids)} documents")
        print(f"Enhanced unique: {len(enhanced_ids - baseline_ids)} documents")
        print(f"Shared: {len(overlap)} documents ({len(overlap)/max(len(baseline_ids), len(enhanced_ids))*100:.1f}%)")
        
        # Score comparison
        if baseline_results and enhanced_results:
            baseline_avg = np.mean([score for _, _, score in baseline_results])
            enhanced_avg = np.mean([score for _, _, score in enhanced_results])
            print(f"Average score - Baseline: {baseline_avg:.4f}, Enhanced: {enhanced_avg:.4f}")

def main():
    """Test enhanced retrieval system."""
    # Load documents
    documents_path = "/Users/i/traffic-intelligence-ir/data/processed/traffic_documents.csv"
    original_df = pd.read_csv(documents_path)
    
    # Use sample for testing
    sample_df = original_df.sample(n=10000, random_state=42)
    print(f"Using sample of {len(sample_df)} documents")
    
    # Create baseline retriever
    print("\nSetting up baseline TF-IDF...")
    from tfidf_retrieval import TFIDFRetriever
    
    processed_df = preprocess_documents(sample_df)
    baseline_retriever = TFIDFRetriever()
    baseline_retriever.fit(processed_df)
    
    # Create enhanced retriever
    print("\nSetting up enhanced TF-IDF...")
    enhanced_retriever = SimpleEnhancedRetriever()
    enhanced_df = enhanced_retriever.create_enhanced_documents(sample_df)
    enhanced_retriever.fit(enhanced_df)
    
    # Test with sample queries
    test_queries = [
        "heavy rain traffic congestion",
        "rush hour problems",
        "weekend traffic flow"
    ]
    
    # Test enhanced retriever
    print("\n" + "="*80)
    print("ENHANCED TF-IDF SYSTEM TEST")
    print("="*80)
    
    for query in test_queries:
        results = enhanced_retriever.search(query, top_k=3)
        enhanced_retriever.print_results(query, results)
        print("-" * 60)
    
    # Compare with baseline
    print("\n\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    
    comparison_queries = ["heavy rain traffic congestion", "rush hour problems"]
    compare_baseline_vs_enhanced(baseline_retriever, enhanced_retriever, comparison_queries)

if __name__ == "__main__":
    main()

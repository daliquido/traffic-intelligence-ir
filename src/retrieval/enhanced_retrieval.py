import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import sys
import os

# Add parent directory to path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.clean_text import tokenize
from preprocessing.enhanced_documents import EnhancedDocumentGenerator

class EnhancedTFIDFRetriever:
    """
    Enhanced TF-IDF retrieval with field weighting and improved documents.
    """
    
    def __init__(self, title_weight: float = 2.0, body_weight: float = 1.0):
        """
        Initialize enhanced TF-IDF retriever with field weighting.
        
        Args:
            title_weight (float): Weight multiplier for title field
            body_weight (float): Weight multiplier for body field
        """
        self.title_weight = title_weight
        self.body_weight = body_weight
        self.title_vectorizer = None
        self.body_vectorizer = None
        self.title_matrix = None
        self.body_matrix = None
        self.documents_df = None
        self.fitted = False
    
    def fit(self, documents_df: pd.DataFrame):
        """
        Fit the enhanced TF-IDF model with field weighting.
        
        Args:
            documents_df (pd.DataFrame): DataFrame with enhanced documents
        """
        print("Fitting Enhanced TF-IDF model with field weighting...")
        self.documents_df = documents_df.copy()
        
        # Create separate vectorizers for title and body
        self.title_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 1),
            min_df=1,
            max_df=0.95,
            lowercase=True
        )
        
        self.body_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english', 
            ngram_range=(1, 1),
            min_df=1,
            max_df=0.95,
            lowercase=True
        )
        
        # Fit on title and body separately
        print("Fitting on title field...")
        self.title_matrix = self.title_vectorizer.fit_transform(documents_df['title'])
        
        print("Fitting on body field...")
        self.body_matrix = self.body_vectorizer.fit_transform(documents_df['body'])
        
        self.fitted = True
        print(f"Enhanced TF-IDF model fitted:")
        print(f"  Title vocabulary: {len(self.title_vectorizer.vocabulary_)} terms")
        print(f"  Body vocabulary: {len(self.body_vectorizer.vocabulary_)} terms")
        print(f"  Field weights: title={self.title_weight}, body={self.body_weight}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Search with field-weighted TF-IDF.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[str, str, float]]: List of (doc_id, title, score) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before searching. Call fit() first.")
        
        # Transform query for both fields
        query_title_vec = self.title_vectorizer.transform([query])
        query_body_vec = self.body_vectorizer.transform([query])
        
        # Calculate similarities for both fields
        title_similarities = cosine_similarity(query_title_vec, self.title_matrix).flatten()
        body_similarities = cosine_similarity(query_body_vec, self.body_matrix).flatten()
        
        # Combine with field weights
        combined_scores = (self.title_weight * title_similarities + 
                          self.body_weight * body_similarities)
        
        # Get top-k most similar documents
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0:
                doc_id = self.documents_df.iloc[idx]['doc_id']
                title = self.documents_df.iloc[idx]['title']
                score = combined_scores[idx]
                
                results.append((doc_id, title, score))
        
        return results
    
    def get_document_details(self, doc_id: str) -> Dict:
        """Get detailed document information."""
        if not self.fitted:
            raise ValueError("Model must be fitted first.")
        
        doc = self.documents_df[self.documents_df['doc_id'] == doc_id]
        if len(doc) == 0:
            return {"error": "Document not found"}
        
        doc = doc.iloc[0]
        return {
            'doc_id': doc['doc_id'],
            'title': doc['title'],
            'body': doc['body'],
            'timestamp': doc['timestamp'],
            'event_type': doc['event_type'],
            'location_type': doc['location_type'],
            'congestion_level': doc['congestion_level'],
            'weather_severity': doc['weather_severity']
        }
    
    def print_results(self, query: str, results: List[Tuple[str, str, float]]):
        """Print enhanced search results."""
        print(f"\n=== Enhanced TF-IDF Results for: '{query}' ===")
        print(f"Found {len(results)} relevant documents\n")
        
        for i, (doc_id, title, score) in enumerate(results, 1):
            print(f"{i}. [{score:.4f}] {title}")
            print(f"   ID: {doc_id}")
            
            # Get document details
            details = self.get_document_details(doc_id)
            if 'error' not in details:
                print(f"   Location: {details['location_type']}")
                print(f"   Congestion: {details['congestion_level']}")
                print(f"   Weather: {details['event_type']} ({details['weather_severity']})")
                
                # Show body preview
                body_preview = details['body'][:120] + "..." if len(details['body']) > 120 else details['body']
                print(f"   Preview: {body_preview}")
            print()

def compare_baseline_vs_enhanced(baseline_retriever, enhanced_retriever, queries: List[str], top_k: int = 5):
    """
    Compare baseline TF-IDF with enhanced TF-IDF.
    
    Args:
        baseline_retriever: Original TF-IDF retriever
        enhanced_retriever: Enhanced TF-IDF retriever
        queries (List[str]): Test queries
        top_k (int): Number of results to compare
    """
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
    # Load original documents
    documents_path = "/Users/i/traffic-intelligence-ir/data/processed/traffic_documents.csv"
    original_df = pd.read_csv(documents_path)
    
    # Use sample for testing
    sample_df = original_df.sample(n=10000, random_state=42)
    print(f"Using sample of {len(sample_df)} documents")
    
    # Create enhanced documents
    print("\nCreating enhanced documents...")
    generator = EnhancedDocumentGenerator()
    enhanced_df = generator.create_enhanced_documents(sample_df)
    
    # Create baseline retriever for comparison
    print("\nSetting up baseline TF-IDF...")
    from tfidf_retrieval import TFIDFRetriever
    from preprocessing.clean_text import preprocess_documents
    
    processed_df = preprocess_documents(sample_df)
    baseline_retriever = TFIDFRetriever()
    baseline_retriever.fit(processed_df)
    
    # Create enhanced retriever
    print("\nSetting up enhanced TF-IDF...")
    enhanced_retriever = EnhancedTFIDFRetriever(title_weight=2.0, body_weight=1.0)
    enhanced_retriever.fit(enhanced_df)
    
    # Test with sample queries
    test_queries = [
        "heavy rain traffic congestion",
        "rush hour road problems",
        "weekend traffic flow",
        "accidents on main roads"
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
    
    comparison_queries = ["heavy rain traffic congestion", "rush hour road problems"]
    compare_baseline_vs_enhanced(baseline_retriever, enhanced_retriever, comparison_queries)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import sys
import os

# Add parent directory to path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.clean_text import clean_text, preprocess_documents
from preprocessing.query_processor import QueryProcessor

class TFIDFRetriever:
    """
    TF-IDF based Information Retrieval system for traffic events.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            lowercase=True
        )
        self.tfidf_matrix = None
        self.documents_df = None
        self.fitted = False
        self.query_processor = QueryProcessor()
    
    def fit(self, documents_df: pd.DataFrame, text_column: str = 'searchable_text'):
        """
        Fit the TF-IDF model on the documents.
        
        Args:
            documents_df (pd.DataFrame): DataFrame with text column
            text_column (str): Column name to use for TF-IDF
        """
        print("Fitting TF-IDF model...")
        self.documents_df = documents_df.copy()
        
        # Fit TF-IDF vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(documents_df[text_column])
        
        self.fitted = True
        print(f"TF-IDF model fitted with {len(self.vectorizer.vocabulary_)} terms")
        print(f"Document matrix shape: {self.tfidf_matrix.shape}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Search for relevant documents given a query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[str, str, float]]: List of (doc_id, title, score) tuples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before searching. Call fit() first.")
        
        # Preprocess query using the same pipeline as documents
        query_result = self.query_processor.preprocess_query(query)
        processed_query = query_result['processed_query']
        
        # Transform query to TF-IDF space
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc_id = self.documents_df.iloc[idx]['doc_id']
                title = self.documents_df.iloc[idx]['title']
                score = similarities[idx]
                
                results.append((doc_id, title, score))
        
        return results
    
    def get_document_text(self, doc_id: str) -> str:
        """
        Get the full text of a document.
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            str: Document text
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first.")
        
        doc = self.documents_df[self.documents_df['doc_id'] == doc_id]
        if len(doc) == 0:
            return "Document not found"
        
        return doc.iloc[0]['text']
    
    def print_results(self, query: str, results: List[Tuple[int, str, float]]):
        """
        Print search results in a readable format.
        
        Args:
            query (str): Original query
            results (List[Tuple[int, str, float]]): Search results
        """
        print(f"\n=== Search Results for: '{query}' ===")
        print(f"Found {len(results)} relevant documents\n")
        
        for i, (doc_id, title, score) in enumerate(results, 1):
            print(f"{i}. [{score:.4f}] {title}")
            print(f"   ID: {doc_id}")
            
            # Get and display first part of document text
            full_text = self.get_document_text(doc_id)
            preview = full_text[:100] + "..." if len(full_text) > 100 else full_text
            print(f"   Preview: {preview}")
            print()

def main():
    """
    Test the TF-IDF retrieval system.
    """
    # Load processed documents
    documents_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'data', 'processed', 'traffic_documents.csv')
    
    print("Loading documents...")
    documents_df = pd.read_csv(documents_path)
    
    # For testing, use a smaller sample
    sample_df = documents_df.sample(n=10000, random_state=42)
    print(f"Using sample of {len(sample_df)} documents for testing")
    
    # Preprocess documents with enhanced pipeline
    from preprocessing.clean_text import preprocess_documents
    processed_df = preprocess_documents(sample_df)
    
    # Use searchable_text (stemmed and stop-word removed) for TF-IDF
    retriever = TFIDFRetriever()
    retriever.fit(processed_df)
    
    # Test queries
    test_queries = [
        "rain traffic congestion",
        "heavy rain road conditions",
        "rush hour traffic",
        "weekend traffic flow",
        "secondary road accident"
    ]
    
    for query in test_queries:
        results = retriever.search(query, top_k=5)
        retriever.print_results(query, results)
        print("-" * 80)

if __name__ == "__main__":
    main()
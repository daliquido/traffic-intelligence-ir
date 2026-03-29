import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.clean_text import tokenize
from preprocessing.query_processor import QueryProcessor

class BM25Retriever:
    """
    BM25 based Information Retrieval system for traffic events.
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25_model = None
        self.documents_df = None
        self.valid_row_indices = []  # FIX: prebuilt mapping
        self.fitted = False
        self.query_processor = QueryProcessor()
    
    def fit(self, documents_df: pd.DataFrame, token_column: str = 'tokens'):
        print("Fitting BM25 model...")
        self.documents_df = documents_df.copy().reset_index(drop=True)  # FIX: reset index
        
        token_lists = self.documents_df[token_column].tolist()
        
        # FIX: build mapping ONCE at fit time, not on every search
        self.valid_row_indices = [i for i, tokens in enumerate(token_lists) if tokens]
        valid_token_lists = [token_lists[i] for i in self.valid_row_indices]
        
        self.bm25_model = BM25Okapi(valid_token_lists, k1=self.k1, b=self.b)
        
        self.fitted = True
        print(f"BM25 model fitted with {len(valid_token_lists)} documents")
        print(f"Vocabulary size: {len(self.bm25_model.idf)} unique terms")
        print(f"Parameters: k1={self.k1}, b={self.b}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        if not self.fitted:
            raise ValueError("Model must be fitted before searching. Call fit() first.")
        
        query_result = self.query_processor.preprocess_query(query)
        query_tokens = query_result['tokens']
        
        if not query_tokens:
            return []
        
        scores = self.bm25_model.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_row_idx = self._get_document_row_index(idx)
            if doc_row_idx is not None:
                doc_id = self.documents_df.iloc[doc_row_idx]['doc_id']
                title = self.documents_df.iloc[doc_row_idx]['title']
                score = scores[idx]
                if score > 0:
                    results.append((doc_id, title, score))
        
        return results
    
    def _get_document_row_index(self, model_index: int):
        # FIX: use prebuilt mapping instead of rebuilding every call
        if model_index < len(self.valid_row_indices):
            return self.valid_row_indices[model_index]
        return None
    
    def get_document_text(self, doc_id: str) -> str:
        if not self.fitted:
            raise ValueError("Model must be fitted first.")
        doc = self.documents_df[self.documents_df['doc_id'] == doc_id]
        if len(doc) == 0:
            return "Document not found"
        return doc.iloc[0]['text']
    
    def print_results(self, query: str, results: List[Tuple[str, str, float]]):
        print(f"\n=== BM25 Search Results for: '{query}' ===")
        print(f"Found {len(results)} relevant documents\n")
        for i, (doc_id, title, score) in enumerate(results, 1):
            print(f"{i}. [{score:.4f}] {title}")
            print(f"   ID: {doc_id}")
            full_text = self.get_document_text(doc_id)
            preview = full_text[:100] + "..." if len(full_text) > 100 else full_text
            print(f"   Preview: {preview}")
            print()


def compare_tfidf_vs_bm25(tfidf_retriever, bm25_retriever, queries: List[str], top_k: int = 5):
    print("=" * 80)
    print("TF-IDF vs BM25 COMPARISON")
    print("=" * 80)
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: '{query}'")
        print(f"{'='*60}")
        
        tfidf_results = tfidf_retriever.search(query, top_k)
        print(f"\n--- TF-IDF Results ---")
        if tfidf_results:
            for i, (doc_id, title, score) in enumerate(tfidf_results, 1):
                print(f"{i}. [{score:.4f}] {title}")
        else:
            print("No results found")
        
        bm25_results = bm25_retriever.search(query, top_k)
        print(f"\n--- BM25 Results ---")
        if bm25_results:
            for i, (doc_id, title, score) in enumerate(bm25_results, 1):
                print(f"{i}. [{score:.4f}] {title}")
        else:
            print("No results found")
        
        tfidf_ids = {doc_id for doc_id, _, _ in tfidf_results}
        bm25_ids = {doc_id for doc_id, _, _ in bm25_results}
        overlap = tfidf_ids.intersection(bm25_ids)
        
        print(f"\n--- Overlap Analysis ---")
        print(f"TF-IDF unique: {len(tfidf_ids - bm25_ids)} documents")
        print(f"BM25 unique: {len(bm25_ids - tfidf_ids)} documents")
        print(f"Shared: {len(overlap)} documents ({len(overlap)/max(len(tfidf_ids), len(bm25_ids))*100:.1f}%)")


def main():
    documents_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'data', 'processed', 'traffic_documents.csv')
    
    print("Loading documents...")
    documents_df = pd.read_csv(documents_path)
    sample_df = documents_df.sample(n=10000, random_state=42)
    print(f"Using sample of {len(sample_df)} documents for testing")
    
    from preprocessing.clean_text import preprocess_documents
    processed_df = preprocess_documents(sample_df)
    
    print("\n=== BM25 Retrieval System ===")
    bm25_retriever = BM25Retriever(k1=1.2, b=0.75)
    bm25_retriever.fit(processed_df)
    
    test_queries = [
        "rain traffic congestion",
        "heavy rain road conditions", 
        "rush hour traffic",
        "weekend traffic flow",
        "secondary road accident"
    ]
    
    for query in test_queries:
        results = bm25_retriever.search(query, top_k=5)
        bm25_retriever.print_results(query, results)
        print("-" * 80)
    
    print("\n\n=== COMPARING WITH TF-IDF ===")
    from tfidf_retrieval import TFIDFRetriever
    tfidf_retriever = TFIDFRetriever()
    tfidf_retriever.fit(processed_df)
    
    comparison_queries = ["heavy rain road conditions", "rush hour traffic", "weekend traffic flow"]
    compare_tfidf_vs_bm25(tfidf_retriever, bm25_retriever, comparison_queries)

if __name__ == "__main__":
    main()
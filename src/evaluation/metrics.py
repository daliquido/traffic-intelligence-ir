import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# Add parent directory to path to import retrieval systems
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.tfidf_retrieval import TFIDFRetriever
from retrieval.bm25_retrieval import BM25Retriever
from preprocessing.clean_text import preprocess_documents

class IREvaluator:
    """
    Information Retrieval evaluation system with standard metrics.
    """
    
    def __init__(self, queries_file: str, qrels_file: str):
        """
        Initialize evaluator with queries and relevance judgments.
        
        Args:
            queries_file (str): Path to queries.csv
            qrels_file (str): Path to qrels.csv
        """
        self.queries_df = pd.read_csv(queries_file)
        self.qrels_df = pd.read_csv(qrels_file)
        
        # Convert qrels to dictionary for faster lookup
        self.qrels_dict = {}
        for _, row in self.qrels_df.iterrows():
            query_id = row['query_id']
            doc_id = row['doc_id']
            relevance = row['relevance']
            
            if query_id not in self.qrels_dict:
                self.qrels_dict[query_id] = {}
            self.qrels_dict[query_id][doc_id] = relevance
        
        print(f"Loaded {len(self.queries_df)} queries and {len(self.qrels_df)} relevance judgments")
    
    def precision_at_k(self, retrieved_docs: List[str], query_id: str, k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved_docs (List[str]): List of retrieved document IDs
            query_id (str): Query identifier
            k (int): Cut-off position
            
        Returns:
            float: Precision@K score
        """
        if query_id not in self.qrels_dict or k == 0:
            return 0.0
        
        # Get top-k retrieved documents
        top_k_docs = retrieved_docs[:k]
        
        # Count relevant documents in top-k
        relevant_count = 0
        for doc_id in top_k_docs:
            if doc_id in self.qrels_dict[query_id] and self.qrels_dict[query_id][doc_id] > 0:
                relevant_count += 1
        
        return relevant_count / k
    
    def recall_at_k(self, retrieved_docs: List[str], query_id: str, k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved_docs (List[str]): List of retrieved document IDs
            query_id (str): Query identifier
            k (int): Cut-off position
            
        Returns:
            float: Recall@K score
        """
        if query_id not in self.qrels_dict:
            return 0.0
        
        # Get total number of relevant documents for this query
        total_relevant = sum(1 for rel in self.qrels_dict[query_id].values() if rel > 0)
        if total_relevant == 0:
            return 0.0
        
        # Get top-k retrieved documents
        top_k_docs = retrieved_docs[:k]
        
        # Count relevant documents in top-k
        relevant_retrieved = 0
        for doc_id in top_k_docs:
            if doc_id in self.qrels_dict[query_id] and self.qrels_dict[query_id][doc_id] > 0:
                relevant_retrieved += 1
        
        return relevant_retrieved / total_relevant
    
    def average_precision(self, retrieved_docs: List[str], query_id: str) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            retrieved_docs (List[str]): List of retrieved document IDs
            query_id (str): Query identifier
            
        Returns:
            float: Average Precision score
        """
        if query_id not in self.qrels_dict:
            return 0.0
        
        # Get relevant documents for this query
        relevant_docs = {doc_id for doc_id, rel in self.qrels_dict[query_id].items() if rel > 0}
        if not relevant_docs:
            return 0.0
        
        # Calculate average precision
        precisions = []
        relevant_found = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precisions.append(precision_at_i)
        
        return sum(precisions) / len(relevant_docs) if precisions else 0.0
    
    def dcg_at_k(self, retrieved_docs: List[str], query_id: str, k: int) -> float:
        """
        Calculate Discounted Cumulative Gain@K.
        
        Args:
            retrieved_docs (List[str]): List of retrieved document IDs
            query_id (str): Query identifier
            k (int): Cut-off position
            
        Returns:
            float: DCG@K score
        """
        if query_id not in self.qrels_dict:
            return 0.0
        
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            relevance = self.qrels_dict[query_id].get(doc_id, 0)
            if relevance > 0:
                dcg += (2**relevance - 1) / np.log2(i + 2)
        
        return dcg
    
    def ndcg_at_k(self, retrieved_docs: List[str], query_id: str, k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            retrieved_docs (List[str]): List of retrieved document IDs
            query_id (str): Query identifier
            k (int): Cut-off position
            
        Returns:
            float: nDCG@K score
        """
        # Calculate DCG@K
        dcg = self.dcg_at_k(retrieved_docs, query_id, k)
        
        # Calculate IDCG@K (Ideal DCG)
        if query_id not in self.qrels_dict:
            return 0.0
        
        # Get all relevant documents sorted by relevance
        relevant_docs = [(doc_id, rel) for doc_id, rel in self.qrels_dict[query_id].items() if rel > 0]
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        
        ideal_retrieved = [doc_id for doc_id, _ in relevant_docs]
        idcg = self.dcg_at_k(ideal_retrieved, query_id, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_retriever(self, retriever, retriever_name: str, k_values: List[int] = [5, 10]) -> pd.DataFrame:
        """
        Evaluate a retrieval system on all metrics.
        
        Args:
            retriever: Fitted retriever with search method
            retriever_name (str): Name of the retriever
            k_values (List[int]): K values for evaluation
            
        Returns:
            pd.DataFrame: Evaluation results
        """
        results = []
        
        for _, query_row in self.queries_df.iterrows():
            query_id = query_row['query_id']
            query_text = query_row['query_text']
            
            # Get retrieved documents
            retrieved_docs = []
            search_results = retriever.search(query_text, top_k=max(k_values))
            
            for doc_id, _, _ in search_results:
                retrieved_docs.append(doc_id)
            
            # Calculate metrics for each K
            for k in k_values:
                precision = self.precision_at_k(retrieved_docs, query_id, k)
                recall = self.recall_at_k(retrieved_docs, query_id, k)
                ndcg = self.ndcg_at_k(retrieved_docs, query_id, k)
                
                results.append({
                    'query_id': query_id,
                    'query_text': query_text,
                    'retriever': retriever_name,
                    'k': k,
                    'precision_at_k': precision,
                    'recall_at_k': recall,
                    'ndcg_at_k': ndcg
                })
            
            # Calculate AP (independent of K)
            ap = self.average_precision(retrieved_docs, query_id)
            results.append({
                'query_id': query_id,
                'query_text': query_text,
                'retriever': retriever_name,
                'k': 'AP',
                'precision_at_k': ap,
                'recall_at_k': ap,
                'ndcg_at_k': ap
            })
        
        return pd.DataFrame(results)
    
    def compare_retrievers(self, retrievers: Dict[str, object], k_values: List[int] = [5, 10]) -> pd.DataFrame:
        """
        Compare multiple retrieval systems.
        
        Args:
            retrievers (Dict[str, object]): Dictionary of retriever_name -> retriever
            k_values (List[int]): K values for evaluation
            
        Returns:
            pd.DataFrame: Comparison results
        """
        all_results = []
        
        for retriever_name, retriever in retrievers.items():
            print(f"\nEvaluating {retriever_name}...")
            results = self.evaluate_retriever(retriever, retriever_name, k_values)
            all_results.append(results)
        
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Calculate summary statistics
        summary = combined_results.groupby(['retriever', 'k']).agg({
            'precision_at_k': ['mean', 'std'],
            'recall_at_k': ['mean', 'std'],
            'ndcg_at_k': ['mean', 'std']
        }).round(4)
        
        return combined_results, summary

def main():
    """
    Main evaluation script comparing TF-IDF and BM25.
    """
    # Load documents and preprocess
    documents_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'data', 'processed', 'traffic_documents.csv')
    queries_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               'data', 'queries', 'queries.csv')
    qrels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'data', 'queries', 'qrels.csv')
    
    print("Loading documents and evaluation data...")
    documents_df = pd.read_csv(documents_path)
    
    # Load qrels to know which documents must be included
    qrels_df = pd.read_csv(qrels_path)

    # Always include qrel documents in the sample
    qrel_doc_ids = set(qrels_df['doc_id'].unique())
    qrel_docs = documents_df[documents_df['doc_id'].isin(qrel_doc_ids)]

    # Fill the rest with random documents
    remaining_size = 10000 - len(qrel_docs)
    other_docs = documents_df[~documents_df['doc_id'].isin(qrel_doc_ids)]
    random_docs = other_docs.sample(n=remaining_size, random_state=42)

    # Combine and shuffle
    sample_df = pd.concat([qrel_docs, random_docs], ignore_index=True)
    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Sample includes {len(qrel_docs)} qrel documents + {len(random_docs)} random documents")

    processed_df = preprocess_documents(sample_df)
    
    # Initialize evaluator
    evaluator = IREvaluator(queries_path, qrels_path)
    
    # Create retrievers
    retrievers = {}
    
    print("\nSetting up TF-IDF retriever...")
    tfidf_retriever = TFIDFRetriever()
    tfidf_retriever.fit(processed_df)
    retrievers['TF-IDF'] = tfidf_retriever
    
    print("Setting up BM25 retriever...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(processed_df)
    retrievers['BM25'] = bm25_retriever
    
    # Evaluate both retrievers
    print("\n" + "="*80)
    print("EVALUATING RETRIEVAL SYSTEMS")
    print("="*80)
    
    results_df, summary_df = evaluator.compare_retrievers(retrievers, k_values=[5, 10])
    
    # Display results
    print("\nDETAILED RESULTS:")
    print(results_df[['query_id', 'retriever', 'k', 'precision_at_k', 'recall_at_k', 'ndcg_at_k']].head(20))
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary_df)
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'metrics_summary.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Create summary table
    summary_file = os.path.join(results_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_file)
    print(f"Summary saved to: {summary_file}")
    
    return results_df, summary_df

if __name__ == "__main__":
    results, summary = main()
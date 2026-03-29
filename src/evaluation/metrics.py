import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.tfidf_retrieval import TFIDFRetriever
from retrieval.bm25_retrieval import BM25Retriever
from preprocessing.clean_text import preprocess_documents

class IREvaluator:
    """
    Information Retrieval evaluation system with standard metrics.
    """
    
    def __init__(self, queries_file: str, qrels_file: str):
        self.queries_df = pd.read_csv(queries_file)
        self.qrels_df = pd.read_csv(qrels_file)
        
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
        if query_id not in self.qrels_dict or k == 0:
            return 0.0
        top_k_docs = retrieved_docs[:k]
        relevant_count = 0
        for doc_id in top_k_docs:
            if doc_id in self.qrels_dict[query_id] and self.qrels_dict[query_id][doc_id] > 0:
                relevant_count += 1
        return relevant_count / k
    
    def recall_at_k(self, retrieved_docs: List[str], query_id: str, k: int) -> float:
        if query_id not in self.qrels_dict:
            return 0.0
        total_relevant = sum(1 for rel in self.qrels_dict[query_id].values() if rel > 0)
        if total_relevant == 0:
            return 0.0
        top_k_docs = retrieved_docs[:k]
        relevant_retrieved = 0
        for doc_id in top_k_docs:
            if doc_id in self.qrels_dict[query_id] and self.qrels_dict[query_id][doc_id] > 0:
                relevant_retrieved += 1
        return relevant_retrieved / total_relevant
    
    def average_precision(self, retrieved_docs: List[str], query_id: str) -> float:
        if query_id not in self.qrels_dict:
            return 0.0
        relevant_docs = {doc_id for doc_id, rel in self.qrels_dict[query_id].items() if rel > 0}
        if not relevant_docs:
            return 0.0
        precisions = []
        relevant_found = 0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precisions.append(precision_at_i)
        return sum(precisions) / len(relevant_docs) if precisions else 0.0
    
    def dcg_at_k(self, retrieved_docs: List[str], query_id: str, k: int) -> float:
        if query_id not in self.qrels_dict:
            return 0.0
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            relevance = self.qrels_dict[query_id].get(doc_id, 0)
            if relevance > 0:
                dcg += (2**relevance - 1) / np.log2(i + 2)
        return dcg
    
    def ndcg_at_k(self, retrieved_docs: List[str], query_id: str, k: int) -> float:
        dcg = self.dcg_at_k(retrieved_docs, query_id, k)
        if query_id not in self.qrels_dict:
            return 0.0
        relevant_docs = [(doc_id, rel) for doc_id, rel in self.qrels_dict[query_id].items() if rel > 0]
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        ideal_retrieved = [doc_id for doc_id, _ in relevant_docs]
        idcg = self.dcg_at_k(ideal_retrieved, query_id, k)
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_retriever(self, retriever, retriever_name: str, k_values: List[int] = [5, 10]) -> pd.DataFrame:
        results = []
        for _, query_row in self.queries_df.iterrows():
            query_id = query_row['query_id']
            query_text = query_row['query_text']
            
            retrieved_docs = []
            search_results = retriever.search(query_text, top_k=max(k_values))
            for doc_id, _, _ in search_results:
                retrieved_docs.append(doc_id)
            
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
        all_results = []
        for retriever_name, retriever in retrievers.items():
            print(f"\nEvaluating {retriever_name}...")
            results = self.evaluate_retriever(retriever, retriever_name, k_values)
            all_results.append(results)
        
        combined_results = pd.concat(all_results, ignore_index=True)
        summary = combined_results.groupby(['retriever', 'k']).agg({
            'precision_at_k': ['mean', 'std'],
            'recall_at_k': ['mean', 'std'],
            'ndcg_at_k': ['mean', 'std']
        }).round(4)
        
        return combined_results, summary


def main():
    documents_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'data', 'processed', 'traffic_documents.csv')
    queries_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               'data', 'queries', 'queries.csv')
    qrels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'data', 'queries', 'qrels.csv')
    
    print("Loading documents and evaluation data...")
    documents_df = pd.read_csv(documents_path)
    qrels_df = pd.read_csv(qrels_path)

    # Get all qrel doc IDs
    qrel_doc_ids = set(qrels_df['doc_id'].unique())
    qrel_docs = documents_df[documents_df['doc_id'].isin(qrel_doc_ids)]
    
    print(f"Found {len(qrel_docs)} qrel documents in corpus (out of {len(qrel_doc_ids)} needed)")

    # FIX: warn immediately if any qrel docs are missing from the corpus
    missing = qrel_doc_ids - set(qrel_docs['doc_id'].values)
    if missing:
        print(f"WARNING: {len(missing)} qrel doc IDs not found in corpus: {missing}")
        print("These queries will always score 0. Consider updating qrels.csv.")

    remaining_size = max(0, 10000 - len(qrel_docs))
    other_docs = documents_df[~documents_df['doc_id'].isin(qrel_doc_ids)]
    random_docs = other_docs.sample(n=remaining_size, random_state=42)

    sample_df = pd.concat([qrel_docs, random_docs], ignore_index=True)
    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Sample: {len(qrel_docs)} qrel docs + {len(random_docs)} random docs = {len(sample_df)} total")

    processed_df = preprocess_documents(sample_df)

    # FIX: verify qrel doc IDs survived preprocessing
    found_after_processing = set(processed_df['doc_id'].values) & qrel_doc_ids
    print(f"Qrel doc IDs confirmed in processed sample: {len(found_after_processing)} / {len(qrel_doc_ids)}")

    evaluator = IREvaluator(queries_path, qrels_path)
    retrievers = {}

    print("\nSetting up TF-IDF retriever...")
    tfidf_retriever = TFIDFRetriever()
    tfidf_retriever.fit(processed_df)
    retrievers['TF-IDF'] = tfidf_retriever

    print("Setting up BM25 retriever...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(processed_df)
    retrievers['BM25'] = bm25_retriever

    print("\n" + "="*80)
    print("EVALUATING RETRIEVAL SYSTEMS")
    print("="*80)

    results_df, summary_df = evaluator.compare_retrievers(retrievers, k_values=[5, 10])

    print("\nDETAILED RESULTS:")
    print(results_df[['query_id', 'retriever', 'k', 'precision_at_k', 'recall_at_k', 'ndcg_at_k']].head(20))

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary_df)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, 'metrics_summary.csv'), index=False)
    summary_df.to_csv(os.path.join(results_dir, 'evaluation_summary.csv'))
    print(f"\nResults saved to results/ folder")

    return results_df, summary_df

if __name__ == "__main__":
    results, summary = main()
import re
import string
from typing import List, Dict

class QueryProcessor:
    """
    Query preprocessing with the same pipeline as documents plus synonyms.
    """
    
    def __init__(self):
        self.synonyms = {
            "traffic jam": "congestion",
            "rainy": "rain",
            "rainfall": "rain",
            "storm": "heavy rain",
            "downpour": "heavy rain",
            "delay": "congestion",
            "backup": "congestion",
            "bottleneck": "congestion",
            "gridlock": "heavy congestion",
            "sunny": "clear",
            "cloudy": "clouds",
            "overcast": "clouds",
            "wet": "rain",
            "jams": "congestion",
            "jam": "congestion",
        }
        
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have',
            'had', 'which', 'their', 'if', 'can', 'would', 'there',
            'all', 'so', 'also', 'than', 'then', 'them', 'or', 'not',
            'been', 'were', 'did', 'does', 'do', 'its', 'our',
            'event', 'events', 'reported', 'detected', 'observed',
            'recorded', 'noted', 'occurred', 'happened'
        }
    
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def apply_synonyms(self, text: str) -> str:
        processed_text = text
        for synonym, target in self.synonyms.items():
            if ' ' in synonym:
                processed_text = processed_text.replace(synonym, target)
        for synonym, target in self.synonyms.items():
            if ' ' not in synonym:
                pattern = r'\b' + re.escape(synonym) + r'\b'
                processed_text = re.sub(pattern, target, processed_text)
        return processed_text
    
    def tokenize(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)
        with_synonyms = self.apply_synonyms(cleaned)
        tokens = with_synonyms.split()
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words 
            and len(token) > 1
        ]
        return filtered_tokens
    
    def preprocess_query(self, query: str) -> Dict[str, any]:
        original = query
        cleaned = self.clean_text(query)
        with_synonyms = self.apply_synonyms(cleaned)
        tokens = self.tokenize(query)
        processed_query = ' '.join(tokens)
        
        return {
            'original': original,
            'cleaned': cleaned,
            'with_synonyms': with_synonyms,
            'tokens': tokens,
            'processed_query': processed_query  # FIX: removed 'stemmed' key that didn't exist
        }
    
    def demonstrate_preprocessing(self):
        test_queries = [
            "heavy rain traffic congestion",
            "rush hour problems on the highway",
            "rainy weather causing traffic jam",
            "accidents during morning commute",
            "weekend traffic flow"
        ]
        
        print("=" * 80)
        print("QUERY PREPROCESSING DEMONSTRATION")
        print("=" * 80)
        
        for query in test_queries:
            print(f"\nOriginal Query: '{query}'")
            print("-" * 60)
            result = self.preprocess_query(query)
            print(f"Cleaned: '{result['cleaned']}'")
            print(f"With Synonyms: '{result['with_synonyms']}'")
            print(f"Tokens: {result['tokens']}")
            print(f"Final Query: '{result['processed_query']}'")  # FIX: was result['stemmed']
        
        print(f"\n{'='*80}")
        print("Query preprocessing pipeline identical to document preprocessing!")
        print("="*80)

def main():
    processor = QueryProcessor()
    processor.demonstrate_preprocessing()

if __name__ == "__main__":
    main()
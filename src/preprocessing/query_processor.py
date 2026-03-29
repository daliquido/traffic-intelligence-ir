import re
import string
from typing import List, Dict

class QueryProcessor:
    """
    Query preprocessing with the same pipeline as documents plus synonyms.
    """
    
    def __init__(self):
        # Traffic-specific synonyms for query enhancement
        self.synonyms = {
            "traffic jam": "congestion",
            "rush hour": "heavy traffic", 
            "problem": "congestion",
            "rainy": "rain",
            "rainfall": "rain",
            "storm": "heavy rain",
            "downpour": "heavy rain",
            "accident": "incident",
            "incident": "accident",
            "delay": "congestion",
            "backup": "congestion",
            "bottleneck": "congestion",
            "gridlock": "heavy congestion",
            "commute": "traffic",
            "highway": "road",
            "freeway": "road",
            "street": "road",
            "avenue": "road",
            "morning": "rush hour",
            "evening": "rush hour",
            "peak": "rush hour",
            "weekday": "work day",
            "sunny": "clear",
            "cloudy": "clouds",
            "overcast": "clouds",
            "wet": "rain",
            "snow": "precipitation",
            "fog": "weather",
            "wind": "weather"
        }
        
        # Custom stop words (same as document preprocessing, but keeping important traffic terms)
        self.stop_words = {
            # Basic English function words only
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have',
            'had', 'which', 'their', 'if', 'can', 'would', 'there',
            'all', 'so', 'also', 'than', 'then', 'them', 'or', 'not',
            'been', 'were', 'did', 'does', 'do', 'its', 'our',
            # Traffic noise words that add no search value
            'event', 'events', 'reported', 'detected', 'observed',
            'recorded', 'noted', 'occurred', 'happened'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean query text - same pipeline as documents.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def apply_synonyms(self, text: str) -> str:
        """
        Apply synonyms to expand query terms.
        """
        processed_text = text
        
        # Apply multi-word synonyms first
        for synonym, target in self.synonyms.items():
            if ' ' in synonym:  # Multi-word synonym
                processed_text = processed_text.replace(synonym, target)
        
        # Apply single-word synonyms
        for synonym, target in self.synonyms.items():
            if ' ' not in synonym:  # Single-word synonym
                # Replace whole words only
                pattern = r'\b' + re.escape(synonym) + r'\b'
                processed_text = re.sub(pattern, target, processed_text)
        
        return processed_text
    
    def simple_stem(self, text: str) -> str:
        """
        Simple stemming - same as document preprocessing.
        """
        # Common suffixes to remove
        suffixes = ['ing', 'ed', 'ly', 'es', 's', 'er', 'est', 'tion', 'ness', 'ment']
        
        words = text.split()
        stemmed_words = []
        
        for word in words:
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    word = word[:-len(suffix)]
                    break
            stemmed_words.append(word)
        
        return ' '.join(stemmed_words)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text - same as document preprocessing.
        """
        # Clean and apply synonyms
        cleaned = self.clean_text(text)
        with_synonyms = self.apply_synonyms(cleaned)
        
        # Split into tokens
        tokens = with_synonyms.split()
        
        # Remove stop words only - NO stemming
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
            'processed_query': processed_query
        }
    
    def demonstrate_preprocessing(self):
        """Demonstrate query preprocessing with examples."""
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
            print(f"Stemmed: '{result['stemmed']}'")
            print(f"Tokens: {result['tokens']}")
            print(f"Final Query: '{result['processed_query']}'")
        
        print(f"\n{'='*80}")
        print("Query preprocessing pipeline identical to document preprocessing!")
        print("This ensures queries and documents use the same token space!")
        print("="*80)

def main():
    """Test the query processor."""
    processor = QueryProcessor()
    processor.demonstrate_preprocessing()

if __name__ == "__main__":
    main()

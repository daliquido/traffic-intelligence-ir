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
        
        # Custom stop words (same as document preprocessing)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
            'up', 'out', 'many', 'then', 'them', 'can', 'would', 'there',
            'all', 'so', 'also', 'her', 'much', 'more', 'very', 'when',
            'make', 'like', 'back', 'after', 'use', 'two', 'how', 'our',
            'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
            'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are',
            'been', 'has', 'had', 'were', 'said', 'did', 'getting', 'made',
            'find', 'where', 'too', 'only', 'come', 'his', 'your', 'now',
            'than', 'call', 'who', 'oil', 'sit', 'now', 'find', 'long',
            'down', 'day', 'did', 'get', 'has', 'him', 'his', 'how', 'man',
            'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did',
            'number', 'no', 'part', 'people', 'my', 'over', 'know', 'water',
            'than', 'call', 'first', 'who', 'may', 'down', 'side', 'been',
            'now', 'find', 'head', 'stand', 'own', 'page', 'should', 'country',
            'found', 'answer', 'school', 'grow', 'study', 'still', 'learn',
            'plant', 'cover', 'food', 'sun', 'four', 'between', 'state',
            'keep', 'eye', 'never', 'last', 'let', 'thought', 'city', 'tree',
            'cross', 'farm', 'hard', 'start', 'might', 'story', 'saw', 'far',
            'sea', 'draw', 'left', 'late', 'run', 'dont', 'while', 'press',
            'close', 'night', 'real', 'life', 'few', 'north', 'open', 'seem',
            'together', 'next', 'white', 'children', 'begin', 'got', 'walk',
            'example', 'ease', 'paper', 'group', 'always', 'music', 'those',
            'both', 'mark', 'often', 'letter', 'until', 'mile', 'river',
            'car', 'feet', 'care', 'second', 'book', 'carry', 'took', 'science',
            'eat', 'room', 'friend', 'began', 'idea', 'fish', 'mountain',
            'stop', 'once', 'base', 'hear', 'horse', 'cut', 'sure', 'watch',
            'color', 'face', 'wood', 'main', 'open', 'seem', 'together',
            'next', 'white', 'children', 'begin', 'got', 'walk', 'example',
            'ease', 'paper', 'group', 'always', 'music', 'those', 'both',
            'mark', 'often', 'letter', 'until', 'mile', 'river', 'car',
            'feet', 'care', 'second', 'book', 'carry', 'took', 'science',
            'eat', 'room', 'friend', 'began', 'idea', 'fish', 'mountain',
            'stop', 'once', 'base', 'hear', 'horse', 'cut', 'sure', 'watch',
            'color', 'face', 'wood', 'main', 'traffic', 'event', 'events',
            'during', 'with', 'conditions', 'condition', 'reported', 'detected',
            'observed', 'recorded', 'noted', 'occurred', 'happened'
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
        
        # Apply stemming
        stemmed = self.simple_stem(with_synonyms)
        
        # Split into tokens
        tokens = stemmed.split()
        
        # Remove stop words
        filtered_tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
        
        return filtered_tokens
    
    def preprocess_query(self, query: str) -> Dict[str, any]:
        """
        Full query preprocessing pipeline.
        
        Args:
            query (str): Original query string
            
        Returns:
            Dict: Preprocessing results
        """
        # Original query
        original = query
        
        # Cleaned query
        cleaned = self.clean_text(query)
        
        # With synonyms
        with_synonyms = self.apply_synonyms(cleaned)
        
        # Stemmed
        stemmed = self.simple_stem(with_synonyms)
        
        # Tokenized
        tokens = self.tokenize(query)
        
        # Reconstructed query for search
        processed_query = ' '.join(tokens)
        
        return {
            'original': original,
            'cleaned': cleaned,
            'with_synonyms': with_synonyms,
            'stemmed': stemmed,
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

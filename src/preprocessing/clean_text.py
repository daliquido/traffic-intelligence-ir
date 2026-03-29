import re
import string
from typing import List

def clean_text(text: str) -> str:
    """
    Enhanced clean and preprocess text for IR.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle traffic-specific terms and numbers
    # Keep "mm" for precipitation, "kmh" for speed, etc.
    text = re.sub(r'(\d+)mm', r'\1_mm', text)  # precipitation
    text = re.sub(r'(\d+)kmh', r'\1_kmh', text)  # speed
    
    # Remove punctuation except underscores
    text = text.translate(str.maketrans('', '', string.punctuation.replace('_', '')))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_traffic_stop_words() -> set:
    """
    Get customized stop words for traffic domain.
    Uses basic English stop words plus traffic-specific terms.
    """
    # Basic English stop words
    basic_stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
        'i', 'you', 'your', 'they', 'them', 'their', 'this', 'these', 'those', 'am',
        'been', 'being', 'but', 'can', 'could', 'did', 'do', 'does', 'had', 'have',
        'having', 'may', 'might', 'must', 'shall', 'should', 'were', 'would', 'or',
        'if', 'than', 'then', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'also'
    }
    
    # Traffic-specific stop words (common but not useful for search)
    traffic_stopwords = {
        'event', 'events', 'reported', 'detected', 'observed',
        'recorded', 'document', 'text', 'preview', 'file', 'data'
        # Removed: 'traffic', 'road', 'weather', 'conditions', 'during', 'with'
        # These are core search terms in your documents - never remove them
    }
    
    # Combine both sets
    all_stopwords = basic_stopwords.union(traffic_stopwords)
    
    # Remove some terms that might be useful for traffic search
    useful_terms = {'rain', 'snow', 'accident', 'congestion', 'closure', 'construction'}
    all_stopwords = all_stopwords - useful_terms
    
    return all_stopwords

def simple_stem(token: str) -> str:
    """
    Conservative stemming - only remove endings when safe.
    """
    if len(token) <= 4:
        return token  # Don't touch short words at all

    # Only remove these specific endings, and only when word is long enough
    # Order matters - check longer suffixes first
    safe_suffixes = [
        ('tion', 4),    # congestion -> congest (min 4 chars left)
        ('ing', 4),     # flowing -> flow
        ('ed', 4),      # blocked -> block
        ('ly', 4),      # heavily -> heavi
        ('es', 4),      # crashes -> crash
    ]

    for suffix, min_remaining in safe_suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_remaining:
            return token[:-len(suffix)]

    return token  # Return unchanged if no safe suffix found

def tokenize(text: str) -> List[str]:
    """
    Enhanced tokenization with stop word removal and simple stemming.
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        List[str]: List of cleaned tokens
    """
    cleaned_text = clean_text(text)
    
    # Split into tokens
    tokens = cleaned_text.split()
    
    # Get stop words
    stop_words = get_traffic_stop_words()
    
    # Process tokens
    processed_tokens = []
    for token in tokens:
        if token.lower() in stop_words:
            continue
        if len(token) < 2:
            continue
        # No stemming - just append the token directly
        processed_tokens.append(token)
    
    return processed_tokens

def preprocess_documents(documents_df):
    """
    Enhanced preprocessing with stemming and stop word removal.
    
    Args:
        documents_df (pd.DataFrame): DataFrame with documents
        
    Returns:
        pd.DataFrame: DataFrame with cleaned text
    """
    df = documents_df.copy()
    
    print("Preprocessing documents with enhanced pipeline...")
    
    # Apply enhanced cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Apply tokenization (for analysis)
    df['tokens'] = df['text'].apply(tokenize)
    
    # Create searchable text from tokens (join back)
    df['searchable_text'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))
    
    # Also clean title
    df['cleaned_title'] = df['title'].apply(clean_text)
    
    print(f"Preprocessed {len(df)} documents")
    print(f"Sample original: {df['text'].iloc[0]}")
    print(f"Sample cleaned: {df['cleaned_text'].iloc[0]}")
    print(f"Sample tokens: {df['tokens'].iloc[0]}")
    print(f"Sample searchable: {df['searchable_text'].iloc[0]}")
    
    # Show vocabulary statistics
    all_tokens = [token for tokens in df['tokens'] for token in tokens]
    unique_tokens = set(all_tokens)
    print(f"Vocabulary size: {len(unique_tokens)} unique tokens")
    print(f"Total tokens: {len(all_tokens)}")
    
    return df

def analyze_vocabulary(documents_df):
    """
    Analyze the vocabulary to understand most common terms.
    
    Args:
        documents_df (pd.DataFrame): Preprocessed documents
    """
    from collections import Counter
    
    # Get all tokens
    all_tokens = [token for tokens in documents_df['tokens'] for token in tokens]
    
    # Count frequency
    token_freq = Counter(all_tokens)
    
    print("\n=== VOCABULARY ANALYSIS ===")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Unique tokens: {len(token_freq)}")
    
    print("\nTop 20 most common tokens:")
    for token, freq in token_freq.most_common(20):
        print(f"  {token}: {freq}")
    
    print("\nBottom 10 least common tokens:")
    for token, freq in token_freq.most_common()[-10:]:
        print(f"  {token}: {freq}")

if __name__ == "__main__":
    # Test the enhanced preprocessing
    test_text = "Traffic event: heavy congestion on secondary road during rain weather conditions with 0.09mm precipitation"
    print(f"Original: {test_text}")
    print(f"Cleaned: {clean_text(test_text)}")
    print(f"Tokens: {tokenize(test_text)}")
    
    # Test stop words
    stop_words = get_traffic_stop_words()
    print(f"\nTotal stop words: {len(stop_words)}")
    print(f"Sample stop words: {list(stop_words)[:10]}")
    
    # Test stemming
    test_tokens = ["congestion", "conditions", "weather", "reported", "accidents"]
    print(f"\nStemming examples:")
    for token in test_tokens:
        print(f"  {token} -> {simple_stem(token)}")
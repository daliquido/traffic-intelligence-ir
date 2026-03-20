# Traffic Event Retrieval for Route Optimization and Traffic Intelligence

This project implements an Information Retrieval system for searching and ranking traffic-event documents such as accidents, weather disruptions, and road closures. The goal is to support route optimization and traffic intelligence by retrieving contextual events relevant to user queries.

## Project Overview

**Team Members**: Ayomide, Martin  
**Course**: Information Retrieval  
**Project Type**: Traffic Event Search and Ranking System  

## Project Structure

```
traffic-event-ir/
├── data/
│   ├── raw/                    # Raw CSV files from ML project
│   │   ├── kigali_weather_traffic_ultimate_clean.csv
│   │   ├── traffic_simulation.csv
│   │   └── kigali_forecast_weather.csv
│   ├── processed/              # IR-processed documents
│   │   └── traffic_documents.csv
│   ├── interim/               # Intermediate processing results
│   └── queries/               # Test queries and relevance judgments
├── src/
│   ├── data/                  # Data loading and document creation
│   │   └── load_data.py
│   ├── preprocessing/         # Text cleaning and tokenization
│   │   └── clean_text.py
│   ├── retrieval/            # Search and ranking algorithms
│   │   ├── tfidf_retrieval.py
│   │   └── bm25_retrieval.py    # TODO: Implement
│   └── evaluation/           # Metrics and evaluation tools
│       └── metrics.py           # TODO: Implement
├── notebooks/               # Jupyter experiments
├── results/                 # Evaluation results and outputs
├── sprint_plan.txt          # Detailed sprint planning
├── sprint1_data_analysis.md # Data understanding documentation
├── sprint2_document_schema.md # Document format specification
└── requirements.txt         # Python dependencies
```

## Dataset Information

**Source**: Kigali traffic and weather simulation data  
**Size**: 544,320 traffic events  
**Time Period**: February 2026 simulation  
**Format**: CSV files with traffic volume, weather conditions, and road network data

### Key Data Columns Used
- **Traffic**: `vehicle_counts`, `highway_type`, `is_rush_hour`, `is_weekend`
- **Weather**: `temperature`, `precipitation`, `is_rain`, `is_heavy_rain`, `visibility`
- **Location**: `source_node`, `target_node`, `highway_type`
- **Time**: `timestamp`, `hour_of_day`, `day_of_week`

## IR Document Format

Each traffic event becomes a searchable document:

```json
{
  "doc_id": "traffic_0",
  "title": "Free Flow - Rain",
  "timestamp": "2026-02-10 00:00:00", 
  "event_type": "Rain",
  "location": "secondary road",
  "vehicle_count": 38,
  "text": "Traffic event: free flow on secondary road during rain weather conditions with 0.09mm precipitation"
}
```

## System Architecture

### 1. Data Processing Pipeline
- Load raw CSV data
- Transform rows into IR documents
- Apply text preprocessing (cleaning, tokenization, stemming)
- Generate searchable text descriptions

### 2. Text Preprocessing
- **Stop word removal**: 110 common English + traffic-specific words
- **Stemming**: Custom suffix removal for term normalization
- **Vocabulary reduction**: From 364 → 320 terms (12% improvement)
- **Traffic-specific handling**: Preserves terms like "rain", "accident", "congestion"

### 3. Retrieval Models
- **TF-IDF**: ✅ Implemented and tested
- **BM25**: 🔄 Next implementation target
- **Cosine Similarity**: Used for ranking

## Current Progress

### ✅ Completed (Sprint 0-2)
- **Project Setup**: Clean folder structure, separate from ML codebase
- **Data Analysis**: Identified 9 key IR columns from 34 total columns
- **Document Schema**: Defined 7-field document format
- **Data Processing**: Transformed 544K rows → searchable IR documents
- **Text Preprocessing**: Enhanced with stop words and stemming
- **TF-IDF Retrieval**: Working baseline with cosine similarity

### 🔄 In Progress (Sprint 2)
- **BM25 Implementation**: Next task for Ayomide
- **Evaluation Metrics**: Next task for Martin

### 📋 Planned (Sprint 3-5)
- **Performance Evaluation**: Precision@K, Recall@K, MAP, nDCG
- **Query Set**: Test queries and relevance judgments
- **Model Comparison**: TF-IDF vs BM25 performance analysis
- **System Improvements**: Query expansion, semantic search
- **Final Demo**: Interactive search interface

## Usage Examples

### Basic Search
```python
from src.retrieval.tfidf_retrieval import TFIDFRetriever
from src.preprocessing.clean_text import preprocess_documents
import pandas as pd

# Load and preprocess documents
documents = pd.read_csv('data/processed/traffic_documents.csv')
processed_docs = preprocess_documents(documents.sample(10000))

# Create and fit retriever
retriever = TFIDFRetriever()
retriever.fit(processed_docs)

# Search for traffic events
results = retriever.search("heavy rain road conditions", top_k=5)
retriever.print_results("heavy rain road conditions", results)
```

### Sample Queries
- "rain traffic congestion"
- "heavy rain road conditions" 
- "rush hour traffic"
- "weekend traffic flow"
- "secondary road accident"

## Performance Metrics

### Current TF-IDF System
- **Vocabulary Size**: 320 unique terms
- **Document Matrix**: 10,000 × 320 (sample)
- **Search Speed**: ~0.1 seconds per query
- **Sample Results**: High relevance for weather + traffic queries

### Example Results
```
Query: "heavy rain road conditions"
1. [0.5473] Heavy Congestion - Heavy Rain
   Preview: heavy congestion on residential road during heavy rain weather conditions...
```

## Dependencies

```txt
pandas>=1.3.0
scikit-learn>=1.0.0
numpy>=1.21.0
jupyter>=1.0.0
rank-bm25>=0.2.2
```

## Next Steps

### Ayomide (Sprint 2)
- [ ] Implement BM25 retrieval model
- [ ] Create TF-IDF vs BM25 comparison framework
- [ ] Document preprocessing improvements

### Martin (Sprint 2)  
- [ ] Implement evaluation metrics (Precision@K, Recall@K, MAP, nDCG)
- [ ] Create test queries and relevance judgments
- [ ] Set up evaluation pipeline

### Both Team Members
- [ ] Integrate BM25 and evaluation components
- [ ] Compare model performance
- [ ] Document findings

## Project Goals

**Primary**: Build a working IR system that retrieves relevant traffic events for user queries

**Success Criteria**:
- ✅ Users can type traffic queries and get ranked results
- 🔄 System performance measured with standard IR metrics  
- 📋 Comparison between TF-IDF and BM25 retrieval models
- 📋 Clear documentation and reproducible results

**Long-term Vision**: Support route optimization systems by providing contextual traffic event information for real-time decision making.
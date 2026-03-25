# Sprint 7 - Query Set and Relevance Judgments

## Goal
Build evaluation data for IR system assessment.

## Files Created

### `queries.csv`
- **25 test queries** across 4 categories
- **Format**: query_id, query_text, category
- **Categories**:
  - `weather-related traffic` (8 queries)
  - `rush-hour` (6 queries) 
  - `congestion` (7 queries)
  - `road-specific` (4 queries)

### `qrels.csv` 
- **Sample relevance judgments** for first 5 queries
- **Format**: query_id, doc_id, relevance
- **Relevance levels**: 
  - `2` = Highly relevant
  - `1` = Somewhat relevant
  - `0` = Not relevant (not shown in sample)

## Team Division

### AYOMIDE - Query Creation ✅ COMPLETE
- [x] Created 25 diverse test queries
- [x] Organized into logical categories
- [x] Saved to `queries.csv`

### MARTIN - Relevance Labeling 🔄 NEEDED
- [ ] Review each query in `queries.csv`
- [ ] Run queries on both TF-IDF and BM25 systems
- [ ] Manually judge relevance of top 20 results per query
- [ ] Assign relevance scores (0=not, 1=somewhat, 2=highly)
- [ ] Complete `qrels.csv` with all 25 queries

### BOTH - Review and Finalize 🔄 NEEDED
- [ ] Meet to review relevance judgments
- [ ] Discuss disagreements in scoring
- [ ] Finalize evaluation set
- [ ] Test evaluation metrics with completed qrels

## Relevance Guidelines

### Highly Relevant (2)
- Document matches ALL major query terms
- Example: Query "heavy rain traffic" → Document mentions heavy rain AND traffic congestion
- Specific location/time matches query intent

### Somewhat Relevant (1)  
- Document matches SOME query terms
- Example: Query "heavy rain traffic" → Document mentions rain but light traffic
- General traffic information without specific weather context

### Not Relevant (0)
- Document doesn't match query terms
- Example: Query "heavy rain" → Document about clear weather
- Completely different topic

## Sample Query Analysis

**Query**: "heavy rain traffic congestion"
- **Highly Relevant**: "heavy congestion during heavy rain weather conditions"
- **Somewhat Relevant**: "light traffic during rain weather conditions"  
- **Not Relevant**: "free flow during clear weather"

## Next Steps

1. **Martin**: Complete relevance labeling for all 25 queries
2. **Both**: Review and finalize qrels
3. **Martin**: Implement evaluation metrics using this data
4. **Ayomide**: Test evaluation results

## Expected Final Dataset

- **25 queries** total
- **~500 relevance judgments** (20 docs × 25 queries)
- **Balanced relevance distribution** (~40% relevant, 60% non-relevant)
- **Ready for Precision@K, Recall@K, MAP, nDCG evaluation**

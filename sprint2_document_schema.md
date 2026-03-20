# Sprint 2 - IR Document Format Design

## Goal
Decide what one searchable document looks like.

## Document Schema

Each traffic event document contains the following fields:

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

## Field Definitions

### Core Fields
- **doc_id**: Unique identifier (format: "traffic_{row_index}")
- **timestamp**: Exact time of the traffic event
- **event_type**: Weather condition (Rain, Heavy Rain, Clear, Clouds, Unknown)
- **location**: Road type (secondary, residential, primary, service road)
- **vehicle_count**: Actual traffic volume
- **text**: Full searchable text description
- **title**: Human-readable summary (congestion_level - event_type)

## Congestion Rules

Traffic congestion levels based on vehicle_count:

```python
if vehicle_count > 500:
    congestion_level = "heavy congestion"
elif vehicle_count > 200:
    congestion_level = "moderate congestion" 
elif vehicle_count > 50:
    congestion_level = "light traffic"
else:
    congestion_level = "free flow"
```

## Weather Rules

Weather conditions derived from data columns:

```python
if is_heavy_rain:
    weather_condition = "Heavy Rain"
elif is_rain:
    weather_condition = "Rain"
elif weather_code == 0:
    weather_condition = "Clear"
elif weather_code == 1:
    weather_condition = "Clouds"
else:
    weather_condition = "Unknown"
```

## Document Text Generation

Template for creating searchable text:

```
"Traffic event: {congestion_level} on {highway_type} road during {weather} weather conditions with {precipitation}mm precipitation"
```

Additional context added when applicable:
- "during rush hour" (if is_rush_hour = True)
- "on weekend" (if is_weekend = True)
- "during hot conditions" (if temperature > 30°C)
- "during cold conditions" (if temperature < 15°C)

## Sample Document Examples

### Example 1: Heavy Rain Congestion
```json
{
  "doc_id": "traffic_349794",
  "title": "Heavy Congestion - Heavy Rain",
  "timestamp": "2026-02-10 07:00:00", 
  "event_type": "Heavy Rain",
  "location": "residential road",
  "vehicle_count": 560,
  "text": "Traffic event: heavy congestion on residential road during heavy rain weather conditions during rush hour with 0.35mm precipitation"
}
```

### Example 2: Light Traffic
```json
{
  "doc_id": "traffic_253229",
  "title": "Light Traffic - Rain",
  "timestamp": "2026-02-10 14:00:00",
  "event_type": "Rain", 
  "location": "residential road",
  "vehicle_count": 45,
  "text": "Traffic event: light traffic on residential road during rain weather conditions with 0.39mm precipitation"
}
```

### Example 3: Free Flow Weekend
```json
{
  "doc_id": "traffic_310774",
  "title": "Free Flow - Unknown",
  "timestamp": "2026-02-15 10:00:00",
  "event_type": "Unknown",
  "location": "residential road", 
  "vehicle_count": 12,
  "text": "Traffic event: free flow on residential road on weekend"
}
```

## Document Strategy

- **One row = one document**: Each timestamped traffic observation becomes a searchable document
- **Text-focused**: Primary search field is the generated `text` description
- **Structured metadata**: Additional fields available for filtering and analysis
- **Human-readable**: Documents written in natural language for intuitive search

## Deliverable Status
✅ **COMPLETED** - Clear document schema and sample examples defined

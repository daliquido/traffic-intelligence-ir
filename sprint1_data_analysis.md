# Sprint 1 - Raw Data Analysis

## Goal
Understand exactly what data we have and what can become an IR document.

## Selected Raw File(s)
**Primary**: `kigali_weather_traffic_ultimate_clean.csv`
- **Reason**: Most comprehensive dataset with both traffic and weather data
- **Size**: 544,320 rows, 34 columns
- **Alternative**: `traffic_simulation.csv` (similar but less comprehensive)

## Selected Columns for IR Documents

### Core Traffic Information
- `timestamp` - When the event occurred
- `vehicle_counts` - Traffic volume (congestion indicator)
- `highway_type` - Road type (location context)
- `is_rush_hour` - Temporal context
- `is_weekend` - Temporal context

### Weather Information  
- `temperature` - Weather severity
- `precipitation` - Rain/snow amount
- `is_rain` - Binary rain indicator
- `is_heavy_rain` - Severe weather indicator
- `visibility` - Weather conditions impact

### Location/Network Context
- `source_node`, `target_node` - Road segment identifiers (used internally)

## Ignored Columns

### ML-Engineered Features (Not needed for IR)
- `day_of_week_num` - Numerical encoding
- `hour_sin`, `hour_cos` - Cyclical encoding  
- `day_sin`, `day_cos` - Cyclical encoding
- `weather_code` - Already covered by boolean flags
- `is_hot`, `is_cold` - Derived from temperature
- `rain_rush_hour`, `rain_weekend` - Combined features
- `temperature_lag_1h`, `precipitation_lag_1h` - Time series features

### Network Engineering Data (Not needed for IR)
- `road_capacity` - ML feature
- `road_length_meters` - Engineering detail
- `speed_limit_kmh` - Technical detail
- `lanes` - Infrastructure detail
- `humidity`, `pressure`, `wind_speed`, `wind_direction` - Too detailed for IR
- `cloud_cover` - Less relevant for traffic events

## IR Document Transformation Strategy

Each row becomes a traffic event document with:
- **Event Type**: Derived from `is_rain`, `is_heavy_rain`, `vehicle_counts`
- **Location**: `highway_type` (secondary, residential, etc.)
- **Time**: `timestamp`, `is_rush_hour`, `is_weekend` context
- **Severity**: `vehicle_counts`, `precipitation`, `temperature`

Example document:
"Traffic event: heavy congestion on secondary road during rain weather conditions with 0.09mm precipitation"

## Deliverable Status
✅ **COMPLETED** - Raw data analyzed and IR document strategy defined

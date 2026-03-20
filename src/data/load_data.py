import pandas as pd
import numpy as np
from pathlib import Path

def load_raw_data():
    """
    Load raw traffic and weather data from CSV files.
    
    Returns:
        dict: Dictionary containing loaded DataFrames
    """
    data_path = Path(__file__).parent.parent.parent / "data" / "raw"
    
    # Load the main datasets
    data = {}
    
    # Load traffic simulation data (has weather_condition + vehicle_counts)
    try:
        traffic_data = pd.read_csv(data_path / "traffic_simulation.csv")
        data['traffic_simulation'] = traffic_data
        print(f"Loaded traffic_simulation.csv: {len(traffic_data)} rows")
    except FileNotFoundError:
        print("Warning: traffic_simulation.csv not found")
    
    # Load weather forecast data
    try:
        weather_data = pd.read_csv(data_path / "kigali_forecast_weather.csv")
        data['weather_forecast'] = weather_data
        print(f"Loaded kigali_forecast_weather.csv: {len(weather_data)} rows")
    except FileNotFoundError:
        print("Warning: kigali_forecast_weather.csv not found")
    
    # Load ultimate clean data (most comprehensive)
    try:
        ultimate_data = pd.read_csv(data_path / "kigali_weather_traffic_ultimate_clean.csv")
        data['ultimate_clean'] = ultimate_data
        print(f"Loaded kigali_weather_traffic_ultimate_clean.csv: {len(ultimate_data)} rows")
    except FileNotFoundError:
        print("Warning: kigali_weather_traffic_ultimate_clean.csv not found")
    
    return data

def create_traffic_documents(df):
    """
    Transform traffic data rows into IR documents.
    
    Each document represents a traffic event that can be searched.
    
    Args:
        df (pd.DataFrame): Traffic data DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with IR document format
    """
    documents = []
    
    for idx, row in df.iterrows():
        # Extract key information for IR document
        timestamp = row.get('timestamp', '')
        
        # Handle weather condition from different datasets
        if 'weather_condition' in row:
            weather_condition = row.get('weather_condition', 'unknown')
        else:
            # Use weather_code to determine condition
            weather_code = row.get('weather_code', 0)
            is_rain = row.get('is_rain', False)
            is_heavy_rain = row.get('is_heavy_rain', False)
            
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
        
        vehicle_count = row.get('vehicle_counts', 0)
        highway_type = row.get('highway_type', 'road')
        is_rush_hour = row.get('is_rush_hour', False)
        is_weekend = row.get('is_weekend', False)
        temperature = row.get('temperature', 0)
        precipitation = row.get('precipitation', 0)
        
        # Determine congestion level based on vehicle count
        if vehicle_count > 500:
            congestion_level = "heavy congestion"
        elif vehicle_count > 200:
            congestion_level = "moderate congestion"
        elif vehicle_count > 50:
            congestion_level = "light traffic"
        else:
            congestion_level = "free flow"
        
        # Create document text (searchable content)
        doc_text = f"Traffic event: {congestion_level} on {highway_type} road"
        
        # Add weather context
        if weather_condition.lower() != 'unknown':
            doc_text += f" during {weather_condition.lower()} weather conditions"
        
        # Add temporal context
        if is_rush_hour:
            doc_text += " during rush hour"
        if is_weekend:
            doc_text += " on weekend"
        
        # Add severity details
        if precipitation > 0:
            doc_text += f" with {precipitation:.2f}mm precipitation"
        if temperature > 30:
            doc_text += " during hot conditions"
        elif temperature < 15:
            doc_text += " during cold conditions"
        
        # Create document record
        doc = {
            'doc_id': f"traffic_{idx}",
            'title': f"{congestion_level.title()} - {weather_condition}",
            'timestamp': timestamp,
            'event_type': weather_condition,
            'location': f"{highway_type} road",
            'vehicle_count': vehicle_count,
            'text': doc_text
        }
        
        documents.append(doc)
    
    return pd.DataFrame(documents)

def main():
    """
    Main function to load data and create IR documents.
    """
    print("Loading raw data...")
    raw_data = load_raw_data()
    
    # Use the most comprehensive dataset for IR documents
    if 'ultimate_clean' in raw_data:
        print("Using ultimate_clean dataset for IR documents...")
        df = raw_data['ultimate_clean']
    elif 'traffic_simulation' in raw_data:
        print("Using traffic_simulation dataset for IR documents...")
        df = raw_data['traffic_simulation']
    else:
        print("No suitable dataset found!")
        return
    
    print("Creating IR documents...")
    documents = create_traffic_documents(df)
    
    # Save processed documents
    output_path = Path(__file__).parent.parent.parent / "data" / "processed"
    output_path.mkdir(exist_ok=True)
    
    documents.to_csv(output_path / "traffic_documents.csv", index=False)
    print(f"Saved {len(documents)} IR documents to data/processed/traffic_documents.csv")
    
    # Show sample documents
    print("\nSample IR documents:")
    print(documents[['doc_id', 'title', 'text']].head(3).to_string())
    
    return documents

if __name__ == "__main__":
    documents = main()
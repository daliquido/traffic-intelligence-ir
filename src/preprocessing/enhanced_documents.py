import pandas as pd
import re
import string
from typing import List, Dict, Set

class EnhancedDocumentGenerator:
    """
    Enhanced document generation with better wording, labels, and field weighting.
    """
    
    def __init__(self):
        # Location aliases for better matching
        self.location_aliases = {
            'secondary': ['secondary road', 'local road', 'side street', 'neighborhood road'],
            'residential': ['residential road', 'housing area', 'home street', 'local street'],
            'primary': ['primary road', 'main road', 'major road', 'arterial'],
            'tertiary': ['tertiary road', 'minor road', 'small road'],
            'service': ['service road', 'access road', 'utility road'],
            'highway': ['highway', 'freeway', 'expressway', 'motorway']
        }
        
        # Enhanced event type labels
        self.event_type_enhancements = {
            'Rain': ['rain', 'rainfall', 'precipitation', 'wet weather', 'rainy conditions'],
            'Heavy Rain': ['heavy rain', 'downpour', 'storm', 'severe rain', 'intense rainfall'],
            'Clear': ['clear', 'sunny', 'dry', 'good weather', 'fair conditions'],
            'Clouds': ['cloudy', 'overcast', 'cloud cover', 'gray sky'],
            'Unknown': ['weather', 'conditions', 'atmospheric']
        }
        
        # Field importance weights (for manual weighting)
        self.field_weights = {
            'congestion_level': 3.0,      # Most important
            'weather_condition': 2.5,     # Very important  
            'location_type': 2.0,         # Important
            'temporal_context': 1.5,      # Moderately important
            'severity_details': 1.0       # Standard importance
        }
    
    def enhance_location_text(self, highway_type: str) -> str:
        """Add location aliases for better matching."""
        aliases = self.location_aliases.get(highway_type, [highway_type])
        return ' '.join(aliases[:3])  # Use top 3 aliases
    
    def enhance_event_text(self, event_type: str) -> str:
        """Add event type synonyms and enhanced descriptions."""
        enhancements = self.event_type_enhancements.get(event_type, [event_type])
        return ' '.join(enhancements[:3])  # Use top 3 enhancements
    
    def create_enhanced_title(self, row) -> str:
        """Create enhanced title with better wording."""
        vehicle_count = row.get('vehicle_counts', 0)
        weather_condition = self.determine_weather_condition(row)
        highway_type = row.get('highway_type', 'road')
        
        # Enhanced congestion labeling
        if vehicle_count > 500:
            congestion = "Severe Traffic Congestion"
        elif vehicle_count > 200:
            congestion = "Moderate Traffic Congestion" 
        elif vehicle_count > 50:
            congestion = "Light Traffic Conditions"
        else:
            congestion = "Free Traffic Flow"
        
        # Enhanced weather labeling
        if weather_condition == "Heavy Rain":
            weather = "Heavy Rainfall Alert"
        elif weather_condition == "Rain":
            weather = "Rain Weather Impact"
        elif weather_condition == "Clear":
            weather = "Clear Weather Conditions"
        elif weather_condition == "Clouds":
            weather = "Cloudy Weather"
        else:
            weather = "Weather Conditions"
        
        # Enhanced location labeling
        location_map = {
            'secondary': 'Secondary Road Network',
            'residential': 'Residential Area Roads',
            'primary': 'Primary Road Corridor',
            'tertiary': 'Tertiary Road System',
            'service': 'Service Road Access',
            'highway': 'Highway System'
        }
        location = location_map.get(highway_type, f"{highway_type.title()} Road")
        
        return f"{congestion} - {weather} on {location}"
    
    def create_enhanced_body(self, row) -> str:
        """Create enhanced body text with field weighting."""
        vehicle_count = row.get('vehicle_counts', 0)
        highway_type = row.get('highway_type', 'road')
        is_rush_hour = row.get('is_rush_hour', False)
        is_weekend = row.get('is_weekend', False)
        temperature = row.get('temperature', 0)
        precipitation = row.get('precipitation', 0)
        weather_condition = self.determine_weather_condition(row)
        
        # Weighted text components
        text_parts = []
        
        # High weight: Congestion level (repeated for emphasis)
        if vehicle_count > 500:
            text_parts.extend(["severe congestion", "heavy traffic", "major traffic backup", "severe congestion"])
        elif vehicle_count > 200:
            text_parts.extend(["moderate congestion", "traffic delays", "moderate congestion"])
        elif vehicle_count > 50:
            text_parts.extend(["light traffic", "minor congestion", "light traffic"])
        else:
            text_parts.extend(["free flow", "clear traffic", "normal traffic flow"])
        
        # High weight: Weather conditions  
        weather_enhanced = self.enhance_event_text(weather_condition)
        text_parts.extend(weather_enhanced)
        
        # Medium weight: Location with aliases
        location_enhanced = self.enhance_location_text(highway_type)
        text_parts.extend(location_enhanced)
        
        # Medium weight: Temporal context
        if is_rush_hour:
            text_parts.extend(["rush hour", "peak traffic time", "commute hours"])
        if is_weekend:
            text_parts.extend(["weekend traffic", "weekend travel", "weekend conditions"])
        
        # Standard weight: Severity details
        if precipitation > 0:
            text_parts.extend([f"{precipitation:.2f}mm precipitation", f"rainfall {precipitation:.2f}mm"])
        if temperature > 30:
            text_parts.extend(["hot weather", "high temperature", "heat conditions"])
        elif temperature < 15:
            text_parts.extend(["cold weather", "low temperature", "cold conditions"])
        
        return '. '.join(text_parts) + '.'
    
    def determine_weather_condition(self, row) -> str:
        """Enhanced weather condition determination."""
        if row.get('is_heavy_rain', False):
            return "Heavy Rain"
        elif row.get('is_rain', False):
            return "Rain"
        elif row.get('weather_code', 0) == 0:
            return "Clear"
        elif row.get('weather_code', 0) == 1:
            return "Clouds"
        else:
            return "Unknown"
    
    def create_enhanced_documents(self, documents_df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced documents with better wording and structure."""
        enhanced_docs = []
        
        print("Creating enhanced documents with improved wording...")
        
        for idx, row in documents_df.iterrows():
            # Enhanced document structure
            doc = {
                'doc_id': f"enhanced_{idx}",
                'title': self.create_enhanced_title(row),
                'body': self.create_enhanced_body(row),
                'timestamp': row.get('timestamp', ''),
                'event_type': self.determine_weather_condition(row),
                'location_type': row.get('highway_type', 'road'),
                'vehicle_count': row.get('vehicle_counts', 0),
                'congestion_level': self._get_congestion_level(row.get('vehicle_counts', 0)),
                'weather_severity': self._get_weather_severity(row)
            }
            
            # Create searchable text (title + body with field weighting)
            doc['searchable_text'] = f"{doc['title']}. {doc['body']}"
            
            enhanced_docs.append(doc)
            
            # Progress indicator
            if (idx + 1) % 100000 == 0:
                print(f"  Processed {idx + 1:,} enhanced documents...")
        
        enhanced_df = pd.DataFrame(enhanced_docs)
        print(f"Created {len(enhanced_df)} enhanced documents")
        
        return enhanced_df
    
    def _get_congestion_level(self, vehicle_count: int) -> str:
        """Get congestion level for categorization."""
        if vehicle_count > 500:
            return "severe"
        elif vehicle_count > 200:
            return "moderate"
        elif vehicle_count > 50:
            return "light"
        else:
            return "free"
    
    def _get_weather_severity(self, row) -> str:
        """Get weather severity for categorization."""
        if row.get('is_heavy_rain', False):
            return "severe"
        elif row.get('is_rain', False):
            return "moderate"
        else:
            return "mild"

def main():
    """Test enhanced document generation."""
    # Load original documents
    documents_path = "/Users/i/traffic-intelligence-ir/data/processed/traffic_documents.csv"
    original_df = pd.read_csv(documents_path)
    
    # Take sample for testing
    sample_df = original_df.sample(n=5, random_state=42)
    
    # Create enhanced documents
    generator = EnhancedDocumentGenerator()
    enhanced_df = generator.create_enhanced_documents(sample_df)
    
    print("\n" + "="*80)
    print("ENHANCED DOCUMENT COMPARISON")
    print("="*80)
    
    for i in range(len(enhanced_df)):
        original = sample_df.iloc[i]
        enhanced = enhanced_df.iloc[i]
        
        print(f"\nDOCUMENT {i+1}:")
        print(f"Original Title: {original['title']}")
        print(f"Enhanced Title: {enhanced['title']}")
        print(f"Original Text: {original['text'][:100]}...")
        print(f"Enhanced Text: {enhanced['searchable_text'][:150]}...")
        print("-" * 60)

if __name__ == "__main__":
    main()

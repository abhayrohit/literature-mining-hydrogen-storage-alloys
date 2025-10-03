"""
Manual data extraction from the hydrogen storage alloy papers.
Creates the final dataset by processing the scopus_export.csv directly.
"""

import pandas as pd
import re
import json
import os
from datetime import datetime

def extract_storage_capacity(text):
    """Extract hydrogen storage capacity from text."""
    patterns = [
        r'(\d+\.?\d*)\s*wt%\s*h[₂2]?',
        r'(\d+\.?\d*)\s*wt%\s*hydrogen',
        r'h/m\s*=\s*[\d.]+\s*\((\d+\.?\d*)\s*wt%',
        r'(\d+\.?\d*)\s*weight\s*%\s*h',
        r'capacity.*?(\d+\.?\d*)\s*wt%',
        r'storage.*?(\d+\.?\d*)\s*wt%',
        r'absorb.*?(\d+\.?\d*)\s*wt%',
        r'desorb.*?(\d+\.?\d*)\s*wt%',
        r'release.*?(\d+\.?\d*)\s*wt%'
    ]
    
    capacities = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            try:
                capacity = float(match)
                if 0.1 <= capacity <= 25:  # Reasonable range
                    capacities.append(capacity)
            except ValueError:
                continue
    
    if capacities:
        return max(capacities)  # Return highest capacity found
    return None

def extract_alloy_name(title, abstract):
    """Extract alloy name from title and abstract."""
    text = (title + " " + abstract).lower()
    
    # Specific alloy patterns
    patterns = {
        'Li-Mg-Al-Ti': ['li-mg-al-ti', 'li₂₀mg₂₀al₂₀ti₂₀'],
        'Mg-Ni': ['mg-ni', 'mg₂ni', 'mgni', 'mg-ni-'],
        'TiFe': ['tife', 'ti-fe', 'ti₀.₅fe₀.₄₅'],
        'ZrCo': ['zrco', 'zr-co', 'zr₀.₇'],
        'Fe-doped Mg₂NiH₄': ['fe-doped mg₂nih₄'],
        'AB₅-type': ['ab₅', 'ab5', 'lani₅'],
        'V-Ti': ['v-ti', 'v–ti', 'v₆₁cr₂₄ti₁₂'],
        'Ti-Al': ['ti-al', 'ti-xal'],
        'High entropy alloy': ['high entropy', 'hea'],
        'Nb-Cr-Mn': ['nb-cr-mn', 'nb₈₅cr₁₀mn₅'],
        'Au-Y': ['au', 'aun y'],
        'Mg₂Ni/TiH₁.₅': ['mg₂ni/tih₁.₅'],
        'NiCo-MOF': ['nico-mof'],
        'Mg-Ce': ['mg₉₅ce₅']
    }
    
    for alloy_name, keywords in patterns.items():
        for keyword in keywords:
            if keyword in text:
                return alloy_name
    
    # Generic patterns
    if 'mg' in text and ('based' in text or 'alloy' in text):
        return 'Mg-based alloy'
    elif 'ti' in text and 'fe' in text:
        return 'Ti-Fe alloy'
    elif 'zr' in text and 'co' in text:
        return 'Zr-Co alloy'
    
    return None

def extract_synthesis_method(abstract):
    """Extract synthesis method from abstract."""
    text = abstract.lower()
    
    methods = {
        'Mechanical alloying': ['mechanical alloying', 'ball milling', 'ball-milled'],
        'Arc melting': ['arc melting', 'arc-melted'],
        'Induction melting': ['induction melting', 'vacuum induction'],
        'DFT calculation': ['dft', 'first-principles', 'density functional'],
        'High-pressure torsion': ['high-pressure torsion', 'hpt'],
        'Melt spinning': ['melt-spun', 'melt spinning'],
        'Sputtering': ['sputtering', 'sputter'],
        'CALPHAD modeling': ['calphad'],
        'Vapor deposition': ['vapor deposition'],
        'Chemical synthesis': ['chemical synthesis']
    }
    
    for method, keywords in methods.items():
        for keyword in keywords:
            if keyword in text:
                return method
    
    return None

def extract_operating_conditions(abstract):
    """Extract operating conditions from abstract."""
    text = abstract.lower()
    conditions = []
    
    # Temperature patterns
    temp_patterns = [
        r'(\d+)\s*°c',
        r'(\d+)\s*℃',
        r'at\s*(\d+)\s*k(?:\s|$|,)',
        r'temperature.*?(\d+)'
    ]
    
    temperatures = []
    for pattern in temp_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            temp = int(match)
            if 20 <= temp <= 1000:
                temperatures.append(temp)
    
    if temperatures:
        # Use the most commonly mentioned temperature or average
        avg_temp = sum(temperatures) / len(temperatures)
        conditions.append(f"{int(avg_temp)}°C")
    
    # Pressure patterns
    pressure_patterns = [
        r'(\d+\.?\d*)\s*bar',
        r'(\d+\.?\d*)\s*mpa',
        r'(\d+\.?\d*)\s*atm'
    ]
    
    pressures = []
    for pattern in pressure_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                pressure = float(match)
                if 0.1 <= pressure <= 1000:
                    pressures.append(pressure)
            except ValueError:
                continue
    
    if pressures:
        avg_pressure = sum(pressures) / len(pressures)
        conditions.append(f"{avg_pressure:.1f} bar")
    
    return ', '.join(conditions) if conditions else None

def extract_advantages(abstract):
    """Extract advantages from abstract."""
    text = abstract.lower()
    advantages = []
    
    advantage_patterns = {
        'High storage capacity': ['high.*capacity', 'high.*storage', 'superior.*capacity'],
        'Fast kinetics': ['fast.*kinetics', 'rapid.*absorption', 'quick.*desorption', 'enhanced.*kinetics'],
        'Low operating temperature': ['low.*temperature', 'reduced.*temperature', 'room temperature'],
        'Good reversibility': ['reversible', 'reversibility', 'cycling'],
        'Cost-effective': ['cost-effective', 'low cost', 'inexpensive'],
        'High efficiency': ['efficient', 'effectiveness', 'enhanced.*performance']
    }
    
    for advantage, patterns in advantage_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text):
                advantages.append(advantage)
                break
    
    return ', '.join(advantages) if advantages else None

def extract_limitations(abstract):
    """Extract limitations from abstract."""
    text = abstract.lower()
    limitations = []
    
    limitation_patterns = {
        'High operating temperature required': ['high.*temperature.*required', 'elevated.*temperature'],
        'Slow kinetics': ['slow.*kinetics', 'sluggish.*kinetics', 'poor.*kinetics'],
        'Phase segregation': ['phase.*segregation', 'phase.*separation'],
        'Capacity degradation': ['degradation', 'capacity.*loss', 'decline'],
        'Thermodynamic stability issues': ['thermodynamic.*stability', 'stability.*challenge']
    }
    
    for limitation, patterns in limitation_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text):
                limitations.append(limitation)
                break
    
    return ', '.join(limitations) if limitations else None

def calculate_confidence_score(alloy_name, capacity, synthesis, conditions):
    """Calculate confidence score based on extracted data quality."""
    score = 0.3  # Base score
    
    if alloy_name:
        score += 0.2
    if capacity:
        score += 0.3
    if synthesis:
        score += 0.1
    if conditions:
        score += 0.1
    
    return min(score, 1.0)

def create_final_dataset():
    """Create the final dataset with all extracted information."""
    print("🚀 Creating Final Dataset for Hydrogen Storage Alloy Literature Mining")
    print("=" * 80)
    
    # Load the Scopus data
    df = pd.read_csv("data/scopus_export.csv")
    print(f"📊 Loaded {len(df)} papers from Scopus export")
    
    # Process each paper
    results = []
    
    for idx, row in df.iterrows():
        paper_id = str(row['id'])
        title = row['Title']
        abstract = row['Abstract']
        
        print(f"📄 Processing Paper {paper_id}: {title[:60]}...")
        
        # Extract structured data
        alloy_name = extract_alloy_name(title, abstract)
        capacity = extract_storage_capacity(abstract)
        synthesis = extract_synthesis_method(abstract)
        conditions = extract_operating_conditions(abstract)
        advantages = extract_advantages(abstract)
        limitations = extract_limitations(abstract)
        
        # Calculate confidence score
        confidence = calculate_confidence_score(alloy_name, capacity, synthesis, conditions)
        
        # Create capacity note
        capacity_note = None
        if capacity:
            capacity_note = f"Extracted from abstract: {capacity} wt%"
        
        # Create result record
        result = {
            'id': paper_id,
            'title': title,
            'authors': row['Authors'],
            'year': row['Year'],
            'doi': row['DOI'],
            'abstract': abstract,
            'pdf_path': row['Pdf_path'],
            'alloy_name': alloy_name,
            'storage_capacity_wt_percent': capacity,
            'storage_capacity_note': capacity_note,
            'synthesis_method': synthesis,
            'operating_conditions': conditions,
            'advantages': advantages,
            'limitations': limitations,
            'extracted_from': 'rule_based_extraction',
            'confidence_score': confidence,
            'raw_text_path': f'data/raw_text/{paper_id}.txt'
        }
        
        results.append(result)
        
        # Show extraction summary
        extracted_fields = []
        if alloy_name:
            extracted_fields.append(f"Alloy: {alloy_name}")
        if capacity:
            extracted_fields.append(f"Capacity: {capacity} wt%")
        if synthesis:
            extracted_fields.append(f"Synthesis: {synthesis}")
        if conditions:
            extracted_fields.append(f"Conditions: {conditions}")
        
        if extracted_fields:
            print(f"   ✅ Extracted: {'; '.join(extracted_fields)}")
            print(f"   📊 Confidence: {confidence:.2f}")
        else:
            print(f"   ⚠️  Limited extraction - only basic metadata")
    
    # Create final dataframe
    final_df = pd.DataFrame(results)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Save as CSV
    csv_path = "outputs/final_dataset.csv"
    final_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Save as JSON
    json_path = "outputs/final_dataset.json"
    final_df.to_json(json_path, orient='records', indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"   📄 CSV: {csv_path}")
    print(f"   📄 JSON: {json_path}")
    
    # Generate summary statistics
    print_summary_statistics(final_df)
    
    return final_df

def print_summary_statistics(df):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("📊 PIPELINE SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n📋 DATASET OVERVIEW:")
    print(f"Total papers processed: {len(df)}")
    print(f"Papers from 2024: {len(df[df['year'] == 2024])}")
    print(f"Papers from 2025: {len(df[df['year'] == 2025])}")
    
    print(f"\n🔬 EXTRACTION RESULTS:")
    extraction_fields = [
        'alloy_name', 'storage_capacity_wt_percent', 'synthesis_method', 
        'operating_conditions', 'advantages', 'limitations'
    ]
    
    for field in extraction_fields:
        count = df[field].notna().sum()
        percentage = (count / len(df)) * 100
        print(f"{field.replace('_', ' ').title()}: {count} papers ({percentage:.1f}%)")
    
    # Confidence score distribution
    mean_confidence = df['confidence_score'].mean()
    high_confidence = len(df[df['confidence_score'] >= 0.7])
    medium_confidence = len(df[(df['confidence_score'] >= 0.5) & (df['confidence_score'] < 0.7)])
    low_confidence = len(df[df['confidence_score'] < 0.5])
    
    print(f"\n📈 CONFIDENCE DISTRIBUTION:")
    print(f"Average confidence score: {mean_confidence:.2f}")
    print(f"High confidence (≥0.7): {high_confidence} papers ({high_confidence/len(df)*100:.1f}%)")
    print(f"Medium confidence (0.5-0.7): {medium_confidence} papers ({medium_confidence/len(df)*100:.1f}%)")
    print(f"Low confidence (<0.5): {low_confidence} papers ({low_confidence/len(df)*100:.1f}%)")
    
    # Storage capacity statistics
    capacity_papers = df[df['storage_capacity_wt_percent'].notna()]
    if len(capacity_papers) > 0:
        print(f"\n⚡ STORAGE CAPACITY ANALYSIS:")
        print(f"Papers with capacity data: {len(capacity_papers)}")
        print(f"Average capacity: {capacity_papers['storage_capacity_wt_percent'].mean():.2f} wt%")
        print(f"Capacity range: {capacity_papers['storage_capacity_wt_percent'].min():.2f} - {capacity_papers['storage_capacity_wt_percent'].max():.2f} wt%")
    
    # Show top extractions
    print(f"\n🎯 TOP EXTRACTIONS (Highest Confidence):")
    top_papers = df.nlargest(5, 'confidence_score')
    
    for idx, (_, row) in enumerate(top_papers.iterrows(), 1):
        print(f"\n{idx}. Paper {row['id']}: {row['title'][:60]}...")
        print(f"   📅 Year: {row['year']}")
        if row['alloy_name']:
            print(f"   🔬 Alloy: {row['alloy_name']}")
        if row['storage_capacity_wt_percent']:
            print(f"   ⚡ Capacity: {row['storage_capacity_wt_percent']} wt%")
        if row['synthesis_method']:
            print(f"   🛠️  Synthesis: {row['synthesis_method']}")
        if row['operating_conditions']:
            print(f"   🌡️  Conditions: {row['operating_conditions']}")
        print(f"   📊 Confidence: {row['confidence_score']:.2f}")
    
    print(f"\n✅ Final dataset created successfully!")
    print(f"Ready for validation and further analysis.")
    print("="*80)

if __name__ == "__main__":
    final_df = create_final_dataset()
    
    # Create a simple validation summary
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📁 Output files created in 'outputs/' directory")
    print(f"📊 {len(final_df)} papers processed with structured data extraction")
    print(f"🚀 Pipeline execution completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
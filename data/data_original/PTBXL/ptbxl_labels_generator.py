import pandas as pd
import re

def create_ptbxl_labels():
    """
    Create labels.csv file for PTBXL dataset by processing header information
    and mapping SNOMED codes to custom class labels.
    """
    
    # Read the G1 dataset containing header file information
    print("Reading PTBXL_g1.csv...")
    g1_df = pd.read_csv('PTBXL_g1.csv')
    
    # Read the class labels mapping
    print("Reading PTBXL_class_labels.csv...")
    class_labels_df = pd.read_csv('PTBXL_class_labels.csv')
    
    # Create SNOMED code to class label mapping
    snomed_to_class = {}
    for _, row in class_labels_df.iterrows():
        if row['SNOMED_code'] != 'none':
            snomed_to_class[str(row['SNOMED_code'])] = row['class_code']
    
    print("SNOMED to Class mapping:")
    for snomed, class_code in snomed_to_class.items():
        description = class_labels_df[class_labels_df['class_code'] == class_code]['description'].iloc[0]
        print(f"  {snomed} -> Class {class_code} ({description})")
    
    # Extract file numbers from filenames and sort
    def extract_file_number(filename):
        """Extract number from filename like 'HR00506.hea' -> 506"""
        match = re.search(r'HR(\d+)\.hea', filename)
        return int(match.group(1)) if match else 0
    
    # Add file number column
    g1_df['file_number'] = g1_df['filename'].apply(extract_file_number)
    
    # Sort by file number
    g1_df = g1_df.sort_values('file_number').reset_index(drop=True)
    
    # Map SNOMED codes to class labels
    def map_dx_to_class(dx_codes):
        """Map Dx codes to class labels based on SNOMED mapping"""
        if pd.isna(dx_codes):
            return 6  # Default to "Other arrhythmias"
        
        # Split multiple codes and clean them
        codes = str(dx_codes).split(',')
        codes = [code.strip() for code in codes]
        
        # Find first matching code in our mapping
        for code in codes:
            if code in snomed_to_class:
                return snomed_to_class[code]
        
        # If no match found, assign to "Other arrhythmias" class
        return 6
    
    # Add class_label column
    g1_df['class_label'] = g1_df['Dx'].apply(map_dx_to_class)
    
    # Select and reorder columns for final output
    labels_df = g1_df[['filename', 'file_number', 'Age', 'Sex', 'Dx', 'class_label']].copy()
    
    # Save to CSV
    labels_df.to_csv('labels.csv', index=False)
    
    # Print summary statistics
    print(f"\nProcessed {len(labels_df)} records")
    print(f"File numbers range from {labels_df['file_number'].min()} to {labels_df['file_number'].max()}")
    
    print("\nClass distribution:")
    class_counts = labels_df['class_label'].value_counts().sort_index()
    
    for class_code in sorted(class_counts.index):
        count = class_counts[class_code]
        description = class_labels_df[class_labels_df['class_code'] == class_code]['description'].iloc[0]
        percentage = (count / len(labels_df)) * 100
        print(f"  Class {class_code} ({description}): {count} samples ({percentage:.1f}%)")
    
    print(f"\nLabels saved to 'labels.csv'")
    print("\nFirst 10 records:")
    print(labels_df.head(10)[['filename', 'file_number', 'class_label']])
    
    return labels_df

# Run the function
if __name__ == "__main__":
    labels_df = create_ptbxl_labels()
    
    # Additional verification
    print("\nVerification - Sample records by class:")
    for class_id in sorted(labels_df['class_label'].unique()):
        samples = labels_df[labels_df['class_label'] == class_id].head(3)
        print(f"\nClass {class_id} examples:")
        for _, row in samples.iterrows():
            print(f"  {row['filename']}: {row['Dx']}")

#!/usr/bin/env python3

# Script to run pose preprocessing on the sample data

import os
import sys

# Add the current directory to the Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pose_preprocessor import preprocess_all_data

def main():
    # Print the start message for preprocessing
    print("Starting pose data preprocessing...")
    print("=" * 50)
    
    # Define paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(script_dir, "Sample Data Points")
    output_directory = os.path.join(script_dir, "Preprocessed Data")
    scaler_filepath = os.path.join(output_directory, "pose_scaler.pkl")
    
    # Check if the input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' not found!")
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    try:
        # Run the preprocessing function
        features, metadata = preprocess_all_data(input_directory, output_directory, scaler_filepath)
        
        # Print the preprocessing summary
        print("\n" + "=" * 50)
        print("✅ Preprocessing completed successfully!")
        print("=" * 50)
        print(f"📊 Features shape: {features.shape}")
        print(f"📋 Metadata shape: {metadata.shape}")
        print(f"📁 Output directory: {output_directory}")
        print(f"⚙️  Scaler saved to: {scaler_filepath}")
        
        print("\n📈 Preprocessing Summary:")
        print(f"   • Total poses processed: {len(features)}")
        print(f"   • Feature vector size: {features.shape[1]}")
        print(f"   • Average confidence: {metadata['average_confidence'].mean():.3f}")
        print(f"   • Average visible keypoints: {metadata['visible_keypoints'].mean():.1f}")
        
        print("\nFirst few rows of metadata:")
        print(metadata.head())
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

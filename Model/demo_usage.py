#!/usr/bin/env python3

# Demo script showing how to use the preprocessed pose data

import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from pose_preprocessor import PosePreprocessor

def load_preprocessed_data():
    # Load all preprocessed data including features, metadata, and scaler
    print("Loading preprocessed data...")
    
    # Load feature vectors from the saved .npy file
    features = np.load('Preprocessed Data/all_processed_features.npy')
    
    # Load metadata from the saved CSV file
    metadata = pd.read_csv('Preprocessed Data/metadata.csv')
    
    # Load the scaler using the PosePreprocessor class
    preprocessor = PosePreprocessor()
    preprocessor.load_scaler('Preprocessed Data/pose_scaler.pkl')
    
    print(f"Loaded {len(features)} pose feature vectors")
    print(f"Feature vector size: {features.shape[1]}")
    
    return features, metadata, preprocessor

def compare_poses(features, metadata):
    # Compare poses using distance metrics
    print("\nPose Comparison Analysis:")
    print("=" * 40)
    
    # Calculate pairwise distances and similarities
    euclidean_dist = euclidean_distances(features)
    cosine_sim = cosine_similarity(features)
    
    n_poses = len(features)
    
    # Iterate through pairs of poses to find similarities and differences
    for i in range(n_poses):
        for j in range(i+1, n_poses):
            euc_dist = euclidean_dist[i, j]
            cos_sim = cosine_sim[i, j]
            
            frame_i = metadata.iloc[i]['frame_number']
            frame_j = metadata.iloc[j]['frame_number']
            score_i = metadata.iloc[i]['posture_score']
            score_j = metadata.iloc[j]['posture_score']
            
            print(f"Frame {frame_i} vs Frame {frame_j}:")
            print(f"  Euclidean Distance: {euc_dist:.3f}")
            print(f"  Cosine Similarity: {cos_sim:.3f}")
            print(f"  Posture Scores: {score_i} vs {score_j}")
            print()

def classify_pose_quality(features, metadata):
    # Classify poses based on quality metrics
    print("Pose Quality Classification:")
    print("=" * 35)
    
    for i, (_, row) in enumerate(metadata.iterrows()):
        frame_num = row['frame_number']
        visible_kp = row['visible_keypoints']
        avg_conf = row['average_confidence']
        posture_score = row['posture_score']
        
        # Simple quality classification based on thresholds
        if avg_conf > 0.5 and visible_kp >= 7:
            quality = "High Quality"
        elif avg_conf > 0.3 and visible_kp >= 5:
            quality = "Medium Quality"
        else:
            quality = "Low Quality"
        
        print(f"Frame {frame_num}: {quality}")
        print(f"  Visible Keypoints: {visible_kp}/17")
        print(f"  Average Confidence: {avg_conf:.3f}")
        print(f"  Posture Score: {posture_score}")
        print()

def demonstrate_new_pose_processing(preprocessor):
    # Show how to process a new incoming pose
    print("Processing New Pose Demo:")
    print("=" * 30)
    
    # Load a sample pose JSON file for demonstration
    with open('Sample Data Points/keypoints_20250825_145506_frame_166.json', 'r') as f:
        sample_pose = json.load(f)
    
    # Process the pose using the preprocessor
    feature_vector, metadata = preprocessor.transform(sample_pose)
    
    print(f"Processed pose from frame {sample_pose['frame_number']}")
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Metadata extracted:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

def plot_feature_analysis(features, metadata):
    # Create visualizations of the feature data
    print("\nCreating feature analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pose Feature Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature vector magnitudes
    feature_norms = np.linalg.norm(features, axis=1)
    axes[0, 0].bar(range(len(feature_norms)), feature_norms)
    axes[0, 0].set_title('Feature Vector Magnitudes')
    axes[0, 0].set_xlabel('Pose Index')
    axes[0, 0].set_ylabel('L2 Norm')
    
    # Plot 2: Confidence vs Visible Keypoints
    axes[0, 1].scatter(metadata['visible_keypoints'], metadata['average_confidence'], 
                      c=metadata['posture_score'], cmap='viridis', s=100)
    axes[0, 1].set_title('Confidence vs Visible Keypoints')
    axes[0, 1].set_xlabel('Visible Keypoints')
    axes[0, 1].set_ylabel('Average Confidence')
    
    # Plot 3: Feature distribution (first 10 features)
    feature_subset = features[:, :10]
    im = axes[1, 0].imshow(feature_subset.T, aspect='auto', cmap='coolwarm')
    axes[1, 0].set_title('Feature Distribution (First 10 Features)')
    axes[1, 0].set_xlabel('Pose Index')
    axes[1, 0].set_ylabel('Feature Index')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 4: Pairwise similarities
    cosine_sim = cosine_similarity(features)
    im2 = axes[1, 1].imshow(cosine_sim, cmap='RdYlBu')
    axes[1, 1].set_title('Pairwise Cosine Similarities')
    axes[1, 1].set_xlabel('Pose Index')
    axes[1, 1].set_ylabel('Pose Index')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('Preprocessed Data/feature_analysis.png', dpi=300, bbox_inches='tight')
    print("Feature analysis plot saved to 'Preprocessed Data/feature_analysis.png'")

def main():
    # Main function to demonstrate the pose preprocessing pipeline
    print("Pose Preprocessing Demo")
    print("=" * 50)
    
    try:
        # Load preprocessed data
        features, metadata, preprocessor = load_preprocessed_data()
        
        # Analyze pose similarities
        compare_poses(features, metadata)
        
        # Classify pose quality
        classify_pose_quality(features, metadata)
        
        # Demonstrate new pose processing
        demonstrate_new_pose_processing(preprocessor)
        
        # Create visualizations
        plot_feature_analysis(features, metadata)
        
        print("\nDemo completed successfully!")
        print("Your pose preprocessing pipeline is ready for ML model training!")
        
    except Exception as e:
        print(f"Error in demo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

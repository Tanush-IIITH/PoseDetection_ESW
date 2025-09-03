# Import necessary libraries for numerical computations, data handling, and machine learning preprocessing
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional
import os
import pickle
from datetime import datetime

class PosePreprocessor:
    # Initialize the PosePreprocessor class with default settings
    def __init__(self):
        # StandardScaler for normalizing feature vectors
        self.scaler = StandardScaler()
        # Flag to check if the scaler is fitted
        self.is_fitted = False
        # Dictionary to store reference poses for comparison
        self.reference_poses = {}
        # List of keypoint names used in the pose data
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    # Load pose data from a JSON file
    def load_json_data(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}

    # Extract keypoints and confidence scores from the JSON data
    def extract_keypoints_from_json(self, pose_json: Dict) -> np.ndarray:
        keypoints = []
        confidence_scores = []
        
        if 'keypoints' not in pose_json:
            return np.array([]), np.array([])
        
        # Iterate through each keypoint name and extract its coordinates and confidence
        for keypoint_name in self.keypoint_names:
            if keypoint_name in pose_json['keypoints']:
                kp_data = pose_json['keypoints'][keypoint_name]
                keypoints.extend([kp_data.get('x', 0), kp_data.get('y', 0)])
                confidence_scores.append(kp_data.get('confidence', 0))
            else:
                keypoints.extend([0, 0])  # Assign default values for missing keypoints
                confidence_scores.append(0)
        
        return np.array(keypoints), np.array(confidence_scores)

    # Normalize pose keypoints relative to the body center and scale
    def normalize_pose(self, keypoints: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        if len(keypoints) == 0:
            return keypoints
        
        # Reshape the keypoints array for easier processing
        kp_reshaped = keypoints.reshape(-1, 2)
        
        # Extract x and y coordinates
        x_coords = kp_reshaped[:, 0]
        y_coords = kp_reshaped[:, 1]
        
        # Filter out low-confidence keypoints
        valid_mask = confidence > 0.3
        
        if np.sum(valid_mask) == 0:
            return keypoints
        
        # Calculate the body center as the average of valid keypoints
        center_x = np.mean(x_coords[valid_mask])
        center_y = np.mean(y_coords[valid_mask])
        
        # Normalize coordinates relative to the body center
        x_coords_norm = x_coords - center_x
        y_coords_norm = y_coords - center_y
        
        # Scale normalization using the distance between shoulders as a reference
        if len(kp_reshaped) > 6 and confidence[5] > 0.3 and confidence[6] > 0.3:
            shoulder_dist = np.sqrt((kp_reshaped[5, 0] - kp_reshaped[6, 0])**2 + 
                                  (kp_reshaped[5, 1] - kp_reshaped[6, 1])**2)
            if shoulder_dist > 0:
                scale_factor = 100 / shoulder_dist  # Normalize to a standard shoulder width
                x_coords_norm *= scale_factor
                y_coords_norm *= scale_factor
        
        # Reconstruct the normalized keypoints array
        normalized_kp = np.column_stack([x_coords_norm, y_coords_norm])
        return normalized_kp.flatten()

    # Calculate joint angles as additional features for the pose
    def calculate_angles(self, keypoints: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        kp_reshaped = keypoints.reshape(-1, 2)
        angles = []
        
        # Define joint triplets for angle calculations
        joint_triplets = [
            (4, 5, 6, "left_shoulder_angle"),    # left_ear, left_shoulder, right_shoulder
            (3, 6, 5, "right_shoulder_angle"),   # right_ear, right_shoulder, left_shoulder
            (5, 7, 9, "left_elbow_angle"),       # left_shoulder, left_elbow, left_wrist
            (6, 8, 10, "right_elbow_angle"),     # right_shoulder, right_elbow, right_wrist
            (5, 11, 13, "left_hip_angle"),       # left_shoulder, left_hip, left_knee
            (6, 12, 14, "right_hip_angle"),      # right_shoulder, right_hip, right_knee
            (11, 13, 15, "left_knee_angle"),     # left_hip, left_knee, left_ankle
            (12, 14, 16, "right_knee_angle")     # right_hip, right_knee, right_ankle
        ]
        
        for p1, p2, p3, angle_name in joint_triplets:
            if (p1 < len(kp_reshaped) and p2 < len(kp_reshaped) and p3 < len(kp_reshaped)):
                angle = self._calculate_angle(kp_reshaped[p1], kp_reshaped[p2], kp_reshaped[p3],
                                            confidence[p1], confidence[p2], confidence[p3])
                angles.append(angle)
            else:
                angles.append(0)  # Default value for missing or low-confidence keypoints
        
        return np.array(angles)

    # Helper function to calculate the angle between three points
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, 
                        c1: float, c2: float, c3: float) -> float:
        if c1 < 0.3 or c2 < 0.3 or c3 < 0.3:  # Skip low-confidence points
            return 0
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Avoid division by zero
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range for arccos
        angle = np.arccos(cos_angle)
        return np.degrees(angle)

    # Create a comprehensive feature vector from the pose JSON data
    def create_feature_vector(self, pose_json: Dict) -> Tuple[np.ndarray, Dict]:
        keypoints, confidence = self.extract_keypoints_from_json(pose_json)
        
        if len(keypoints) == 0:
            return np.zeros(34 + 8), {}  # Default feature vector size
        
        # Normalize keypoints and calculate angles
        normalized_kp = self.normalize_pose(keypoints, confidence)
        angles = self.calculate_angles(normalized_kp, confidence)
        
        # Combine normalized keypoints and angles into a single feature vector
        feature_vector = np.concatenate([normalized_kp, angles])
        
        # Create metadata for the pose
        metadata = {
            'timestamp': pose_json.get('timestamp', ''),
            'frame_number': pose_json.get('frame_number', 0),
            'visible_keypoints': np.sum(confidence > 0.3),
            'average_confidence': np.mean(confidence[confidence > 0]),
            'posture_score': pose_json.get('posture_analysis', {}).get('score', 0)
        }
        
        return feature_vector, metadata

    # Fit the scaler on a list of pose data for normalization
    def fit_scaler(self, pose_data_list: List[Dict]):
        features = []
        for pose_json in pose_data_list:
            feature_vec, _ = self.create_feature_vector(pose_json)
            features.append(feature_vec)
        
        features_array = np.array(features)
        # Replace NaN or infinite values with zeros
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.scaler.fit(features_array)
        self.is_fitted = True
        print(f"Scaler fitted on {len(features)} samples")

    # Transform a single pose JSON into a standardized feature vector
    def transform(self, pose_json: Dict) -> Tuple[np.ndarray, Dict]:
        feature_vector, metadata = self.create_feature_vector(pose_json)
        
        # Replace NaN or infinite values with zeros
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.is_fitted:
            feature_vector = self.scaler.transform([feature_vector])[0]
        
        return feature_vector, metadata

    # Save the fitted scaler to a file
    def save_scaler(self, filepath: str):
        if self.is_fitted:
            with open(filepath, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler saved to {filepath}")
        else:
            print("Scaler not fitted yet!")

    # Load a pre-fitted scaler from a file
    def load_scaler(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True
            print(f"Scaler loaded from {filepath}")
        except Exception as e:
            print(f"Error loading scaler: {e}")

# Preprocess all JSON files in the input directory and save the results
def preprocess_all_data(input_dir: str, output_dir: str, scaler_path: str = None):
    # Initialize the PosePreprocessor
    preprocessor = PosePreprocessor()
    
    # Collect all JSON files from the input directory
    json_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files")
    
    # Load all pose data for fitting the scaler
    all_pose_data = []
    for json_file in json_files:
        pose_data = preprocessor.load_json_data(json_file)
        if pose_data:
            all_pose_data.append(pose_data)
    
    # Fit the scaler on the loaded data
    print("Fitting scaler...")
    preprocessor.fit_scaler(all_pose_data)
    
    # Save the fitted scaler if a path is provided
    if scaler_path:
        preprocessor.save_scaler(scaler_path)
    
    # Process each pose and save the results
    processed_data = []
    metadata_list = []
    
    for i, pose_data in enumerate(all_pose_data):
        feature_vector, metadata = preprocessor.transform(pose_data)
        
        processed_data.append(feature_vector)
        metadata_list.append(metadata)
        
        # Save the processed feature vector as a .npy file
        output_filename = f"processed_{os.path.basename(json_files[i]).replace('.json', '.npy')}"
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, feature_vector)
    
    # Save all processed feature vectors as a batch
    all_features = np.array(processed_data)
    batch_output_path = os.path.join(output_dir, "all_processed_features.npy")
    np.save(batch_output_path, all_features)
    
    # Save metadata as a CSV file
    metadata_df = pd.DataFrame(metadata_list)
    metadata_output_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_output_path, index=False)
    
    print(f"Preprocessing complete!")
    print(f"- Processed {len(processed_data)} poses")
    print(f"- Feature vector size: {all_features.shape[1]}")
    print(f"- Individual files saved to: {output_dir}")
    print(f"- Batch features saved to: {batch_output_path}")
    print(f"- Metadata saved to: {metadata_output_path}")
    
    return all_features, metadata_df


if __name__ == "__main__":
    # Define paths
    input_directory = "Sample Data Points"
    output_directory = "Preprocessed Data"
    scaler_filepath = os.path.join(output_directory, "pose_scaler.pkl")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Run preprocessing
    features, metadata = preprocess_all_data(input_directory, output_directory, scaler_filepath)
    
    print("\nPreprocessing Summary:")
    print(f"Features shape: {features.shape}")
    print(f"Metadata shape: {metadata.shape}")
    print("\nFirst few rows of metadata:")
    print(metadata.head())

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional


def get_coordinates_from_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Extract landmark coordinates from heatmaps using argmax.
    
    Args:
        heatmaps (torch.Tensor): Predicted heatmaps of shape (batch_size, num_landmarks, height, width)
        
    Returns:
        torch.Tensor: Coordinates of shape (batch_size, num_landmarks, 2) where each coordinate is (x, y)
    """
    batch_size, num_landmarks, height, width = heatmaps.shape
    
    # Flatten the spatial dimensions
    heatmaps_flat = heatmaps.view(batch_size, num_landmarks, -1)
    
    # Get the indices of the maximum values
    max_indices = torch.argmax(heatmaps_flat, dim=2)
    
    # Convert indices to 2D coordinates (y, x)
    y_coords = max_indices // width
    x_coords = max_indices % width
    
    # Stack coordinates to get (x, y) format - matching the dataset's keypoint format
    coords = torch.stack([x_coords, y_coords], dim=2)
    
    return coords


def calculate_landmark_distance(pred_coords: torch.Tensor, gt_coords: torch.Tensor) -> torch.Tensor:
    """
    Calculate Euclidean distance between predicted and ground truth landmarks.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (batch_size, num_landmarks, 2)
        gt_coords (torch.Tensor): Ground truth coordinates of shape (batch_size, num_landmarks, 2)
        
    Returns:
        torch.Tensor: Distances of shape (batch_size, num_landmarks)
    """
    return torch.sqrt(torch.sum((pred_coords - gt_coords) ** 2, dim=2))


def calculate_mean_average_precision(
    pred_coords: torch.Tensor, 
    gt_coords: torch.Tensor,
    threshold: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate mean average precision (mAP) at a specific threshold.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (batch_size, num_landmarks, 2)
        gt_coords (torch.Tensor): Ground truth coordinates of shape (batch_size, num_landmarks, 2)
        threshold (float): Distance threshold in pixels for a detection to be considered correct
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Per-landmark precision of shape (num_landmarks,)
            - Overall mean average precision (scalar)
    """
    # Calculate distances between predicted and ground truth landmarks
    distances = calculate_landmark_distance(pred_coords, gt_coords)
    
    # Check which landmarks are correctly detected (distance < threshold)
    correct_detections = (distances < threshold).float()
    
    # Calculate precision per landmark
    landmark_precision = correct_detections.mean(dim=0)
    
    # Calculate overall mAP
    map_score = correct_detections.mean()
    
    return landmark_precision, map_score


def evaluate_landmarks(
    pred_heatmaps: torch.Tensor, 
    gt_keypoints: torch.Tensor,
    threshold: float = 5.0
) -> Dict[str, Union[float, torch.Tensor, List[float]]]:
    """
    Evaluate landmark detection performance with simplified metrics.
    
    Args:
        pred_heatmaps (torch.Tensor): Predicted heatmaps of shape (batch_size, num_landmarks, height, width)
        gt_keypoints (torch.Tensor): Ground truth keypoints of shape (batch_size, num_landmarks, 2)
        threshold (float): Distance threshold in pixels for mAP calculation
        
    Returns:
        Dict[str, Union[float, torch.Tensor, List[float]]]: Dictionary with evaluation metrics
    """
    # Extract coordinates from heatmaps
    pred_coords = get_coordinates_from_heatmaps(pred_heatmaps)
    
    # Calculate pixel distances
    distances = calculate_landmark_distance(pred_coords, gt_keypoints)
    
    # Calculate mean distance (overall and per landmark)
    mean_distance = distances.mean().item()
    per_landmark_distance = distances.mean(dim=0)
    
    # Calculate mean average precision
    landmark_precision, map_score = calculate_mean_average_precision(
        pred_coords, gt_keypoints, threshold
    )
    
    # Prepare results dictionary
    results = {
        'mean_distance': mean_distance,
        'per_landmark_distance': per_landmark_distance.tolist(),
        'map_score': map_score.item(),
        'landmark_precision': landmark_precision.tolist(),
        'threshold': threshold
    }
    
    return results


def print_evaluation_summary(results: Dict[str, Union[float, torch.Tensor, List[float]]]) -> None:
    """
    Print a summary of evaluation results.
    
    Args:
        results (Dict[str, Union[float, torch.Tensor, List[float]]]): Results from evaluate_landmarks
    """
    landmark_names = [
        'Left Eye', 'Right Eye', 'Mouth',
        'Left Ear-1', 'Left Ear-2', 'Left Ear-3',
        'Right Ear-1', 'Right Ear-2', 'Right Ear-3'
    ]
    
    print("\n===== Landmark Detection Evaluation =====")
    print(f"Mean Distance: {results['mean_distance']:.2f} pixels")
    print(f"Mean Average Precision (threshold={results['threshold']}px): {results['map_score']:.4f}")
    
    print("\nPer-Landmark Results:")
    print("-" * 50)
    print(f"{'Landmark':<15} {'Distance (px)':<15} {'Precision':<10}")
    print("-" * 50)
    
    for i, name in enumerate(landmark_names):
        if i < len(results['per_landmark_distance']):
            distance = results['per_landmark_distance'][i]
            precision = results['landmark_precision'][i]
            print(f"{name:<15} {distance:.2f}           {precision:.4f}")


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    num_landmarks = 9
    height, width = 256, 256
    
    # Create dummy data
    pred_heatmaps = torch.rand(batch_size, num_landmarks, height, width)
    gt_keypoints = torch.randint(0, 256, (batch_size, num_landmarks, 2)).float()
    
    # Evaluate
    results = evaluate_landmarks(pred_heatmaps, gt_keypoints)
    
    # Print summary
    print_evaluation_summary(results)

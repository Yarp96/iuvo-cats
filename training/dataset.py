import os
from sys import exception
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from typing import List, Tuple, Dict, Optional, Union
from scipy.ndimage import gaussian_filter
from augmentation import get_train_transforms, get_test_transforms

class CatDataset(Dataset):
    """
    Dataset for loading cat images and facial landmarks.
    
    The .cat files contain coordinates for 9 facial landmarks:
    1. Left Eye
    2. Right Eye
    3. Mouth
    4. Left Ear-1
    5. Left Ear-2
    6. Left Ear-3
    7. Right Ear-1
    8. Right Ear-2
    9. Right Ear-3
    """
    
    def __init__(self, root_dir: str, transform=None, train: bool = True, split_ratio: float = 0.8, heatmap_sigma: float = 5):
        """
        Initialize the CatDataset.
        
        Args:
            root_dir (str): Directory containing the cat dataset
            transform: Optional transform to be applied on the images
            train (bool): If True, load training set, else load test set
            split_ratio (float): Ratio of data to use for training (0.0 to 1.0)
            heatmap_sigma (float): Sigma for Gaussian filter when creating heatmaps
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.split_ratio = split_ratio
        self.heatmap_sigma = heatmap_sigma
        
        # Get all cat directories
        self.cat_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('CAT_')]
        
        # Get all image and annotation pairs
        all_samples = []
        for cat_dir in self.cat_dirs:
            cat_path = os.path.join(root_dir, cat_dir)
            cat_files = glob.glob(os.path.join(cat_path, '*.cat'))
            
            for cat_file in cat_files:
                img_file = cat_file[:-4]  # Remove .cat extension
                if os.path.exists(img_file):
                    all_samples.append((img_file, cat_file))
        
        # Split into train and test sets
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(all_samples))
        split_idx = int(len(indices) * split_ratio)
        
        if train:
            self.samples = [all_samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [all_samples[i] for i in indices[split_idx:]]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def _create_heatmap(self, coordinates, shape):
        """
        Create a Gaussian heatmap for multiple motor positions.
        
        Args:
            coordinates (list): List of (y, x) tuples for motor positions.
            shape (tuple): Shape of the output heatmap (height, width).
            
        Returns:
            numpy.ndarray: Gaussian heatmap with peaks at all motor positions.
        """
        # For negative examples, return an empty heatmap
        if not coordinates:
            return np.zeros(shape, dtype=np.float32)
        
        # Create a blank heatmap
        heatmap = np.zeros(shape, dtype=np.float32)
        
        # Add each motor position to the heatmap
        for y_center, x_center in coordinates:
            # Check if coordinates are within the image bounds
            if 0 <= y_center < shape[0] and 0 <= x_center < shape[1]:
                # Place a single 1 at the center position
                heatmap[int(y_center), int(x_center)] = 1.0
        
        # Apply Gaussian filter to create the heatmap
        heatmap = gaussian_filter(heatmap, sigma=self.heatmap_sigma)
        
        # Normalize the heatmap to have a maximum value of 1
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to get
                
            Returns:
            dict: A dictionary containing the image and landmarks/heatmaps
        """
        img_path, cat_path = self.samples[idx]
        
        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load landmarks as a list of (x, y) tuples
        landmarks = self._load_cat_file(cat_path)
        
        # Apply transformations if specified
        if self.transform:
            transformed = self.transform(image=image, keypoints=landmarks)
            image = transformed['image']
            keypoints = transformed['keypoints']
        else:
            raise Exception("Transform is not specified")
            
        # Get image dimensions after transformation
        img_height, img_width = image.shape[1], image.shape[2]
        
        # Create heatmaps for each landmark after transformation
        heatmaps = []
        for i, landmark in enumerate(keypoints):
            # Convert to (y, x) format for the heatmap function
            y, x = landmark[1], landmark[0]
            # Create a heatmap for this landmark
            heatmap = self._create_heatmap([(y, x)], (img_height, img_width))
            heatmaps.append(heatmap)

        # Stack heatmaps along a new dimension
        heatmaps = np.stack(heatmaps, axis=2)
        
        return {
            'image': image,
            'keypoints': torch.from_numpy(np.array(keypoints)).float(),
            'heatmaps': torch.from_numpy(heatmaps).permute(2, 0, 1).float(),
        }
    
    def _load_cat_file(self, cat_path: str) -> list:
        """
        Load landmarks from a .cat file.
        
        Args:
            cat_path (str): Path to the .cat file
            
        Returns:
            list: List of landmark coordinates as (x, y) tuples
        """
        with open(cat_path, 'r') as f:
            content = f.read().strip().split()
        
        # First value is the number of points (should be 9)
        num_points = int(content[0])
        assert num_points == 9, f"Expected 9 points, got {num_points} in {cat_path}"
        
        # Rest of the values are x,y coordinates
        flat_landmarks = np.array([float(x) for x in content[1:]], dtype=np.float32)
        
        # Convert to list of (x, y) tuples
        landmarks_list = []
        for i in range(0, len(flat_landmarks), 2):
            x, y = flat_landmarks[i], flat_landmarks[i+1]
            landmarks_list.append((x, y))
        
        return landmarks_list


def get_cat_dataloader(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a DataLoader for the CatDataset.
    
    Args:
        root_dir (str): Directory containing the cat dataset
        batch_size (int): Batch size for the DataLoader
        train (bool): If True, load training set, else load test set
        transform: Optional transform to be applied on the images
        num_workers (int): Number of workers for the DataLoader
        shuffle (bool): If True, shuffle the dataset
        
    Returns:
        DataLoader: DataLoader for the CatDataset
    """
    train_dataset = CatDataset(root_dir=root_dir, transform=get_train_transforms(), train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)

    val_dataset = CatDataset(root_dir=root_dir, transform=get_test_transforms(), train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    
    return train_dataloader, val_dataloader


def visualize_sample(sample: Dict[str, torch.Tensor], save_path: Optional[str] = None) -> None:
    """
    Visualize a sample from the dataset.
    
    Args:
        sample (dict): Sample from the dataset
        save_path (str, optional): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    # Convert tensor to numpy array for visualization
    if isinstance(sample['image'], torch.Tensor):
        image = sample['image'].numpy().transpose((1, 2, 0))
    else:
        image = sample['image']
    
    # Convert keypoints tensor to numpy array of (x, y) points
    if isinstance(sample['keypoints'], torch.Tensor):
        landmarks = sample['keypoints'].numpy()
    else:
        landmarks = sample['keypoints']
    
    print(f"Landmarks shape in visualize_sample: {landmarks.shape}")
    
    # Define colors for different landmark groups
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange']
    landmark_names = [
        'Left Eye', 'Right Eye', 'Mouth',
        'Left Ear-1', 'Left Ear-2', 'Left Ear-3',
        'Right Ear-1', 'Right Ear-2', 'Right Ear-3'
    ]
    
    # Limit the number of landmarks to visualize based on what's available
    num_landmarks = min(len(landmark_names), landmarks.shape[0])
    landmark_names = landmark_names[:num_landmarks]
    colors = colors[:num_landmarks]
        
    # Create a figure with two subplots: original image with landmarks and combined heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image with landmarks
    ax1.imshow(image)
    
    # Plot landmarks on the original image
    for i, (name, color) in enumerate(zip(landmark_names, colors)):
        # Get the x, y coordinates from the landmarks
        if i < landmarks.shape[0]:  # Make sure we don't exceed the number of landmarks
            x, y = landmarks[i][0], landmarks[i][1]
            
            ax1.scatter(x, y, c=color, s=50)
            ax1.text(x + 5, y + 5, name, fontsize=9, color=color)
    
    ax1.set_title('Cat Facial Landmarks')
    ax1.axis('off')
    
    # Convert heatmaps tensor to numpy
    if isinstance(sample['heatmaps'], torch.Tensor):
        heatmaps = sample['heatmaps'].permute(1, 2, 0).numpy()
    else:
        heatmaps = sample['heatmaps']
    
    print(f"Heatmaps shape in visualize_sample: {heatmaps.shape}")
    
    # Sum the heatmaps along axis 2 (channel dimension)
    combined_heatmap = np.sum(heatmaps, axis=2)
    
    # Display the combined heatmap
    im = ax2.imshow(combined_heatmap, cmap='hot')
    ax2.set_title('Combined Heatmap')
    ax2.axis('off')
    
    plt.show()


if __name__ == "__main__":

    train_dataloader, val_dataloader = get_cat_dataloader(root_dir="cats", batch_size=16)

    print("Train dataset size:", len(train_dataloader.dataset))
    print("Validation dataset size:", len(val_dataloader.dataset))

    sample = val_dataloader.dataset[0]

    print("Image shape:", sample['image'].shape)
    print("Keypoints shape:", sample['keypoints'].shape)
    print("Heatmaps shape:", sample['heatmaps'].shape)

    visualize_sample(sample)
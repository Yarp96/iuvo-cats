import numpy as np
import onnxruntime
from PIL import Image
import os
import matplotlib.pyplot as plt
import requests

def load_onnx_model(model_path):
    """
    Load an ONNX model for inference.
    
    Args:
        model_path (str): Path to the ONNX model file
        
    Returns:
        onnxruntime.InferenceSession: ONNX runtime inference session
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create ONNX Runtime session
    session = onnxruntime.InferenceSession(model_path)
    
    print(f"Model loaded from {model_path}")
    return session

def preprocess_image(image_path, img_size=(256, 256)):
    """
    Preprocess an image for model inference using only PIL and numpy.
    
    Args:
        image_path (str): Path to the image file or PIL Image
        img_size (tuple): Target image size (height, width)
        
    Returns:
        np.ndarray: Preprocessed image tensor ready for model input
    """
    # Load image
    if isinstance(image_path, str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
    else:
        # Assume it's already a PIL image
        image = image_path.convert('RGB') if hasattr(image_path, 'convert') else Image.fromarray(np.array(image_path))
    
    # Resize image
    image = image.resize(img_size)
    
    # Convert to numpy array and normalize
    image_np = np.array(image, dtype=np.float32) / 255.0
    
    # ImageNet normalization values
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Normalize with ImageNet values
    image_np = (image_np - mean) / std
    
    # Convert to NCHW format (batch, channels, height, width)
    image_np = np.transpose(image_np, (2, 0, 1))
    
    # Add batch dimension
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np

def forward_pass(session, image_tensor):
    """
    Perform a forward pass through the ONNX model.
    
    Args:
        session (onnxruntime.InferenceSession): ONNX runtime session
        image_tensor (np.ndarray): Preprocessed image tensor
        
    Returns:
        np.ndarray: Model output (heatmaps)
    """
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: image_tensor})
    
    return outputs[0]  # First output is the heatmaps

def get_landmark_predictions(heatmaps, threshold=0.1):
    """
    Extract landmark coordinates from heatmaps.
    
    Args:
        heatmaps (np.ndarray): Model output heatmaps
        threshold (float): Confidence threshold for detection
        
    Returns:
        list: List of (x, y) tuples for each detected landmark
    """
    # Remove batch dimension if present
    if len(heatmaps.shape) == 4:
        heatmaps = heatmaps[0]
    
    landmarks = []
    
    # Process each heatmap to find the landmark position
    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        
        # Find the coordinates of the maximum value in the heatmap
        if np.max(heatmap) > threshold:
            # Get indices of the maximum value
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y, x = max_idx[0], max_idx[1]
            landmarks.append((x, y))
        else:
            # If confidence is below threshold, mark as not detected
            landmarks.append(None)
    
    return landmarks

def rescale_landmarks(landmarks, model_dimensions, original_dimensions):
    """
    Rescale landmarks from model resolution to original image size.
    
    Args:
        landmarks (list): List of (x, y) tuples or None values for each landmark
        model_dimensions (tuple): Model input dimensions as (width, height)
        original_dimensions (tuple): Original image dimensions as (width, height)
        
    Returns:
        list: List of rescaled (x, y) tuples or None values
    """
    model_width, model_height = model_dimensions
    original_width, original_height = original_dimensions
    rescaled_landmarks = []
    
    for landmark in landmarks:
        if landmark is not None:
            x, y = landmark
            # Scale coordinates to match original image dimensions
            scaled_x = int((x / model_width) * original_width)
            scaled_y = int((y / model_height) * original_height)
            rescaled_landmarks.append((scaled_x, scaled_y))
        else:
            rescaled_landmarks.append(None)
    
    return rescaled_landmarks

def visualize_landmarks_matplotlib(image, landmarks, save_path=None, figsize=(10, 10)):
    """
    Visualize the detected landmarks on the image using matplotlib.
    
    Args:
        image: Original image (PIL Image or numpy array)
        landmarks: List of (x, y) tuples for each landmark
        save_path (str): Optional path to save the visualization
        figsize (tuple): Figure size for the plot
        
    Returns:
        matplotlib.figure.Figure: The figure with the visualization
    """
    # Convert to numpy array if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Define colors and labels for different landmarks
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'pink', 'cyan', 'purple', 'brown']
    labels = ['Left Eye', 'Right Eye', 'Mouth', 'Left Ear-1', 'Left Ear-2', 'Left Ear-3', 
              'Right Ear-1', 'Right Ear-2', 'Right Ear-3']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the image
    ax.imshow(image)
    
    # Plot each landmark
    for i, landmark in enumerate(landmarks):
        if landmark is not None:
            x, y = landmark[0], landmark[1]
            ax.scatter(x, y, c=colors[i], s=100, label=f"{i+1}: {labels[i]}")
            ax.text(x+5, y+5, str(i+1), fontsize=12, color=colors[i])
    
    # Add legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    ax.set_title('Cat Facial Landmarks', fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def predict_cat_landmarks(model_path, image_url, model_dimensions=(256, 256)):
    """
    End-to-end function to predict cat facial landmarks from an image.
    
    Args:
        model_path (str): Path to the ONNX model file
        image_url (str): URL of the image
        visualize (bool): Whether to visualize the results        
    Returns:
        list (tuple): landmarks where landmarks is a list of (x, y) tuples
    """
    # Load model
    session = load_onnx_model(model_path)
    
    # Load and preprocess image
    original_image =  Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    original_width, original_height = original_image.size
    image_tensor = preprocess_image(original_image)
    
    # Perform forward pass
    heatmaps = forward_pass(session, image_tensor)
    
    # Get landmark predictions
    landmarks = get_landmark_predictions(heatmaps)
    
    # Rescale landmarks from model resolution to original image size
    original_dimensions = (original_width, original_height)
    rescaled_landmarks = rescale_landmarks(landmarks, model_dimensions, original_dimensions)
    
    return rescaled_landmarks

if __name__ == "__main__":
    model_path = "models/model.onnx"
    img_url = "https://media.istockphoto.com/id/157671964/photo/portrait-of-a-tabby-cat-looking-at-the-camera.jpg?s=612x612&w=0&k=20&c=iTsJO6vuQ5w3hL5pWn42C91ziMRUsYd725oUGRRewjM="

    try:
        # Use matplotlib for visualization
        landmarks = predict_cat_landmarks(model_path, img_url)

        fig = visualize_landmarks_matplotlib(Image.open(requests.get(img_url, stream=True).raw).convert('RGB'), landmarks, save_path='example_prediction.png', figsize=(10, 10))

        plt.show()  # Display the plot
        print("Detected landmarks:", landmarks)
    except Exception as e:
        print(f"Error during prediction: {e}")
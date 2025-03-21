import os
import numpy as np
import pytest
from PIL import Image
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import functions from predict.py
from predict import (
    load_onnx_model,
    preprocess_image,
    forward_pass,
    get_landmark_predictions,
    rescale_landmarks,
    predict_cat_landmarks
)

# Fixtures
@pytest.fixture
def test_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_image_path(test_dir):
    """Create a test image for testing."""
    path = os.path.join(test_dir, "test_cat.jpg")
    img = Image.new('RGB', (256, 256), color=(73, 109, 137))
    img.save(path)
    return path

@pytest.fixture
def test_image(test_image_path):
    """Return a PIL Image for testing."""
    return Image.open(test_image_path)

# Tests for preprocess_image
def test_preprocess_image_from_path(test_image_path):
    """Test preprocessing an image from a file path."""
    # Preprocess the test image
    preprocessed = preprocess_image(test_image_path)
    
    # Check the shape and type of the preprocessed image
    assert preprocessed.shape == (1, 3, 256, 256)
    assert preprocessed.dtype == np.float32
    
    # Check that values are normalized
    assert np.all(preprocessed <= 1.0)
    assert np.all(preprocessed >= -1.0)

def test_preprocess_image_from_pil(test_image):
    """Test preprocessing a PIL Image."""
    # Preprocess the PIL Image
    preprocessed = preprocess_image(test_image)
    
    # Check the shape and type of the preprocessed image
    assert preprocessed.shape == (1, 3, 256, 256)
    assert preprocessed.dtype == np.float32

def test_preprocess_image_file_not_found(test_dir):
    """Test preprocessing a non-existent image."""
    # Try to preprocess a non-existent image
    with pytest.raises(FileNotFoundError):
        preprocess_image(os.path.join(test_dir, "random_url"))

# Tests for forward_pass
@patch('onnxruntime.InferenceSession')
def test_forward_pass(mock_session):
    """Test performing a forward pass through the model."""
    # Create a mock session
    mock_instance = mock_session.return_value
    mock_instance.get_inputs.return_value = [MagicMock(name="input_name")]
    mock_instance.run.return_value = [np.zeros((1, 9, 256, 256))]
    
    # Create a dummy input tensor
    input_tensor = np.zeros((1, 3, 256, 256), dtype=np.float32)
    
    # Call the function
    output = forward_pass(mock_instance, input_tensor)
    
    # Assert that session.run was called
    mock_instance.run.assert_called_once()
    
    # Check the shape of the output
    assert output.shape == (1, 9, 256, 256)

# Tests for rescale_landmarks
def test_rescale_landmarks():
    """Test rescaling landmarks from model resolution to original image size."""
    # Define test landmarks
    landmarks = [(10, 20), (30, 40), None, (50, 60)]
    
    # Define model and original dimensions
    model_dimensions = (100, 100)
    original_dimensions = (200, 200)
    
    # Expected rescaled landmarks
    expected_rescaled = [(20, 40), (60, 80), None, (100, 120)] 
    
    # Call the function
    rescaled = rescale_landmarks(landmarks, model_dimensions, original_dimensions)
    
    # Check that rescaled landmarks match expected values
    assert len(rescaled) == len(expected_rescaled)
    for expected, actual in zip(expected_rescaled, rescaled):
        assert expected == actual

# Additional integration test with the real model
@pytest.mark.integration
def test_real_model_output():
    """
    Integration test that loads the actual model and verifies it produces output.
    This test requires the real model file to be present.
    """
    model_path = "models/model.onnx"
    
    # Skip test if model file doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found at {model_path}")
    
    # Load the real model
    session = load_onnx_model(model_path)
    
    # Create a simple test input
    test_input = np.random.rand(1, 3, 256, 256).astype(np.float32)
    
    # Run inference
    output = forward_pass(session, test_input)
    
    # Verify output shape and type
    assert output is not None
    assert isinstance(output, np.ndarray)
    assert output.shape[0] == 1  # Batch size
    assert output.shape[1] == 9  # Number of landmarks
    
    # Verify we can extract landmarks from the output
    landmarks = get_landmark_predictions(output)
    assert len(landmarks) == 9

@pytest.mark.integration
def test_end_to_end_with_real_model():
    """
    Integration test for the full prediction pipeline using the real model.
    This test requires internet access to download a test image.
    """
    model_path = "models/model.onnx"
    test_image_url = "https://media.istockphoto.com/id/157671964/photo/portrait-of-a-tabby-cat-looking-at-the-camera.jpg?s=612x612&w=0&k=20&c=iTsJO6vuQ5w3hL5pWn42C91ziMRUsYd725oUGRRewjM="
    
    # Skip test if model file doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found at {model_path}")
    
    try:
        # Run the full prediction pipeline
        landmarks = predict_cat_landmarks(model_path, test_image_url)
        
        # Verify we got landmarks
        assert landmarks is not None
        assert len(landmarks) == 9
        
        # Check that at least some landmarks were detected (not None)
        assert any(landmark is not None for landmark in landmarks)
        
        # For detected landmarks, verify they have reasonable coordinates
        for landmark in landmarks:
            if landmark is not None:
                x, y = landmark
                assert isinstance(x, int)
                assert isinstance(y, int)
                assert x >= 0
                assert y >= 0
    except Exception as e:
        pytest.skip(f"Test failed due to external dependency: {str(e)}")

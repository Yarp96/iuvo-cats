import os
import argparse
import torch
from pathlib import Path

from train import CatLandmarkDetector


def load_lightning_model(checkpoint_path, encoder_name="resnet34"):
    """
    Load a trained PyTorch Lightning model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        encoder_name (str): Name of the encoder backbone used during training
        
    Returns:
        CatLandmarkDetector: Loaded PyTorch Lightning model
    """
    # Load the model from checkpoint
    model = CatLandmarkDetector.load_from_checkpoint(
        checkpoint_path,
        encoder_name=encoder_name
    )
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def export_to_onnx(model, output_path, input_shape=(1, 3, 256, 256), dynamic_axes=True):
    """
    Export PyTorch Lightning model to ONNX format.
    
    Args:
        model (CatLandmarkDetector): PyTorch Lightning model to export
        output_path (str): Path to save the ONNX model
        input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width)
        dynamic_axes (bool): Whether to use dynamic axes for batch size and image dimensions
    """
    # Move model to CPU to avoid device mismatch issues
    model = model.to('cpu')
    
    # Create dummy input tensor on CPU
    dummy_input = torch.randn(input_shape, device='cpu')
    
    # Define dynamic axes if specified
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},  # Dynamic batch size and image dimensions
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}   # Dynamic batch size and image dimensions
        }
    
    # Export the model
    torch.onnx.export(
        model,                       # PyTorch model
        dummy_input,                 # Dummy input
        output_path,                 # Output file path
        export_params=True,          # Store the trained parameter weights inside the model file
        opset_version=13,            # ONNX opset version
        do_constant_folding=True,    # Optimize constant folding for better performance
        input_names=['input'],       # Input names
        output_names=['output'],     # Output names
        dynamic_axes=dynamic_axes_dict,  # Dynamic axes
        verbose=False
    )
    
    print(f"Model exported to ONNX format at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch Lightning model to ONNX format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output path for the ONNX model")
    parser.add_argument("--encoder", type=str, default="timm-efficientnet-b0", help="Encoder name used in the model")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size (square)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the input tensor")
    parser.add_argument("--no_dynamic_axes", action="store_true", help="Disable dynamic axes for fixed input size")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Load the model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_lightning_model(args.checkpoint, encoder_name=args.encoder)
    
    # Export to ONNX
    input_shape = (args.batch_size, 3, args.img_size, args.img_size)
    export_to_onnx(
        model, 
        str(output_path), 
        input_shape=input_shape, 
        dynamic_axes=not args.no_dynamic_axes
    )
    
    print("ONNX export completed successfully!")


if __name__ == "__main__":
    main()

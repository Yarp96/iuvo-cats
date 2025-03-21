import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_segmentation_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=9):
    """
    Create a Segmentation Model instance.
    
    Args:
        encoder_name (str): Name of the encoder backbone
        encoder_weights (str): Pre-trained weights for encoder
        in_channels (int): Number of input channels
        classes (int): Number of output segmentation classes
        
    Returns:
        smp.UnetPlusPlus: The initialized model
    """
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation="sigmoid"
    )


if __name__ == "__main__":
    import torch
    model = create_segmentation_model()
    print(model)
    print("Output shape:", model(torch.rand(1, 3, 256, 256)).shape)
    
import os
import argparse
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import get_cat_dataloader
from model import create_segmentation_model
from evaluation import evaluate_landmarks

class CatLandmarkDetector(pl.LightningModule):
    """
    PyTorch Lightning module for cat facial landmark detection.
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
    ):
        """
        Initialize the CatLandmarkDetector.
        
        Args:
            encoder_name (str): Name of the encoder backbone
            encoder_weights (str): Pre-trained weights for encoder
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            max_epochs (int): Maximum number of epochs for training (used for LR scheduler)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create the segmentation model
        self.model = create_segmentation_model(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=9  # 9 facial landmarks
        )
        
        # Define loss function
        self.criterion = nn.BCELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with loss
        """
        images = batch['image']
        target_heatmaps = batch['heatmaps']
        target_coords = batch['keypoints']  # [batch_size, num_keypoints, 2]
        
        # Forward pass
        with torch.amp.autocast(enabled=False, device_type='cuda'):
            pred_heatmaps = self(images)
            loss = self.criterion(pred_heatmaps, target_heatmaps)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with loss
        """
        images = batch['image']
        target_heatmaps = batch['heatmaps']
        target_coords = batch['keypoints']  # [batch_size, num_keypoints, 2]
        
        # Forward pass
        with torch.amp.autocast(enabled=False, device_type='cuda'):
            pred_heatmaps = self(images)
            loss = self.criterion(pred_heatmaps, target_heatmaps)
        
        # Calculate metrics
        eval_results = evaluate_landmarks(pred_heatmaps, target_coords,threshold=10.0)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
        # Log metrics
        self.log('val_mean_distance', eval_results['mean_distance'], on_epoch=True, logger=True)
        self.log('val_map_score', eval_results['map_score'], on_epoch=True, logger=True)
        
        return {'val_loss': loss}
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dict[str, Any]: Dictionary with optimizer and scheduler
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,  # Use the total number of training epochs
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


def train(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    gpus: int = 1,
    precision: int = 16,
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Train the cat facial landmark detection model.
    
    Args:
        data_dir (str): Directory containing the cat dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        encoder_name (str): Name of the encoder backbone
        encoder_weights (str): Pre-trained weights for encoder
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): Weight decay for optimizer
        max_epochs (int): Maximum number of epochs to train
        gpus (int): Number of GPUs to use
        precision (int): Precision for training (16 or 32)
        log_dir (str): Directory for TensorBoard logs
        checkpoint_dir (str): Directory for model checkpoints
    """
    # Create data loaders
    train_dataloader, val_dataloader = get_cat_dataloader(
        root_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create model
    model = CatLandmarkDetector(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs
    )
    
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create logger
    logger = TensorBoardLogger(log_dir, name="cat_landmarks")
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_map_score:.4f}",
        monitor="val_map_score",
        mode="max",
        save_top_k=2,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else None,
        precision=precision,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        deterministic=True,
    )
    
    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cat facial landmark detection model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the cat dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--encoder_name", type=str, default="timm-efficientnet-b0", help="Name of the encoder backbone")
    parser.add_argument("--encoder_weights", type=str, default="imagenet", help="Pre-trained weights for encoder")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum number of epochs to train")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--precision", type=int, default=16, help="Precision for training (16 or 32)")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for TensorBoard logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for model checkpoints")
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        precision=args.precision,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
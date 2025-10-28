"""
Training and validation logic for MIL model
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, Tuple

from config import DATA_PATHS, TRAINING_CONFIG, DEVICE


class MILTrainer:
    """
    Trainer class for MIL model
    """
    
    def __init__(self, model: nn.Module, device: str = None):
        self.model = model
        self.device = device if device else DEVICE
        self.model.to(self.device)
        
        # Initialize optimizer and criterion
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        Returns: average training loss
        """
        self.model.train()
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            case_data = batch[0]  # Get the first (and only) case in the batch
            
            stain_slices = case_data["stain_slices"]
            label = case_data["label"].to(self.device)
            
            # Forward pass - model outputs [num_classes] logits
            outputs = self.model(stain_slices)
            
            # Add batch dimension: [num_classes] -> [1, num_classes]
            outputs = outputs.unsqueeze(0)
            
            # Ensure label has batch dimension: scalar -> [1]
            if label.dim() == 0:
                label = label.unsqueeze(0)
            
            # Calculate loss
            loss = self.criterion(outputs, label)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model
        Returns: (average_loss, accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        correct_total = 0
        sample_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                case_data = batch[0]
                stain_slices = case_data["stain_slices"]
                label = case_data["label"].to(self.device)
                
                outputs = self.model(stain_slices)
                
                # Add batch dimension for loss calculation
                outputs = outputs.unsqueeze(0)  # [1, num_classes]
                if label.dim() == 0:
                    label = label.unsqueeze(0)  # [1]
                
                loss = self.criterion(outputs, label)
                val_loss += loss.item()
                
                # Calculate accuracy
                pred = torch.argmax(outputs, dim=1)  # [1]
                correct_total += (pred == label).sum().item()
                sample_total += 1
        
        avg_loss = val_loss / max(sample_total, 1)
        accuracy = correct_total / max(sample_total, 1)
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, arch: str = "HierarchicalAttnMIL", 
                       checkpoint_dir: str = None):
        """
        Save model checkpoint
        """
        if checkpoint_dir is None:
            checkpoint_dir = DATA_PATHS['checkpoint_dir']
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(checkpoint_dir, f"{timestamp}_{arch}_epoch{epoch}.pth")
        
        checkpoint = {
            "arch": arch,
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }
        
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint
        Returns: epoch number
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load training history if available
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        if "val_losses" in checkpoint:
            self.val_losses = checkpoint["val_losses"]
        if "val_accuracies" in checkpoint:
            self.val_accuracies = checkpoint["val_accuracies"]
        
        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from epoch {epoch}")
        return epoch
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = None, start_epoch: int = 0, save_every: int = 1):
        """
        Full training loop
        """
        if epochs is None:
            epochs = TRAINING_CONFIG['epochs']
        
        print(f"Starting training from epoch {start_epoch + 1} to {epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        print("\nTraining completed!")
    
    def evaluate(self, test_loader: DataLoader, save_predictions: bool = True, 
                 output_dir: str = None, checkpoint_name: str = None) -> Dict[str, Any]:
        """
        Evaluate model on test set
        Returns: evaluation metrics
        """
        self.model.eval()
        test_loss = 0.0
        correct_total = 0
        sample_total = 0
        predictions = []
        true_labels = []
        case_ids = []
        prediction_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                case_data = batch[0]
                case_id = case_data["case_id"]
                stain_slices = case_data["stain_slices"]
                label = case_data["label"].to(self.device)
                
                outputs = self.model(stain_slices)
                
                # Add batch dimension for loss calculation
                outputs = outputs.unsqueeze(0)
                if label.dim() == 0:
                    label = label.unsqueeze(0)
                
                loss = self.criterion(outputs, label)
                test_loss += loss.item()
                
                # Get prediction probabilities
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1)
                
                correct_total += (pred == label).sum().item()
                sample_total += 1
                
                # Store results
                case_ids.append(case_id)
                predictions.append(pred.cpu().item())
                true_labels.append(label.cpu().item())
                prediction_probs.append(probs.cpu().numpy()[0])  # [prob_class0, prob_class1]
        
        avg_loss = test_loss / max(sample_total, 1)
        accuracy = correct_total / max(sample_total, 1)
        
        results = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'case_ids': case_ids,
            'prediction_probs': prediction_probs,
            'num_samples': sample_total
        }
        
        # Save predictions to CSV if requested
        if save_predictions:
            self._save_predictions_csv(results, output_dir, checkpoint_name)
        
        print(f"\nTest Results:")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Samples: {sample_total}")
        
        return results
    
    def _save_predictions_csv(self, results: Dict[str, Any], output_dir: str = None, 
                             checkpoint_name: str = None):
        """Save predictions to CSV file"""
        import pandas as pd
        
        if output_dir is None:
            output_dir = DATA_PATHS['checkpoint_dir']
        
        # Create filename based on checkpoint name
        if checkpoint_name:
            # Extract base name without extension
            base_name = os.path.splitext(os.path.basename(checkpoint_name))[0]
            csv_filename = f"{base_name}_predictions.csv"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"predictions_{timestamp}.csv"
        
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Create DataFrame with results
        df_data = {
            'case_id': results['case_ids'],
            'true_label': results['true_labels'],
            'predicted_label': results['predictions'],
            'prob_benign': [probs[0] for probs in results['prediction_probs']],
            'prob_high_grade': [probs[1] for probs in results['prediction_probs']],
            'correct': [t == p for t, p in zip(results['true_labels'], results['predictions'])]
        }
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        
        print(f"Predictions saved to: {csv_path}")
        return csv_path


def count_patches_by_class(case_dict: Dict, label_map: Dict, split_name: str):
    """
    Count patches by class for analysis
    """
    from collections import defaultdict
    
    class_patch_counts = defaultdict(int)
    
    for case_id, stains in case_dict.items():
        if case_id not in label_map:
            continue
        
        label = label_map[case_id]
        total_patches = 0
        
        for stain_data in stains.values():
            for slice_patches in stain_data:
                total_patches += len(slice_patches)
        
        class_patch_counts[label] += total_patches
    
    print(f"\nPatch count by class for {split_name}:")
    print(f"  Benign (0):     {class_patch_counts[0]} patches")
    print(f"  High-grade (1): {class_patch_counts[1]} patches")
    
    return class_patch_counts
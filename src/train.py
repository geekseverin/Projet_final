# src/train.py
import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import random
import gc  # Pour la gestion mémoire

# Imports depuis nos modules
from autoencoder import create_autoencoder, train_autoencoder
from convlstm import create_tumor_predictor


class TumorDataset(Dataset):
    """Dataset pour l'entraînement du modèle de prédiction"""
    def __init__(self, processed_dir, timepoints=('Timepoint_1', 'Timepoint_2'), target_timepoint='Timepoint_5'):
        self.processed_dir = processed_dir
        self.timepoints = timepoints
        self.target_timepoint = target_timepoint
        self.samples = []
        
        # Collecter tous les échantillons valides
        patients = [d for d in os.listdir(processed_dir) if d.startswith("PatientID_")]
        
        for patient in patients:
            patient_dir = os.path.join(processed_dir, patient)
            
            # Vérifier que tous les timepoints existent
            has_all_timepoints = True
            sample_data = {'patient': patient}
            
            # Timepoints d'entrée
            for tp in timepoints:
                tp_dir = os.path.join(patient_dir, tp)
                brain_file = f"{patient}_{tp}_brain_t1c.nii.gz"
                mask_file = f"{patient}_{tp}_tumorMask.nii.gz"
                
                brain_path = os.path.join(tp_dir, brain_file)
                mask_path = os.path.join(tp_dir, mask_file)
                
                if not (os.path.exists(brain_path) and os.path.exists(mask_path)):
                    has_all_timepoints = False
                    break
                    
                sample_data[f'{tp}_brain'] = brain_path
                sample_data[f'{tp}_mask'] = mask_path
            
            # Timepoint cible
            target_dir = os.path.join(patient_dir, target_timepoint)
            target_mask_file = f"{patient}_{target_timepoint}_tumorMask.nii.gz"
            target_mask_path = os.path.join(target_dir, target_mask_file)
            
            if has_all_timepoints and os.path.exists(target_mask_path):
                sample_data[f'{target_timepoint}_mask'] = target_mask_path
                self.samples.append(sample_data)
        
        print(f"Dataset créé avec {len(self.samples)} échantillons valides")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Charger les images d'entrée (T1, T2)
        input_images = []
        input_masks = []
        
        for tp in self.timepoints:
            brain = nib.load(sample[f'{tp}_brain']).get_fdata().astype(np.float32)
            mask = nib.load(sample[f'{tp}_mask']).get_fdata().astype(np.float32)
            
            # S'assurer que les masques sont binaires [0, 1]
            mask = np.clip(mask, 0, 1)
            
            # Ajouter dimension channel
            brain = brain[np.newaxis, ...]  # (1, D, H, W)
            mask = mask[np.newaxis, ...]    # (1, D, H, W)
            
            input_images.append(torch.from_numpy(brain))
            input_masks.append(torch.from_numpy(mask))
        
        # Stack temporel
        input_images = torch.stack(input_images, dim=0)  # (T, 1, D, H, W)
        input_masks = torch.stack(input_masks, dim=0)    # (T, 1, D, H, W)
        
        # Charger le masque cible (T3)
        target_mask = nib.load(sample[f'{self.target_timepoint}_mask']).get_fdata().astype(np.float32)
        # S'assurer que le masque cible est binaire [0, 1]
        target_mask = np.clip(target_mask, 0, 1)
        target_mask = torch.from_numpy(target_mask[np.newaxis, ...])  # (1, D, H, W)
        
        return {
            'input_images': input_images,
            'input_masks': input_masks,
            'target_mask': target_mask,
            'patient_id': sample['patient']
        }


def create_data_loaders(data_path, batch_size=1, train_split=0.8):  # Réduire batch_size à 1
    """Crée les dataloaders d'entraînement et de validation"""
    dataset = TumorDataset(data_path)
    
    # Split train/val
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)  # Désactiver pin_memory
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=False)  # Désactiver pin_memory
    
    return train_loader, val_loader


def plot_training_curves(train_losses, val_losses, title, save_path):
    """Trace et sauvegarde les courbes d'entraînement"""
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train')
    plt.plot(range(1, len(val_losses)+1), val_losses, marker='o', label='Val')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def dice_coefficient(pred, target, smooth=1e-6):
    """Calcule le coefficient de Dice"""
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def dice_loss(pred, target, smooth=1e-6):
    """Loss basée sur le coefficient de Dice"""
    return 1 - dice_coefficient(pred, target, smooth)


def debug_tensor_values(tensor, name):
    """Debug helper pour afficher les stats d'un tensor"""
    if torch.isfinite(tensor).all():
        print(f"{name}: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")
    else:
        print(f"{name}: CONTIENT DES VALEURS NON-FINIES!")
        print(f"  NaN: {torch.isnan(tensor).sum()}")
        print(f"  Inf: {torch.isinf(tensor).sum()}")


def safe_forward_pass(model, inputs, masks_in, max_retries=3):
    """Passe forward avec gestion d'erreurs et retry"""
    for attempt in range(max_retries):
        try:
            # Vérifier les entrées
            if not torch.isfinite(inputs).all():
                print(f"Attention: inputs contient des valeurs non-finies à l'attempt {attempt+1}")
                inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
                
            if not torch.isfinite(masks_in).all():
                print(f"Attention: masks_in contient des valeurs non-finies à l'attempt {attempt+1}")
                masks_in = torch.nan_to_num(masks_in, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Nettoyer le cache CUDA si nécessaire
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Passage forward
            pred_logits = model(inputs, masks_in)
            
            # Vérifier la sortie
            if not torch.isfinite(pred_logits).all():
                print(f"Attention: pred_logits contient des valeurs non-finies")
                pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return pred_logits
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Erreur mémoire à l'attempt {attempt+1}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if attempt == max_retries - 1:
                    raise e
                continue
            else:
                print(f"Erreur RuntimeError: {e}")
                raise e
        except Exception as e:
            print(f"Erreur inattendue à l'attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise e
            continue
    
    raise RuntimeError("Échec après plusieurs tentatives")


def train_complete_pipeline(data_path, output_dir, config):
    """
    Entraîne le pipeline complet : AutoEncoder puis ConvLSTM predictor
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilisé: {device}")

    # Configuration mémoire CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Mémoire GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()

    # Créer les dossiers de sortie
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Chargement des données
    print("Chargement des données...")
    train_loader, val_loader = create_data_loaders(
        data_path, 
        batch_size=config['batch_size'], 
        train_split=config['train_split']
    )
    print(f"Données chargées: {len(train_loader)} batches train, {len(val_loader)} batches val")

    # 1) AutoEncoder (optionnel)
    print("\n=== AutoEncoder ===")
    autoencoder = create_autoencoder(latent_dim=config['latent_dim']).to(device)
    
    if config.get('train_autoencoder', True):
        print("Entraînement de l'AutoEncoder...")
        ae_train_losses, ae_val_losses = train_autoencoder(
            autoencoder, train_loader, val_loader,
            num_epochs=config['ae_epochs'], 
            lr=config['ae_lr'],
            output_dir=output_dir
        )
        
        # Sauvegarder l'autoencoder
        torch.save(autoencoder.state_dict(), os.path.join(output_dir, "models", "autoencoder.pth"))
        print("AutoEncoder sauvegardé")
        
    else:
        # Essayer de charger un AE pré-entraîné
        ae_path = os.path.join(output_dir, "models", "autoencoder.pth")
        if os.path.exists(ae_path):
            autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
            print("AutoEncoder chargé depuis:", ae_path)
        else:
            print("Attention: autoencoder non entraîné et aucun checkpoint trouvé.")

    # 2) ConvLSTM predictor
    print("\n=== Modèle de Prédiction (ConvLSTM) ===")
    predictor = create_tumor_predictor(
        autoencoder,
        latent_dim=config['latent_dim'],
        use_mask=config.get('use_mask', True),
        mask_channels=1,
        hidden_dims=config.get('pred_hidden_dims', [64, 32]),
        kernel_sizes=config.get('pred_kernel_sizes', [(3,3,3),(3,3,3)]),
        num_layers=len(config.get('pred_hidden_dims', [64,32])),
        grid_size=(4,4,4)
    ).to(device)

    # Freeze l'autoencoder si souhaité
    if config.get('freeze_autoencoder', False):
        for param in autoencoder.parameters():
            param.requires_grad = False
        print("AutoEncoder gelé")

    # Configuration de l'entraînement
    criterion_bce = nn.BCEWithLogitsLoss(reduction='mean')  # Spécifier reduction explicitement
    optimizer = torch.optim.Adam(
        predictor.parameters(), 
        lr=config['pred_lr'], 
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )

    # Variables pour l'entraînement
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 10)

    print(f"Début de l'entraînement du prédicteur ({config['pred_epochs']} epochs)...")
    
    for epoch in range(config['pred_epochs']):
        # Phase d'entraînement
        predictor.train()
        sum_loss, sum_dice = 0.0, 0.0
        nb_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['pred_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Format des données: 
                # input_images: (B, T, 1, D, H, W)
                # input_masks: (B, T, 1, D, H, W)  
                # target_mask: (B, 1, D, H, W)
                
                inputs = batch['input_images'].to(device, non_blocking=False)      # (B, T, 1, D, H, W)
                masks_in = batch['input_masks'].to(device, non_blocking=False)     # (B, T, 1, D, H, W)
                target = batch['target_mask'].to(device, non_blocking=False)       # (B, 1, D, H, W)

                # Debug: vérifier les valeurs des tenseurs (seulement pour le premier batch)
                if epoch == 0 and batch_idx == 0:
                    debug_tensor_values(inputs, "inputs")
                    debug_tensor_values(masks_in, "masks_in")
                    debug_tensor_values(target, "target")
                    print(f"Shapes - inputs: {inputs.shape}, masks_in: {masks_in.shape}, target: {target.shape}")

                optimizer.zero_grad(set_to_none=True)
                
                # Prédiction (logits) avec gestion d'erreurs
                pred_logits = safe_forward_pass(predictor, inputs, masks_in)  # (B, 1, D, H, W)
                
                # Vérifier les dimensions
                if pred_logits.shape != target.shape:
                    print(f"Erreur de dimension: pred_logits {pred_logits.shape} vs target {target.shape}")
                    continue
                
                # Calcul des losses
                bce = criterion_bce(pred_logits, target)
                
                # Pour le dice, appliquer sigmoid aux logits
                pred_probs = torch.sigmoid(pred_logits)
                dice = dice_loss(pred_probs, target)
                
                # Vérifier que les losses sont finies
                if not (torch.isfinite(bce) and torch.isfinite(dice)):
                    print(f"Loss non-finie détectée: bce={bce}, dice={dice}")
                    continue
                
                total_loss = config['loss_alpha'] * bce + (1 - config['loss_alpha']) * dice
                
                if not torch.isfinite(total_loss):
                    print(f"Total loss non-finie: {total_loss}")
                    continue
                
                # Backpropagation
                total_loss.backward()
                
                # Gradient clipping (optionnel)
                if config.get('grad_clip', None):
                    grad_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), config['grad_clip'])
                    if not torch.isfinite(grad_norm):
                        print(f"Gradient norm non-fini: {grad_norm}")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                
                optimizer.step()

                # Métriques
                with torch.no_grad():
                    dice_coef = dice_coefficient(pred_probs, target).item()
                    if torch.isfinite(torch.tensor(dice_coef)):
                        sum_loss += total_loss.item()
                        sum_dice += dice_coef
                        nb_batches += 1
                    else:
                        print(f"Dice coefficient non-fini: {dice_coef}")
                
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'bce': f"{bce.item():.4f}", 
                    'dice_loss': f"{dice.item():.4f}",
                    'dice_coef': f"{dice_coef:.4f}"
                })

                # Libérer la mémoire GPU périodiquement
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Erreur dans le batch {batch_idx}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        avg_train_loss = sum_loss / max(1, nb_batches)
        avg_train_dice = sum_dice / max(1, nb_batches)
        train_losses.append(avg_train_loss)

        # Phase de validation
        predictor.eval()
        sum_val_loss, sum_val_dice = 0.0, 0.0
        nb_val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    inputs = batch['input_images'].to(device, non_blocking=False)
                    masks_in = batch['input_masks'].to(device, non_blocking=False)
                    target = batch['target_mask'].to(device, non_blocking=False)
                    
                    pred_logits = safe_forward_pass(predictor, inputs, masks_in)
                    pred_probs = torch.sigmoid(pred_logits)
                    
                    bce = criterion_bce(pred_logits, target)
                    dice = dice_loss(pred_probs, target)
                    total_loss = config['loss_alpha'] * bce + (1 - config['loss_alpha']) * dice
                    
                    if torch.isfinite(total_loss):
                        dice_coef = dice_coefficient(pred_probs, target).item()
                        if torch.isfinite(torch.tensor(dice_coef)):
                            sum_val_loss += total_loss.item()
                            sum_val_dice += dice_coef
                            nb_val_batches += 1
                    
                except Exception as e:
                    print(f"Erreur validation batch {batch_idx}: {e}")
                    continue

        avg_val_loss = sum_val_loss / max(1, nb_val_batches)
        avg_val_dice = sum_val_dice / max(1, nb_val_batches)
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch+1}] "
              f"Train Loss: {avg_train_loss:.6f} (Dice: {avg_train_dice:.4f}) | "
              f"Val Loss: {avg_val_loss:.6f} (Dice: {avg_val_dice:.4f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Scheduler
        if np.isfinite(avg_val_loss):
            scheduler.step(avg_val_loss)

        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss and np.isfinite(avg_val_loss):
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_dice': avg_val_dice,
                'config': config
            }, os.path.join(output_dir, "models", "best_predictor.pth"))
            print("✅ Meilleur modèle sauvegardé")
        else:
            patience_counter += 1

        # Sauvegarder les courbes
        if len(train_losses) > 0 and len(val_losses) > 0:
            plot_training_curves(
                train_losses, val_losses, 
                "Prediction Training Loss",
                os.path.join(output_dir, "logs", "prediction_loss_curve.png")
            )
            
            # Sauvegarder l'historique CSV
            import csv
            with open(os.path.join(output_dir, "logs", "prediction_loss_history.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
                for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
                    writer.writerow([i, tr, va])

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping activé (patience={patience})")
            break

        # Nettoyage mémoire
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Sauvegarde finale
    torch.save({
        'model_state_dict': predictor.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config
    }, os.path.join(output_dir, "models", "final_predictor.pth"))
    
    print(f"\n✅ Entraînement terminé!")
    print(f"Meilleure validation loss: {best_val_loss:.6f}")
    
    return predictor, train_losses, val_losses


def get_default_config():
    """Configuration par défaut pour l'entraînement"""
    return {
        # Dataset
        'batch_size': 1,  # Réduire le batch size
        'train_split': 0.8,
        
        # Entraînement général
        'patience': 15,
        'grad_clip': 1.0,
        
        # AutoEncoder
        'train_autoencoder': True,
        'freeze_autoencoder': False,
        'latent_dim': 128,
        'ae_epochs': 20,
        'ae_lr': 1e-3,
        
        # Prédicteur ConvLSTM
        'pred_epochs': 5,
        'pred_lr': 1e-4,
        'loss_alpha': 0.7,  # balance BCE vs Dice
        'use_mask': True,
        'pred_hidden_dims': [64, 32],
        'pred_kernel_sizes': [(3,3,3), (3,3,3)]
    }


if __name__ == "__main__":
    # Paramètres
    data_path = "data/processed"
    output_dir = "outputs"
    
    # Configuration
    config = get_default_config()
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Lancer l'entraînement
    try:
        predictor, train_losses, val_losses = train_complete_pipeline(
            data_path, output_dir, config
        )
        print("\n🎉 Pipeline d'entraînement terminé avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur durant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
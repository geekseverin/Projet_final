# src/evaluate.py
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import pandas as pd

from autoencoder import create_autoencoder
from convlstm import create_tumor_predictor
from train import TumorDataset, dice_coefficient


class ModelEvaluator:
    """Classe pour évaluer les modèles entraînés"""
    
    def __init__(self, model_dir, config):
        self.model_dir = model_dir
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger les modèles
        self.autoencoder = self._load_autoencoder()
        self.predictor = self._load_predictor()
        
    def _load_autoencoder(self):
        """Charge l'autoencoder pré-entraîné"""
        autoencoder = create_autoencoder(latent_dim=self.config['latent_dim'])
        ae_path = os.path.join(self.model_dir, "autoencoder.pth")
        
        if os.path.exists(ae_path):
            autoencoder.load_state_dict(torch.load(ae_path, map_location=self.device))
            autoencoder.to(self.device)
            autoencoder.eval()
            print(f"✅ AutoEncoder chargé depuis {ae_path}")
        else:
            raise FileNotFoundError(f"AutoEncoder non trouvé: {ae_path}")
            
        return autoencoder
    
    def _load_predictor(self):
        """Charge le modèle de prédiction"""
        predictor = create_tumor_predictor(
            self.autoencoder,
            latent_dim=self.config['latent_dim'],
            use_mask=self.config.get('use_mask', True),
            mask_channels=1,
            hidden_dims=self.config.get('pred_hidden_dims', [64, 32]),
            kernel_sizes=self.config.get('pred_kernel_sizes', [(3,3,3),(3,3,3)]),
            num_layers=len(self.config.get('pred_hidden_dims', [64,32])),
            grid_size=(4,4,4)
        )
        
        # Essayer de charger le meilleur modèle d'abord
        best_path = os.path.join(self.model_dir, "best_predictor.pth")
        final_path = os.path.join(self.model_dir, "final_predictor.pth")
        
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            predictor.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Meilleur prédicteur chargé depuis {best_path}")
            print(f"   Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
            print(f"   Validation Dice: {checkpoint.get('val_dice', 'N/A')}")
        elif os.path.exists(final_path):
            checkpoint = torch.load(final_path, map_location=self.device)
            predictor.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Prédicteur final chargé depuis {final_path}")
        else:
            raise FileNotFoundError(f"Aucun modèle de prédiction trouvé dans {self.model_dir}")
        
        predictor.to(self.device)
        predictor.eval()
        return predictor


def compute_metrics(pred_mask, true_mask, threshold=0.5):
    """Calcule diverses métriques de segmentation"""
    pred_binary = (pred_mask > threshold).astype(np.float32)
    true_binary = true_mask.astype(np.float32)
    
    # Dice coefficient
    dice = dice_coefficient(
        torch.from_numpy(pred_binary[None, None, ...]),
        torch.from_numpy(true_binary[None, None, ...])
    ).item()
    
    # IoU (Jaccard)
    intersection = np.sum(pred_binary * true_binary)
    union = np.sum(pred_binary) + np.sum(true_binary) - intersection
    iou = intersection / (union + 1e-8)
    
    # Sensibilité et Spécificité
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    
    tp = np.sum((pred_flat == 1) & (true_flat == 1))
    tn = np.sum((pred_flat == 0) & (true_flat == 0))
    fp = np.sum((pred_flat == 1) & (true_flat == 0))
    fn = np.sum((pred_flat == 0) & (true_flat == 1))
    
    sensitivity = tp / (tp + fn + 1e-8)  # Recall
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    
    # Hausdorff distance (simplifiée)
    try:
        from scipy.spatial.distance import directed_hausdorff
        if np.sum(pred_binary) > 0 and np.sum(true_binary) > 0:
            pred_coords = np.argwhere(pred_binary > 0)
            true_coords = np.argwhere(true_binary > 0)
            hausdorff_dist = max(
                directed_hausdorff(pred_coords, true_coords)[0],
                directed_hausdorff(true_coords, pred_coords)[0]
            )
        else:
            hausdorff_dist = float('inf')
    except:
        hausdorff_dist = -1  # Non calculé
    
    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'hausdorff': hausdorff_dist
    }


def evaluate_model(model_dir, data_path, config, output_dir="outputs/evaluation"):
    """Évalue le modèle sur le dataset de test"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialiser l'évaluateur
    evaluator = ModelEvaluator(model_dir, config)
    
    # Créer le dataset de test (on utilise tous les patients)
    test_dataset = TumorDataset(
        data_path, 
        timepoints=('Timepoint_1', 'Timepoint_2'), 
        target_timepoint='Timepoint_5'
    )
    
    print(f"Évaluation sur {len(test_dataset)} échantillons...")
    
    results = []
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Évaluation"):
            sample = test_dataset[i]
            patient_id = sample['patient_id']
            
            # Préparer les données
            input_images = sample['input_images'].unsqueeze(0).to(evaluator.device)  # (1, T, 1, D, H, W)
            input_masks = sample['input_masks'].unsqueeze(0).to(evaluator.device)    # (1, T, 1, D, H, W)
            true_mask = sample['target_mask'].squeeze().cpu().numpy()                # (D, H, W)
            
            # Prédiction
            pred_logits = evaluator.predictor(input_images, input_masks)  # (1, 1, D, H, W)
            pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()  # (D, H, W)
            
            # Calculer les métriques
            metrics = compute_metrics(pred_prob, true_mask)
            metrics['patient_id'] = patient_id
            results.append(metrics)
            
            # Sauvegarder la prédiction
            pred_nii = nib.Nifti1Image(pred_prob.astype(np.float32), np.eye(4))
            pred_path = os.path.join(predictions_dir, f"{patient_id}_Timepoint_5_predicted.nii.gz")
            nib.save(pred_nii, pred_path)
            
            # Sauvegarder aussi la version binaire
            pred_binary = (pred_prob > 0.5).astype(np.float32)
            pred_bin_nii = nib.Nifti1Image(pred_binary, np.eye(4))
            pred_bin_path = os.path.join(predictions_dir, f"{patient_id}_Timepoint_5_predicted_binary.nii.gz")
            nib.save(pred_bin_nii, pred_bin_path)
    
    # Créer un DataFrame avec les résultats
    df_results = pd.DataFrame(results)
    
    # Statistiques globales
    print("\n=== RÉSULTATS D'ÉVALUATION ===")
    print(f"Nombre de patients évalués: {len(df_results)}")
    print("\nMétriques moyennes:")
    for metric in ['dice', 'iou', 'sensitivity', 'specificity', 'precision']:
        mean_val = df_results[metric].mean()
        std_val = df_results[metric].std()
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Sauvegarder les résultats
    results_path = os.path.join(output_dir, "evaluation_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\n✅ Résultats sauvegardés dans {results_path}")
    
    # Créer des visualisations
    create_evaluation_plots(df_results, output_dir)
    
    return df_results


def create_evaluation_plots(df_results, output_dir):
    """Crée des graphiques d'évaluation"""
    
    # 1. Distribution des métriques
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['dice', 'iou', 'sensitivity', 'specificity', 'precision']
    for i, metric in enumerate(metrics):
        axes[i].hist(df_results[metric], bins=20, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution {metric.upper()}')
        axes[i].set_xlabel(metric.upper())
        axes[i].set_ylabel('Fréquence')
        axes[i].axvline(df_results[metric].mean(), color='red', linestyle='--', 
                       label=f'Moyenne: {df_results[metric].mean():.3f}')
        axes[i].legend()
    
    # Retirer le subplot vide
    axes[5].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot des métriques
    plt.figure(figsize=(12, 6))
    metrics_data = [df_results[metric].values for metric in metrics]
    plt.boxplot(metrics_data, labels=[m.upper() for m in metrics])
    plt.title('Distribution des Métriques d\'Évaluation')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "metrics_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Matrice de corrélation
    plt.figure(figsize=(8, 6))
    corr_matrix = df_results[metrics].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('Corrélation entre les Métriques')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_correlation.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Performance par patient
    plt.figure(figsize=(15, 6))
    x_pos = range(len(df_results))
    plt.bar(x_pos, df_results['dice'], alpha=0.7, label='Dice')
    plt.bar(x_pos, df_results['iou'], alpha=0.7, label='IoU')
    plt.xlabel('Patient')
    plt.ylabel('Score')
    plt.title('Performance par Patient')
    plt.legend()
    plt.xticks(x_pos, df_results['patient_id'], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_per_patient.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphiques d'évaluation sauvegardés dans {output_dir}")


def visualize_predictions(data_path, model_dir, config, patient_ids=None, output_dir="outputs/visualization"):
    """Visualise les prédictions pour quelques patients"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les modèles
    evaluator = ModelEvaluator(model_dir, config)
    
    # Dataset
    dataset = TumorDataset(data_path)
    
    # Sélectionner des patients à visualiser
    if patient_ids is None:
        # Prendre les 3 premiers patients
        patient_ids = [dataset.samples[i]['patient'] for i in range(min(3, len(dataset)))]
    
    for patient_id in patient_ids:
        # Trouver l'échantillon correspondant
        sample_idx = None
        for i, sample in enumerate(dataset.samples):
            if sample['patient'] == patient_id:
                sample_idx = i
                break
        
        if sample_idx is None:
            print(f"Patient {patient_id} non trouvé")
            continue
        
        sample = dataset[sample_idx]
        
        # Préparer les données
        input_images = sample['input_images'].unsqueeze(0).to(evaluator.device)
        input_masks = sample['input_masks'].unsqueeze(0).to(evaluator.device)
        true_mask = sample['target_mask'].squeeze().cpu().numpy()
        
        # Prédiction
        with torch.no_grad():
            pred_logits = evaluator.predictor(input_images, input_masks)
            pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
            pred_binary = (pred_prob > 0.5).astype(np.float32)
        
        # Charger les images originales
        t1_brain = nib.load(dataset.samples[sample_idx]['Timepoint_1_brain']).get_fdata()
        t2_brain = nib.load(dataset.samples[sample_idx]['Timepoint_2_brain']).get_fdata()
        t1_mask = nib.load(dataset.samples[sample_idx]['Timepoint_1_mask']).get_fdata()
        t2_mask = nib.load(dataset.samples[sample_idx]['Timepoint_2_mask']).get_fdata()
        
        # Créer la visualisation
        create_patient_visualization(
            patient_id, t1_brain, t2_brain, t1_mask, t2_mask, 
            true_mask, pred_prob, pred_binary, output_dir
        )


def create_patient_visualization(patient_id, t1_brain, t2_brain, t1_mask, t2_mask, 
                               true_mask, pred_prob, pred_binary, output_dir):
    """Crée une visualisation complète pour un patient"""
    
    # Sélectionner quelques slices représentatives
    depth = true_mask.shape[0]
    slices = [depth//4, depth//2, 3*depth//4]
    
    fig, axes = plt.subplots(len(slices), 7, figsize=(20, 4*len(slices)))
    if len(slices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, slice_idx in enumerate(slices):
        # T1 Brain
        axes[i, 0].imshow(t1_brain[slice_idx], cmap='gray')
        axes[i, 0].set_title(f'T1 Brain\nSlice {slice_idx}')
        axes[i, 0].axis('off')
        
        # T1 Mask
        axes[i, 1].imshow(t1_mask[slice_idx], cmap='Reds', alpha=0.7)
        axes[i, 1].imshow(t1_brain[slice_idx], cmap='gray', alpha=0.3)
        axes[i, 1].set_title('T1 Mask')
        axes[i, 1].axis('off')
        
        # T2 Brain
        axes[i, 2].imshow(t2_brain[slice_idx], cmap='gray')
        axes[i, 2].set_title('T2 Brain')
        axes[i, 2].axis('off')
        
        # T2 Mask
        axes[i, 3].imshow(t2_mask[slice_idx], cmap='Reds', alpha=0.7)
        axes[i, 3].imshow(t2_brain[slice_idx], cmap='gray', alpha=0.3)
        axes[i, 3].set_title('T2 Mask')
        axes[i, 3].axis('off')
        
        # True T3 Mask
        axes[i, 4].imshow(true_mask[slice_idx], cmap='Reds', alpha=0.7)
        axes[i, 4].set_title('True T3 Mask')
        axes[i, 4].axis('off')
        
        # Predicted T3 (probability)
        axes[i, 5].imshow(pred_prob[slice_idx], cmap='Blues', vmin=0, vmax=1)
        axes[i, 5].set_title('Predicted T3\n(Probability)')
        axes[i, 5].axis('off')
        
        # Predicted T3 (binary)
        axes[i, 6].imshow(pred_binary[slice_idx], cmap='Blues', alpha=0.7)
        axes[i, 6].set_title('Predicted T3\n(Binary)')
        axes[i, 6].axis('off')
    
    # Calculer les métriques pour ce patient
    metrics = compute_metrics(pred_prob, true_mask)
    
    plt.suptitle(f'Patient: {patient_id} | '
                f'Dice: {metrics["dice"]:.3f} | '
                f'IoU: {metrics["iou"]:.3f} | '
                f'Sensitivity: {metrics["sensitivity"]:.3f}', 
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{patient_id}_visualization.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualisation sauvegardée pour {patient_id}")


def compare_with_diffusion_masks(data_path, predictions_dir, diffusion_masks_dir, output_dir="outputs/comparison"):
    """Compare les prédictions avec les masques générés par diffusion"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    patients = [d for d in os.listdir(data_path) if d.startswith("PatientID_")]
    
    for patient in patients:
        # Charger le vrai masque T3
        true_mask_path = os.path.join(data_path, patient, "Timepoint_5", 
                                     f"{patient}_Timepoint_5_tumorMask.nii.gz")
        if not os.path.exists(true_mask_path):
            continue
            
        true_mask = nib.load(true_mask_path).get_fdata()
        
        # Charger la prédiction ConvLSTM
        pred_lstm_path = os.path.join(predictions_dir, f"{patient}_Timepoint_5_predicted_binary.nii.gz")
        if not os.path.exists(pred_lstm_path):
            continue
        pred_lstm = nib.load(pred_lstm_path).get_fdata()
        
        # Charger le masque de diffusion
        diff_mask_path = os.path.join(diffusion_masks_dir, patient, 
                                     f"{patient}_Timepoint_5_generated_mask_bin.nii.gz")
        if not os.path.exists(diff_mask_path):
            continue
        diff_mask = nib.load(diff_mask_path).get_fdata()
        
        # Calculer les métriques
        metrics_lstm = compute_metrics(pred_lstm, true_mask)
        metrics_diff = compute_metrics(diff_mask, true_mask)
        
        results.append({
            'patient_id': patient,
            'lstm_dice': metrics_lstm['dice'],
            'diff_dice': metrics_diff['dice'],
            'lstm_iou': metrics_lstm['iou'],
            'diff_iou': metrics_diff['iou'],
            'lstm_sensitivity': metrics_lstm['sensitivity'],
            'diff_sensitivity': metrics_diff['sensitivity']
        })
    
    df_comparison = pd.DataFrame(results)
    
    # Statistiques comparatives
    print("\n=== COMPARAISON CONVLSTM vs DIFFUSION ===")
    print(f"ConvLSTM - Dice moyen: {df_comparison['lstm_dice'].mean():.4f} ± {df_comparison['lstm_dice'].std():.4f}")
    print(f"Diffusion - Dice moyen: {df_comparison['diff_dice'].mean():.4f} ± {df_comparison['diff_dice'].std():.4f}")
    
    # Test statistique
    from scipy.stats import wilcoxon
    stat, p_value = wilcoxon(df_comparison['lstm_dice'], df_comparison['diff_dice'])
    print(f"Test de Wilcoxon (Dice): p-value = {p_value:.4f}")
    
    # Sauvegarder
    df_comparison.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Créer un graphique de comparaison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df_comparison['diff_dice'], df_comparison['lstm_dice'], alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    plt.xlabel('Diffusion Model Dice')
    plt.ylabel('ConvLSTM Dice')
    plt.title('Comparaison Dice Score')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    x_pos = range(len(df_comparison))
    width = 0.35
    plt.bar([x - width/2 for x in x_pos], df_comparison['lstm_dice'], 
           width, label='ConvLSTM', alpha=0.7)
    plt.bar([x + width/2 for x in x_pos], df_comparison['diff_dice'], 
           width, label='Diffusion', alpha=0.7)
    plt.xlabel('Patient')
    plt.ylabel('Dice Score')
    plt.title('Performance par Patient')
    plt.legend()
    plt.xticks(x_pos, df_comparison['patient_id'], rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    return df_comparison


if __name__ == "__main__":
    # Configuration par défaut
    config = {
        'latent_dim': 128,
        'use_mask': True,
        'pred_hidden_dims': [64, 32],
        'pred_kernel_sizes': [(3,3,3), (3,3,3)]
    }
    
    # Chemins
    data_path = "data/processed"
    model_dir = "outputs/models"
    output_dir = "outputs/evaluation"
    
    try:
        print("🔍 Début de l'évaluation...")
        
        # 1. Évaluation principale
        results = evaluate_model(model_dir, data_path, config, output_dir)
        
        # 2. Visualisations
        print("\n📊 Création des visualisations...")
        visualize_predictions(data_path, model_dir, config, output_dir="outputs/visualization")
        
        # 3. Comparaison avec diffusion (si disponible)
        diffusion_dir = "data/masks"
        predictions_dir = os.path.join(output_dir, "predictions")
        if os.path.exists(diffusion_dir):
            print("\n⚖️ Comparaison avec les masques de diffusion...")
            compare_with_diffusion_masks(data_path, predictions_dir, diffusion_dir)
        
        print("\n✅ Évaluation terminée avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur durant l'évaluation: {e}")
        import traceback
        traceback.print_exc()
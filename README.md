# Prédiction d'Évolution Tumorale - IA en Grande Dimension

## 📋 Description du Projet

Ce projet implémente un système de prédiction de l'évolution tumorale sous traitement en utilisant des techniques d'apprentissage profond avancées. Le système combine deux approches complémentaires :

1. **Modèle de Diffusion** pour la segmentation automatique des tumeurs
2. **AutoEncoder + ConvLSTM** pour la prédiction temporelle de l'évolution tumorale

### Objectif
Prédire l'état d'une tumeur au **Timepoint_5** à partir des données des **Timepoint_1** et **Timepoint_2**.

## 🏗️ Architecture du Projet

```
projet_tumeur/
│
├── data/
│   ├── raw/                # Images NIfTI originales
│   │   └── PatientID_XX/
│   │       ├── Timepoint_1/
│   │       ├── Timepoint_2/
│   │       └── Timepoint_5/
│   ├── processed/          # Images prétraitées (64×128×128)
│   └── masks/              # Masques générés par diffusion
│
├── src/
│   ├── data_loader.py      # Prétraitement des données NIfTI
│   ├── diffusion_model.py  # Modèle de diffusion pour segmentation
│   ├── autoencoder.py      # AutoEncoder 3D débruitant
│   ├── convlstm.py         # ConvLSTM pour prédiction temporelle
│   ├── train.py            # Pipeline d'entraînement
│   ├── evaluate.py         # Évaluation et métriques
│   └── dif1.py             # Version alternative du modèle de diffusion
│
└── outputs/
    ├── models/             # Modèles entraînés (.pth)
    ├── predictions/        # Prédictions générées
    └── logs/               # Historiques d'entraînement
```

## 🔧 Installation

### Prérequis
- Python 3.8+
- CUDA compatible GPU (recommandé)
- 16GB+ RAM
- 50GB+ espace disque

### Installation des dépendances

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nibabel scipy matplotlib seaborn pandas tqdm scikit-learn opencv-python
```

## 📊 Données

### Format des Données
- **Format** : NIfTI (.nii.gz)
- **Modalité** : T1c (T1 avec contraste)
- **Structure** : Un patient par dossier avec 3 timepoints
- **Fichiers par timepoint** :
  - `PatientID_XX_Timepoint_Y_brain_t1c.nii.gz` (image cérébrale)
  - `PatientID_XX_Timepoint_Y_tumorMask.nii.gz` (masque tumoral)

### Prétraitement
- Redimensionnement à **64×128×128** voxels
- Normalisation Z-score pour les images
- Préservation des masques binaires (0/1)

## 🚀 Exécution du Projet

### Prérequis avant exécution

1. **Structure des données** : Placez vos données NIfTI dans `data/raw/` selon cette structure :
```
data/raw/
├── PatientID_01/
│   ├── Timepoint_1/
│   │   ├── PatientID_01_Timepoint_1_brain_t1c.nii.gz
│   │   └── PatientID_01_Timepoint_1_tumorMask.nii.gz
│   ├── Timepoint_2/
│   │   ├── PatientID_01_Timepoint_2_brain_t1c.nii.gz
│   │   └── PatientID_01_Timepoint_2_tumorMask.nii.gz
│   └── Timepoint_5/
│       └── PatientID_01_Timepoint_5_tumorMask.nii.gz
├── PatientID_02/
└── ...
```

### Exécution Complète du Pipeline

#### **Étape 1 : Prétraitement des Données**

```bash
cd src
python data_loader.py
```

**Ce que fait cette étape :**
- Charge toutes les images NIfTI depuis `data/raw/`
- Redimensionne à 64×128×128 voxels
- Applique la normalisation Z-score
- Sauvegarde dans `data/processed/`
- Affiche une visualisation de contrôle

**Temps d'exécution :** ~5-10 minutes selon le nombre de patients

#### **Étape 2 : Partie 1 - Segmentation par Diffusion**

```bash
python diffusion_model.py
```

**Ce que fait cette étape :**
- Entraîne le modèle de diffusion UNet3D (8 epochs par défaut)
- Génère des masques pour tous les timepoints
- Sauvegarde les modèles dans `outputs/models/`
- Crée des visualisations dans `outputs/logs/`

**Temps d'exécution :** ~2-4 heures selon le GPU
**Sorties :**
- `outputs/models/diffusion_model_best.pth`
- `data/masks/PatientID_XX/PatientID_XX_Timepoint_Y_generated_mask_*.nii.gz`

#### **Étape 3 : Partie 2 - Pipeline AutoEncoder + ConvLSTM**

```bash
python train.py
```

**Ce que fait cette étape :**
- Entraîne l'AutoEncoder débruitant (20 epochs)
- Entraîne le modèle ConvLSTM de prédiction (50 epochs)
- Sauvegarde les meilleurs modèles
- Génère les courbes d'entraînement

**Temps d'exécution :** ~3-6 heures selon le GPU
**Sorties :**
- `outputs/models/autoencoder.pth`
- `outputs/models/best_predictor.pth`
- `outputs/logs/ae_loss_curve.png`
- `outputs/logs/prediction_loss_curve.png`

#### **Étape 4 : Évaluation et Comparaison**

```bash
python evaluate.py
```

**Ce que fait cette étape :**
- Évalue le modèle ConvLSTM sur tous les patients
- Génère les prédictions finales
- Calcule toutes les métriques (Dice, IoU, etc.)
- Compare avec les masques de diffusion
- Crée des visualisations complètes

**Temps d'exécution :** ~30 minutes
**Sorties :**
- `outputs/evaluation/evaluation_results.csv`
- `outputs/evaluation/predictions/`
- `outputs/visualization/`

### 🚀 Exécution Rapide (Script Unique)

Si vous voulez exécuter tout le pipeline d'un coup, créez ce script `run_all.py` :

```python
import subprocess
import sys
import os

def run_command(cmd, description):
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Erreur: {result.stderr}")
        sys.exit(1)
    print(f"✅ {description} terminé avec succès")

if __name__ == "__main__":
    os.chdir("src")
    
    # Étape 1: Prétraitement
    run_command("python data_loader.py", "Prétraitement des données")
    
    # Étape 2: Diffusion
    run_command("python diffusion_model.py", "Entraînement modèle de diffusion")
    
    # Étape 3: ConvLSTM
    run_command("python train.py", "Entraînement AutoEncoder + ConvLSTM")
    
    # Étape 4: Évaluation
    run_command("python evaluate.py", "Évaluation et métriques")
    
    print("\n🎉 Pipeline complet terminé avec succès!")
    print("📊 Consultez les résultats dans outputs/")
```

Puis exécutez :
```bash
python run_all.py
```

### 🔧 Exécution avec Options Personnalisées

#### Configuration GPU limitée :
```bash
# Réduire les paramètres pour GPU < 8GB
export CUDA_VISIBLE_DEVICES=0
python train.py --batch_size 1 --patch_size 16,32,32 --epochs 10
```

#### Mode de débogage rapide :
```bash
# Exécution rapide avec moins d'epochs
python train.py --ae_epochs 5 --pred_epochs 10 --debug
```

#### Reprendre l'entraînement :
```bash
# Reprendre depuis un checkpoint
python train.py --resume outputs/models/best_predictor.pth
```

### 📋 Vérification de l'Exécution

Après chaque étape, vérifiez que les fichiers suivants sont créés :

**Après prétraitement :**
```bash
ls data/processed/PatientID_*/Timepoint_*/*.nii.gz
```

**Après diffusion :**
```bash
ls data/masks/PatientID_*/
ls outputs/models/diffusion_model_*.pth
```

**Après ConvLSTM :**
```bash
ls outputs/models/autoencoder.pth
ls outputs/models/best_predictor.pth
```

**Après évaluation :**
```bash
ls outputs/evaluation/evaluation_results.csv
ls outputs/evaluation/predictions/*.nii.gz
```

**Pipeline d'entraînement :**

#### AutoEncoder 3D Débruitant
- Architecture encoder-decoder avec skip connections
- Compression vers espace latent 128D
- Débruitage par ajout de bruit gaussien
- Reconstruction avec interpolation trilinéaire

#### ConvLSTM 3D
- Cellules ConvLSTM3D multi-couches
- Integration des features latentes + masques
- Prédiction de l'évolution spatiotemporelle

**Configuration par défaut :**
```python
config = {
    'batch_size': 1,
    'latent_dim': 128,
    'ae_epochs': 20,
    'pred_epochs': 50,
    'ae_lr': 1e-3,
    'pred_lr': 1e-4,
    'loss_alpha': 0.7,  # Balance BCE/Dice
}
```

### 4. Évaluation

```bash
python evaluate.py
```

**Métriques calculées :**
- Dice Coefficient
- IoU (Jaccard Index)
- Sensibilité/Spécificité
- Distance de Hausdorff

## 📈 Résultats et Performances

### Métriques Typiques
- **Dice Score** : 0.65-0.85 selon la complexité tumorale
- **IoU** : 0.50-0.75
- **Sensibilité** : 0.70-0.90

### Visualisations Générées
- Courbes d'entraînement (loss, métriques)
- Comparaisons prédictions vs. vérité terrain
- Distribution des performances par patient
- Matrices de corrélation entre métriques

## 🔬 Détails Techniques

### AutoEncoder 3D
- **Encoder** : 4 blocs Conv3D avec stride=2
- **Latent** : AdaptiveAvgPool3D + Linear (128D)
- **Decoder** : 4 blocs ConvTranspose3D symétriques
- **Régularisation** : BatchNorm3D, Dropout

### ConvLSTM 3D
- **Cellules** : Gates classiques (input, forget, output, cell)
- **Architecture** : Multi-couches [64, 32] hidden dims
- **Integration** : Concaténation features latentes + masques
- **Prédiction** : Upsampling + Conv3D final avec Sigmoid

### Optimisations Mémoire
- Gradient checkpointing
- Mixed precision training (AMP)
- Batch size adaptif
- Nettoyage périodique du cache CUDA

## 📋 Configuration Avancée

### Hyperparamètres Diffusion
```python
TIMESTEPS = 50
BETA_START = 1e-4
BETA_END = 0.02
PATCH_SIZE = (32, 64, 64)
```

### Hyperparamètres ConvLSTM
```python
hidden_dims = [64, 32]
kernel_sizes = [(3,3,3), (3,3,3)]
grid_size = (4, 4, 4)
```

## 🐛 Dépannage

### Problèmes Courants

**Erreur de mémoire GPU :**
```bash
# Réduire la taille des patches ou batch_size
PATCH_SIZE = (16, 32, 32)
batch_size = 1
```

**Convergence lente :**
```bash
# Ajuster les learning rates
ae_lr = 5e-4
pred_lr = 5e-5
```

**Données manquantes :**
```bash
# Vérifier la structure des dossiers
ls data/raw/PatientID_*/Timepoint_*/
```

## 📚 Références

1. **Ambiguous Medical Image Segmentation using Diffusion Models** - Segmentation par diffusion
2. **MedSegDiff-V2: Diffusion-Based Medical Image Segmentation with Transformer** - Architecture avancée
3. **Prediction of Ocean Weather Based on Denoising AutoEncoder and Convolutional LSTM** - Base méthodologique

## 🤝 Contribution

### Structure des Commits
```bash
git commit -m "feat: ajout modèle diffusion 3D"
git commit -m "fix: correction memory leak CUDA"
git commit -m "docs: mise à jour README"
```

### Tests
```bash
# Test du pipeline complet
python -m pytest tests/
# Test spécifique
python src/autoencoder.py  # Test unitaire
```

## 📄 Licence

Ce projet est développé dans le cadre de l'UE "IA en Grande Dimension". 
Usage académique uniquement.

---

## 🎯 Points Clés du Projet

✅ **Implémenté :**
- Modèle de diffusion 3D avec UNet
- AutoEncoder débruitant 3D
- ConvLSTM pour prédiction temporelle
- Pipeline d'évaluation complet
- Gestion optimisée de la mémoire

🔄 **Améliorations Possibles :**
- Attention mechanisms dans ConvLSTM
- Augmentation de données 3D
- Ensembling de modèles
- Optimisation hyperparamètres par Optuna

📊 **Résultats Attendus :**
- Segmentation précise des tumeurs (Dice > 0.7)
- Prédiction fiable de l'évolution temporelle
- Comparaison quantitative diffusion vs. ConvLSTM
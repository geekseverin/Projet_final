import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import random

# Dossiers
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Taille cible (D, H, W)
TARGET_SIZE = (64, 128, 128)

def load_nifti(filepath):
    """Charge un fichier NIfTI et retourne le volume numpy."""
    img = nib.load(filepath)
    return img.get_fdata()

def resize_and_normalize(volume, is_mask=False):
    """Redimensionne en 3D et normalise si c'est une image (Z-score)."""
    zoom_factors = (
        TARGET_SIZE[0] / volume.shape[0],
        TARGET_SIZE[1] / volume.shape[1],
        TARGET_SIZE[2] / volume.shape[2]
    )

    if is_mask:
        # Nearest neighbor pour ne pas lisser les labels
        resized = zoom(volume, zoom_factors, order=0)
    else:
        resized = zoom(volume, zoom_factors, order=1)
        if resized.std() > 0:
            resized = (resized - resized.mean()) / resized.std()

    return resized

def process_all_patients():
    """Parcourt tous les patients et sauvegarde les volumes normalisés dans processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    patients = [d for d in os.listdir(RAW_DIR) if d.startswith("PatientID_")]

    for patient in patients:
        patient_raw_path = os.path.join(RAW_DIR, patient)
        patient_processed_path = os.path.join(PROCESSED_DIR, patient)
        os.makedirs(patient_processed_path, exist_ok=True)

        for tp in ["Timepoint_1", "Timepoint_2", "Timepoint_5"]:
            tp_raw_path = os.path.join(patient_raw_path, tp)
            if not os.path.exists(tp_raw_path):
                continue

            brain_file = f"{patient}_{tp}_brain_t1c.nii.gz"
            mask_file = f"{patient}_{tp}_tumorMask.nii.gz"

            brain_path = os.path.join(tp_raw_path, brain_file)
            mask_path = os.path.join(tp_raw_path, mask_file)

            tp_processed_path = os.path.join(patient_processed_path, tp)
            os.makedirs(tp_processed_path, exist_ok=True)

            if os.path.exists(brain_path):
                brain_data = resize_and_normalize(load_nifti(brain_path), is_mask=False)
                nib.save(nib.Nifti1Image(brain_data, np.eye(4)),
                         os.path.join(tp_processed_path, brain_file))

            if os.path.exists(mask_path):
                mask_data = resize_and_normalize(load_nifti(mask_path), is_mask=True)
                nib.save(nib.Nifti1Image(mask_data, np.eye(4)),
                         os.path.join(tp_processed_path, mask_file))

            print(f"✅ {patient} - {tp} traité et sauvegardé")

def visualize_random_patient(overlay=True):
    """Affiche quelques slices image+masque, avec overlay optionnel."""
    patients = [d for d in os.listdir(PROCESSED_DIR) if d.startswith("PatientID_")]
    if not patients:
        print("Aucun patient trouvé dans processed/.")
        return

    patient = random.choice(patients)
    tp = "Timepoint_1"

    brain_file = f"{patient}_{tp}_brain_t1c.nii.gz"
    mask_file = f"{patient}_{tp}_tumorMask.nii.gz"

    brain_path = os.path.join(PROCESSED_DIR, patient, tp, brain_file)
    mask_path = os.path.join(PROCESSED_DIR, patient, tp, mask_file)

    brain = load_nifti(brain_path)
    mask = load_nifti(mask_path) if os.path.exists(mask_path) else np.zeros_like(brain)

    depth = brain.shape[0]
    slices_to_show = [depth // 4, depth // 2, 3 * depth // 4]

    fig, axes = plt.subplots(1, len(slices_to_show), figsize=(12, 5))
    for i, sl in enumerate(slices_to_show):
        axes[i].imshow(brain[sl], cmap="gray")
        if overlay:
            axes[i].imshow(mask[sl], cmap="Reds", alpha=0.4)
        axes[i].set_title(f"{patient} - Slice {sl}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Étape 1 : traitement et sauvegarde
    process_all_patients()

    # Étape 2 : affichage de contrôle
    visualize_random_patient(overlay=True)

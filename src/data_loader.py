import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Dossiers
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Taille cible (D, H, W)
TARGET_SIZE = (64, 128, 128)

def load_nifti(filepath):
    """Charge un fichier NIfTI"""
    img = nib.load(filepath)
    data = img.get_fdata()
    return data

def resize_and_normalize(volume, is_mask=False):
    """Redimensionne en 3D et applique la normalisation Z-score (seulement pour brain)"""
    zoom_factors = (
        TARGET_SIZE[0] / volume.shape[0],
        TARGET_SIZE[1] / volume.shape[1],
        TARGET_SIZE[2] / volume.shape[2]
    )
    if is_mask:
        # Nearest neighbor pour masques (préserve 0/1)
        resized = zoom(volume, zoom_factors, order=0)
    else:
        # Interpolation linéaire pour brain
        resized = zoom(volume, zoom_factors, order=1)
        if resized.std() > 0:
            resized = (resized - resized.mean()) / resized.std()
    return resized




def process_all_patients():
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

            if os.path.exists(brain_path):
                brain_data = resize_and_normalize(load_nifti(brain_path), is_mask=False)
                tp_processed_path = os.path.join(patient_processed_path, tp)
                os.makedirs(tp_processed_path, exist_ok=True)
                nib.save(nib.Nifti1Image(brain_data, np.eye(4)),
                         os.path.join(tp_processed_path, brain_file))

                if os.path.exists(mask_path):
                    mask_data = resize_and_normalize(load_nifti(mask_path), is_mask=True)
                    nib.save(nib.Nifti1Image(mask_data, np.eye(4)),
                             os.path.join(tp_processed_path, mask_file))

                print(f"✅ {patient} - {tp} traité et sauvegardé")

def visualize_random_patient():
    """Affiche quelques slices image+masque"""
    import random
    patients = [d for d in os.listdir(PROCESSED_DIR) if d.startswith("PatientID_")]
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

    fig, axes = plt.subplots(2, len(slices_to_show), figsize=(12, 6))
    for i, sl in enumerate(slices_to_show):
        axes[0, i].imshow(brain[sl], cmap="gray")
        axes[0, i].set_title(f"Image - Slice {sl}")
        axes[0, i].axis("off")

        axes[1, i].imshow(mask[sl], cmap="gray")
        axes[1, i].set_title(f"Masque - Slice {sl}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Étape 1 : traitement et sauvegarde
    process_all_patients()

    # Étape 2 : affichage de contrôle
    visualize_random_patient()
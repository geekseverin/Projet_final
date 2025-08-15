import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# --- Chemins vers tes fichiers ---
image_path = "data/processed/PatientID_0003/Timepoint_1/PatientID_0003_Timepoint_1_brain_t1c.nii.gz"
mask_path = "/home/severin/Bureau/vendredi/data/processed/PatientID_0003/Timepoint_1/PatientID_0003_Timepoint_1_tumorMask.nii.gz"

# --- Chargement des volumes ---
image_nii = nib.load(image_path)
mask_nii = nib.load(mask_path)

# Conversion en tableaux NumPy
image_data = image_nii.get_fdata()
mask_data = mask_nii.get_fdata()

# --- Sélection d'une coupe au milieu du volume ---
slice_index = image_data.shape[2] // 2  # milieu sur l'axe Z
image_slice = image_data[:, :, slice_index]
mask_slice = mask_data[:, :, slice_index]

# --- Affichage côte à côte ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image_slice.T, cmap='gray', origin='lower')
axes[0].set_title("IRM (slice)")
axes[0].axis('off')

axes[1].imshow(image_slice.T, cmap='gray', origin='lower')
axes[1].imshow(mask_slice.T, cmap='Reds', alpha=0.5, origin='lower')  # masque en rouge transparent
axes[1].set_title("IRM + Masque")
axes[1].axis('off')

plt.tight_layout()
plt.show()

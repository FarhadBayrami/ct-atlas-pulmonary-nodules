import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =========================
# 1️⃣ Frequency Map Builder
# =========================
mask_paths = [
    "sub_100012_ses_T0_acq_3_rec_B50f2_ct_lesion_1WarpedToTemplate.nii",
    "sub_100147_ses_T0_acq_2_rec_STANDARD250_ct_lesion_1WarpedToTemplate.nii",
    "sub_100158_ses_T1_acq_2_rec_STANDARD250_ct_lesion_1WarpedToTemplate.nii",
    "sub_100242_ses_T0_acq_2_rec_STANDARD250_ct_lesion_1WarpedToTemplate.nii",
    "sub_100280_ses_T0_acq_2_rec_B50f2_ct_lesion_1WarpedToTemplate.nii"
]

mask_paths = [p for p in mask_paths if os.path.exists(p)]
print(f"✅ Found {len(mask_paths)} warped masks.")
if not mask_paths:
    raise FileNotFoundError("❌ No mask files found! Please check paths.")

ref_img = sitk.ReadImage(mask_paths[0])
ref_arr = sitk.GetArrayFromImage(ref_img)
sum_all_subjects = np.zeros_like(ref_arr, dtype=np.uint16)

for subj_idx, path in enumerate(mask_paths):
    print(f"\n🧩 Subject {subj_idx+1}/{len(mask_paths)} → {os.path.basename(path)}")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    subject_sum = np.zeros_like(arr, dtype=np.uint16)
    for slice_idx in range(arr.shape[0]):
        slice_mask = (arr[slice_idx] > 0).astype(np.uint16)
        subject_sum[slice_idx] += slice_mask
    print(f"   ➕ Added {np.sum(subject_sum)} active voxels for subject {subj_idx+1}")
    sum_all_subjects += subject_sum

out_img = sitk.GetImageFromArray(sum_all_subjects)
out_img.CopyInformation(ref_img)
out_path = "frequency_absolute_map_doubleloop.nii.gz"
sitk.WriteImage(out_img, out_path)
print(f"\n💾 Saved absolute frequency map to: {out_path}")

nonzero = np.sum(sum_all_subjects > 0)
max_val = np.max(sum_all_subjects)
print(f"\n📊 Nonzero voxels: {nonzero:,}")
print(f"📈 Max frequency value: {max_val}")
print("✅ Process completed successfully.")


# =========================
# 2️⃣ Mid-Slice Visualization
# =========================
print("\n=== Visualizing Middle Non-Empty Slice ===")

img = sitk.ReadImage(out_path)
arr = sitk.GetArrayFromImage(img)
print("Shape:", arr.shape)

nonempty_slices = [z for z in range(arr.shape[0]) if np.any(arr[z] > 0)]
if not nonempty_slices:
    raise ValueError("❌ No non-empty slices found — image is fully empty!")

mid_slice = nonempty_slices[len(nonempty_slices)//2]
print(f"✅ Displaying non-empty slice Z = {mid_slice}")

plt.figure(figsize=(7,7))
plt.imshow(arr[mid_slice], cmap='hot', origin='lower')
plt.title(f"Axial Slice Z = {mid_slice} (Non-zero area)")
plt.axis("off")
plt.colorbar(label="Frequency (subjects count)")
plt.savefig("frequency_mid_slice.png", dpi=300, bbox_inches="tight")
plt.close()
print("💾 Saved mid-slice visualization → frequency_mid_slice.png")


# =========================
# 3️⃣ 3-View Projection Map
# =========================
print("\n=== Creating 3-View Frequency Projection ===")

freq_img = sitk.ReadImage(out_path)
freq_arr = sitk.GetArrayFromImage(freq_img).astype(np.float32)
freq_norm = freq_arr / np.max(freq_arr) if np.max(freq_arr) > 0 else freq_arr

proj_axial = np.max(freq_norm, axis=0)
proj_sagittal = np.max(freq_norm, axis=1)
proj_coronal = np.max(freq_norm, axis=2)

def pad_to_shape(arr, target_shape):
    pad_y = target_shape[0] - arr.shape[0]
    pad_x = target_shape[1] - arr.shape[1]
    pad_y1, pad_y2 = pad_y // 2, pad_y - pad_y // 2
    pad_x1, pad_x2 = pad_x // 2, pad_x - pad_x // 2
    return np.pad(arr, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant')

max_y = max(proj_axial.shape[0], proj_sagittal.shape[0], proj_coronal.shape[0])
max_x = max(proj_axial.shape[1], proj_sagittal.shape[1], proj_coronal.shape[1])
target_shape = (max_y, max_x)

proj_axial = pad_to_shape(proj_axial, target_shape)
proj_sagittal = pad_to_shape(proj_sagittal, target_shape)
proj_coronal = pad_to_shape(proj_coronal, target_shape)

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
cm = 'hot'
titles = ['Axial Projection (Y–X)', 'Sagittal Projection (Z–Y)', 'Coronal Projection (Z–X)']
xlabels = ['X', 'Z', 'Z']
ylabels = ['Y', 'Y', 'X']

for ax, data, title, xlabel, ylabel in zip(axs, [proj_axial, proj_sagittal, proj_coronal], titles, xlabels, ylabels):
    im = ax.imshow(data, cmap=cm, origin='lower', vmin=0, vmax=1, aspect='equal')
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)

divider = make_axes_locatable(axs[-1])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Absolute Frequency', fontsize=11)

plt.subplots_adjust(wspace=0.3, right=0.88)
plt.savefig("frequency_heatmap_3view_clean.png", dpi=300, bbox_inches="tight")
plt.close()
print("💾 Saved 3-view frequency heatmap → frequency_heatmap_3view_clean.png")

print("\n✅ All steps completed successfully!")

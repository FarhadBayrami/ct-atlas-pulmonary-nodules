#!/usr/bin/env python3
"""
Full pipeline script for processing frequency maps:
1. Mask with lung atlas
2. Apply Jet colormap
3. Overlay with BHI template
4. Flip to RAS
5. Extract and save best three anatomical views
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import SimpleITK as sitk


def flip_to_RAS(image):
    RAS = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if image.GetDirection() == RAS:
        return image
    flip_axes = [False, False, False]
    for i in (0, 4, 8):
        if image.GetDirection()[i] < 0:
            flip_axes[i // 3] = True
    return sitk.Flip(image, flip_axes)


def best_slice_by_variance(arr, axis):
    variances = []
    num_slices = arr.shape[axis]
    for i in range(num_slices):
        if axis == 0:
            slc = arr[i, :, :, :] if arr.ndim == 4 else arr[i, :, :]
        elif axis == 1:
            slc = arr[:, i, :, :] if arr.ndim == 4 else arr[:, i, :]
        else:
            slc = arr[:, :, i, :] if arr.ndim == 4 else arr[:, :, i]
        variances.append(np.var(slc))
    return int(np.argmax(variances))


def plot_best_three_views(img, image_name, output_dir):
    arr = sitk.GetArrayFromImage(img)
    best_z = best_slice_by_variance(arr, 0)
    best_y = best_slice_by_variance(arr, 1)
    best_x = best_slice_by_variance(arr, 2)

    if arr.ndim == 4:
        axial_slice = arr[best_z, :, :, :]
        sagittal_slice = arr[:, :, best_x, :]
        coronal_slice = arr[:, best_y, :, :]
    else:
        axial_slice = arr[best_z, :, :]
        sagittal_slice = arr[:, :, best_x]
        coronal_slice = arr[:, best_y, :]

    sagittal_slice = np.flipud(np.fliplr(sagittal_slice))
    coronal_slice = np.flipud(np.fliplr(coronal_slice))
    ref_shape = axial_slice.shape[:2]
    sagittal_resized = resize(sagittal_slice, ref_shape, anti_aliasing=True)
    coronal_resized = resize(coronal_slice, ref_shape, anti_aliasing=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('black')

    titles = [
        f"Axial (Z={best_z})",
        f"Sagittal (X={best_x})",
        f"Coronal (Y={best_y})"
    ]
    slices = [axial_slice, sagittal_resized, coronal_resized]

    for ax, slc, title in zip(axes, slices, titles):
        ax.imshow(slc)
        ax.set_title(title, fontsize=13, color='white', pad=8)
        ax.axis('off')

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.82, wspace=0.03, hspace=0.02)
    fig.text(
        0.5, 0.97,
        f"{image_name} — Best Axial/Sagittal/Coronal Slices (RAS)",
        color='white', fontsize=15, ha='center', va='top', fontweight='bold'
    )

    output_path = os.path.join(output_dir, f"{image_name}_best3views_RAS.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"💾 Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process frequency map through 5 stages.")
    parser.add_argument("--freq", required=True, help="Path to frequency_absolute_map.nii")
    parser.add_argument("--atlas", required=True, help="Path to BHI_atlas_binary.nii.gz")
    parser.add_argument("--bhi", required=True, help="Path to BHI_template.nii")
    parser.add_argument("--output", default="./output", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("\n[1/5] Masking frequency map with atlas...")
    freq_img = sitk.ReadImage(args.freq, sitk.sitkFloat32)
    # Read atlas, it will be used in Step 1 and Step 3
    atlas_img = sitk.ReadImage(args.atlas, sitk.sitkUInt8) 
    
    atlas_resampled = sitk.Resample(
        atlas_img, referenceImage=freq_img, transform=sitk.Transform(),
        interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0,
        outputPixelType=sitk.sitkUInt8
    )
    masked_freq = sitk.Mask(freq_img, atlas_resampled)

    print("[2/5] Applying Jet colormap...")
    # Note: 0 values (outside lungs) will be mapped to dark blue
    img_rescaled = sitk.RescaleIntensity(masked_freq, 0, 255)
    img_uint8 = sitk.Cast(img_rescaled, sitk.sitkUInt8)
    img_rgb = sitk.ScalarToRGBColormap(img_uint8, sitk.ScalarToRGBColormapImageFilter.Jet)

    print("[3/5] Overlaying with BHI template...")
    # Load BHI template (this will be the background)
    bhi_img = sitk.ReadImage(args.bhi, sitk.sitkFloat32)

    # Resample the colormap (foreground) to the BHI template's space
    freq_resampled_rgb = sitk.Resample(
        img_rgb, referenceImage=bhi_img, transform=sitk.Transform(),
        interpolator=sitk.sitkLinear, defaultPixelValue=0,
        outputPixelType=sitk.sitkVectorUInt8
    )

    # Convert the grayscale BHI template to an RGB image (to match foreground)
    bhi_scaled = sitk.RescaleIntensity(bhi_img, 0, 255)
    bhi_u8 = sitk.Cast(bhi_scaled, sitk.sitkUInt8)
    bhi_rgb = sitk.Compose(bhi_u8, bhi_u8, bhi_u8)

# --- START: CORRECTED (BACKWARDS-COMPATIBLE) OVERLAY LOGIC ---
    
    # We need the atlas mask resampled to the BHI template's space to act as a stencil.
    # We re-use 'atlas_img' which was read in Step 1.
    atlas_resampled_to_bhi = sitk.Resample(
        atlas_img, referenceImage=bhi_img, transform=sitk.Transform(),
        interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0,
        outputPixelType=sitk.sitkUInt8
    )

    # 1. Get the foreground (colormap inside the lungs)
    # sitk.Mask keeps pixels where the mask is 1 (or non-zero) and sets others to 0.
    # This works on vector (RGB) images.
    fg_image = sitk.Mask(freq_resampled_rgb, atlas_resampled_to_bhi)

    # 2. Get the background (BHI template outside the lungs)
    # Create an inverted mask (where atlas == 0)
    inverted_mask = sitk.Equal(atlas_resampled_to_bhi, 0)
    bg_image = sitk.Mask(bhi_rgb, inverted_mask)

    # 3. Add the foreground and background together.
    # Since their regions don't overlap, this is a simple addition.
    overlay_u8 = sitk.Add(fg_image, bg_image)
    
    # --- END: CORRECTED (BACKWARDS-COMPATIBLE) OVERLAY LOGIC ---
    
    overlay_path = os.path.join(args.output, "overlay_BHI_Frequency_JET.nii")
    sitk.WriteImage(overlay_u8, overlay_path)
    print(f"✅ Saved final overlay NIfTI: {overlay_path}")

    print("[4/5] Flipping to RAS orientation...")
    overlay_ras = flip_to_RAS(overlay_u8)

    print("[5/5] Generating best three anatomical views...")
    plot_best_three_views(overlay_ras, "overlay_BHI_Frequency_JET", args.output)

    print("\n🎉 Processing complete.")
    print(f"Final outputs:\n - {overlay_path}\n - overlay_BHI_Frequency_JET_best3views_RAS.png")


def viewer(overlay_path, base_template_path=None):
    overlay_img = sitk.ReadImage(overlay_path)
    overlay_arr = sitk.GetArrayFromImage(overlay_img).astype(np.float32)
    if overlay_arr.ndim == 3:
        overlay_arr = np.stack([overlay_arr] * 3, axis=-1)
    overlay_arr /= overlay_arr.max() + 1e-6
    if base_template_path:
        base_img = sitk.ReadImage(base_template_path)
        base_arr = sitk.GetArrayFromImage(base_img).astype(np.float32)
        base_arr = (base_arr - base_arr.min()) / (base_arr.max() - base_arr.min() + 1e-6)
        base_arr = np.stack([base_arr] * 3, axis=-1)
    else:
        base_arr = np.zeros_like(overlay_arr)
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()
    axis = 0
    slice_idx = overlay_arr.shape[axis] // 2
    alpha = 0.6
    axes_names = ['Z (axial)', 'Y (coronal)', 'X (sagittal)']
    def get_slice():
        if axis == 0:
            o, b = overlay_arr[slice_idx], base_arr[slice_idx]
        elif axis == 1:
            o, b = overlay_arr[:, slice_idx], base_arr[:, slice_idx]
        else:
            o, b = overlay_arr[:, :, slice_idx], base_arr[:, :, slice_idx]
        return np.flipud((1 - alpha) * b + alpha * o)
    img = ax.imshow(get_slice())
    ax.set_title(f"{axes_names[axis]} — Slice {slice_idx}", color='white')
    ax.axis('off')
    def on_key(event):
        nonlocal axis, slice_idx, alpha
        if event.key == 'up':
            slice_idx = min(slice_idx + 1, overlay_arr.shape[axis] - 1)
        elif event.key == 'down':
            slice_idx = max(slice_idx - 1, 0)
        elif event.key == 'right':
            axis = (axis + 1) % 3
            slice_idx = overlay_arr.shape[axis] // 2
        elif event.key == 'left':
            axis = (axis - 1) % 3
            slice_idx = overlay_arr.shape[axis] // 2
        elif event.key == 'a':
            alpha = max(0.0, alpha - 0.05)
        elif event.key == 'd':
            alpha = min(1.0, alpha + 0.05)
        else:
            return
        img.set_data(get_slice())
        ax.set_title(f"{axes_names[axis]} — Slice {slice_idx} — α={alpha:.2f}", color='white')
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)

    print("\n[6/6] Launching viewer...")
    viewer(overlay_path, args.bhi)

if __name__ == "__main__":
    main()

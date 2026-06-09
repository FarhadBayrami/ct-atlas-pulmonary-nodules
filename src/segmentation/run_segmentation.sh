--annotations_file /path/to/nifti_annotations.csv
--bids_dir /path/to/bids
--output_dir /path/to/output
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
import scipy.ndimage as nd
import SimpleITK as sitk


def flip_to_RAS(image):
    RAS = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if image.GetDirection() == RAS:
        print("✅ Image is already in RAS orientation.")
        return image
    else:
        flip_axes = [False, False, False]
        for i in (0, 4, 8):
            if image.GetDirection()[i] < 0:
                flip_axes[i // 3] = True
        print("↻ Flipping image to RAS orientation.")
        return sitk.Flip(image, flip_axes)


def denoise_if_needed(image_array, image_name):
    if "STANDARD" in image_name.upper():
        print(f"🧹 Applying Gaussian denoising to noisy scan: {image_name}")
        return nd.gaussian_filter(image_array, sigma=0.6)
    else:
        print(f"✅ No denoising applied (sharp image): {image_name}")
        return image_array


def plot_single_slice(img, seed_x, seed_y, bbox_x, bbox_y, bbox_w, bbox_h, slice_number, image_name, output_dir):
    arr = sitk.GetArrayFromImage(img)
    slice_number_fixed = arr.shape[0] - slice_number
    slice_to_plot = arr[slice_number_fixed, :, :].squeeze()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice_to_plot, cmap='gray')
    rect = patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h,
                             linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.scatter(seed_x, seed_y, color='red', marker='*', s=50)
    plt.title(f"{image_name}\nSlice {slice_number}")
    plt.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{image_name}_single.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"💾 Saved single-slice: {output_path}")


def plot_all_views(img, seed_x, seed_y, bbox_x, bbox_y, bbox_w, bbox_h, slice_number, image_name, output_dir):
    arr = sitk.GetArrayFromImage(img)
    arr = denoise_if_needed(arr, image_name)
    z_idx = arr.shape[0] - slice_number
    axial_slice = arr[z_idx, :, :]
    sagittal_slice = arr[:, :, int(seed_x)]
    coronal_index = arr.shape[1] // 2
    coronal_slice = arr[:, coronal_index, :]
    sagittal_slice = np.flipud(np.fliplr(sagittal_slice))
    coronal_slice = np.flipud(np.fliplr(coronal_slice))
    ref_shape = axial_slice.shape
    sagittal_slice_resized = resize(sagittal_slice, ref_shape, anti_aliasing=True)
    coronal_slice_resized = resize(coronal_slice, ref_shape, anti_aliasing=True)
    sag_scale_y = ref_shape[0] / sagittal_slice.shape[0]
    sag_scale_x = ref_shape[1] / sagittal_slice.shape[1]
    cor_scale_y = ref_shape[0] / coronal_slice.shape[0]
    cor_scale_x = ref_shape[1] / coronal_slice.shape[1]
    seed_sag_y = (arr.shape[0] - z_idx) * sag_scale_y
    seed_sag_x = (arr.shape[1] - seed_y) * sag_scale_x
    seed_cor_y = (arr.shape[0] - z_idx) * cor_scale_y
    seed_cor_x = (arr.shape[2] - seed_x) * cor_scale_x
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    slices = [axial_slice, sagittal_slice_resized, coronal_slice_resized]
    titles = [
        f'Axial – Slice {slice_number}',
        f'Sagittal – Slice {int(seed_x)}',
        f'Coronal – Slice {coronal_index}'
    ]
    for ax, slc, title in zip(axes, slices, titles):
        ax.imshow(slc, cmap='gray')
        ax.set_title(title, fontsize=13, color='white', pad=8)
        ax.axis('off')
    rect_ax = patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h, linewidth=2, edgecolor='lime', facecolor='none')
    axes[0].add_patch(rect_ax)
    axes[0].scatter(seed_x, seed_y, color='red', marker='*', s=50)
    axes[1].scatter(seed_sag_x, seed_sag_y, color='red', marker='*', s=50)
    rect_sag = patches.Rectangle((seed_sag_x - bbox_w / 2, seed_sag_y - bbox_h / 2),
                                 bbox_w, bbox_h, linewidth=2, edgecolor='lime', facecolor='none')
    axes[1].add_patch(rect_sag)
    axes[2].scatter(seed_cor_x, seed_cor_y, color='red', marker='*', s=50)
    rect_cor = patches.Rectangle((seed_cor_x - bbox_w / 2, seed_cor_y - bbox_h / 2),
                                 bbox_w, bbox_h, linewidth=2, edgecolor='lime', facecolor='none')
    axes[2].add_patch(rect_cor)
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.82, wspace=0.03, hspace=0.02)
    fig.patch.set_facecolor('black')
    fig.text(0.5, 0.97, f"{image_name}   —   RAS",
             color='white', fontsize=15, ha='center', va='top', fontweight='bold')
    output_path = os.path.join(output_dir, f"{image_name}_3views.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"💾 Saved 3-view: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_file", required=True, help="Path to nifti_annotations.csv")
    parser.add_argument("--bids_dir", required=True, help="Path to root BIDS directory")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    annotations = pd.read_csv(args.annotations_file)

    for _, row in annotations.iterrows():
        image_name = row['bidsname']
        img_path = None
        for root, _, files in os.walk(args.bids_dir):
            for f in files:
                if f == image_name or f == image_name + '.gz':
                    img_path = os.path.join(root, f)
                    break
            if img_path:
                break
        if not img_path:
            print(f"⚠️ File not found for: {image_name}")
            continue

        try:
            img = sitk.ReadImage(img_path)
            flipped = flip_to_RAS(img)
            plot_single_slice(flipped, row['seed_x'], row['seed_y'],
                              row['bbox_x'], row['bbox_y'],
                              row['bbox_w'], row['bbox_h'],
                              row['slice_number'], image_name, args.output_dir)
            plot_all_views(flipped, row['seed_x'], row['seed_y'],
                           row['bbox_x'], row['bbox_y'],
                           row['bbox_w'], row['bbox_h'],
                           row['slice_number'], image_name, args.output_dir)
        except Exception as e:
            print(f"⚠️ Skipped {image_name}: {e}")


if __name__ == "__main__":
    main()

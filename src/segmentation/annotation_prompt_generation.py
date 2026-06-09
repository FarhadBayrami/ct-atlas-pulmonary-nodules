import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
import SimpleITK as sitk

def flip_to_RAS(image):
    RAS = (1.0, 0.0, 0.0,
           0.0, 1.0, 0.0,
           0.0, 0.0, 1.0)
    if image.GetDirection() == RAS:
        return image
    flip_axes = [False, False, False]
    for i in (0, 4, 8):
        if image.GetDirection()[i] < 0:
            flip_axes[i // 3] = True
    return sitk.Flip(image, flip_axes)

def plot_with_annotations(img, seed_x, seed_y, bbox_x, bbox_y, bbox_w, bbox_h, slice_number, image_name, output_dir):
    arr = sitk.GetArrayFromImage(img)
    slice_number_fixed = arr.shape[0] - int(slice_number)
    slice_to_plot = arr[slice_number_fixed, :, :].squeeze()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice_to_plot, cmap='gray')
    rect = patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h,
                             linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.scatter(seed_x, seed_y, color='red', marker='*', s=50)
    plt.axis('off')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{image_name}_slice{slice_number}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"💾 Saved single-slice: {out_path}")

def plot_three_views(img, seed_x, seed_y, bbox_x, bbox_y, bbox_w, bbox_h, slice_number, image_name, output_dir):
    arr = sitk.GetArrayFromImage(img)
    z_idx = arr.shape[0] - int(slice_number)
    axial = arr[z_idx, :, :]
    sagittal = arr[:, :, int(seed_x)]
    coronal_index = arr.shape[1] // 2
    coronal = arr[:, coronal_index, :]
    sagittal = np.flipud(np.fliplr(sagittal))
    coronal = np.flipud(np.fliplr(coronal))
    ref_shape = axial.shape
    sag_resized = resize(sagittal, ref_shape, anti_aliasing=True)
    cor_resized = resize(coronal, ref_shape, anti_aliasing=True)
    sag_sy = ref_shape[0] / sagittal.shape[0]
    sag_sx = ref_shape[1] / sagittal.shape[1]
    cor_sy = ref_shape[0] / coronal.shape[0]
    cor_sx = ref_shape[1] / coronal.shape[1]
    seed_sag_y = (arr.shape[0] - z_idx) * sag_sy
    seed_sag_x = (arr.shape[1] - seed_y) * sag_sx
    seed_cor_y = (arr.shape[0] - z_idx) * cor_sy
    seed_cor_x = (arr.shape[2] - seed_x) * cor_sx
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    slices = [axial, sag_resized, cor_resized]
    titles = [f'Axial – Slice {slice_number}', f'Sagittal – X {int(seed_x)}', f'Coronal – Y {coronal_index}']
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
    fig.text(0.5, 0.97, f"{image_name}   —   RAS", color='white', fontsize=15,
             ha='center', va='top', fontweight='bold')
    out_path = os.path.join(output_dir, f"{image_name}_3views_RAS.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"💾 Saved 3-view: {out_path}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.annotations_file)
    for i, row in df.iterrows():
        image_name = row["bidsname"]
        if pd.isna(row.get("seed_x")):
            continue
        nifti_path = None
        for root, _, files in os.walk(args.bids_dir):
            for f in files:
                if f == image_name:
                    nifti_path = os.path.join(root, f)
                    break
            if nifti_path:
                break
        if not nifti_path:
            print(f"⚠️ NIFTI not found: {image_name}")
            continue
        try:
            img = sitk.ReadImage(nifti_path)
            img = flip_to_RAS(img)
            base_name = os.path.splitext(image_name)[0]
            plot_with_annotations(img, float(row["seed_x"]), float(row["seed_y"]),
                                  float(row["bbox_x"]), float(row["bbox_y"]),
                                  float(row["bbox_w"]), float(row["bbox_h"]),
                                  int(row["slice_number"]), base_name, args.output_dir)
            plot_three_views(img, float(row["seed_x"]), float(row["seed_y"]),
                             float(row["bbox_x"]), float(row["bbox_y"]),
                             float(row["bbox_w"]), float(row["bbox_h"]),
                             int(row["slice_number"]), base_name, args.output_dir)
        except Exception as e:
            print(f"❌ Error in {image_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch NIFTI Annotation Visualizer (single & 3-view)")
    parser.add_argument("--annotations_file", required=True, help="Path to nifti_annotations.csv")
    parser.add_argument("--bids_dir", required=True, help="Path to root BIDS dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to save output PNGs")
    args = parser.parse_args()
    main(args)

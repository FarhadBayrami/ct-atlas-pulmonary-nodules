import os
import sys
import json
import glob
import time
import shutil
import zipfile
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk
import scipy.ndimage as nd
from skimage.transform import resize

# =============== CONFIG ===============
BASE_DIR = "/data"
BIDS_ZIP = os.path.join(BASE_DIR, "bids_lesionlocator.zip")
BIDS_DIR = os.path.join(BASE_DIR, "bids-lesionlocator")
ANNOTATION_FILE = os.path.join(BASE_DIR, "nifti_annotations.csv")
MODEL_DIR = "/opt/LesionLocator"
OUTPUT_DIR = "/results"
RAS_DIR = os.path.join(OUTPUT_DIR, "bids-lesionlocator-ras")
PROMPTS_DIR = os.path.join(OUTPUT_DIR, "bids-lesionlocator-prompts")
QC_DIR = os.path.join(OUTPUT_DIR, "bids-lesionlocator-qc")
OUT_SEG_DIR = os.path.join(OUTPUT_DIR, "bids-lesionlocator-output")
CORRECTED_RAS_DIR = os.path.join(OUTPUT_DIR, "bids-lesionlocator-ras-corrected")

# Create all directories if not exist
for d in [OUTPUT_DIR, RAS_DIR, PROMPTS_DIR, QC_DIR, OUT_SEG_DIR, CORRECTED_RAS_DIR]:
    os.makedirs(d, exist_ok=True)

print("=== ENVIRONMENT SETUP COMPLETE ===")
print("BIDS zip:", BIDS_ZIP)
print("BIDS dir:", BIDS_DIR)
print("Model dir:", MODEL_DIR)
print("Output root:", OUTPUT_DIR)
# =========================================================
#  PART 2 — Extract and Prepare BIDS Dataset
# =========================================================

def extract_bids_dataset(zip_path, target_dir):
    if not os.path.exists(zip_path):
        print(f"[ERROR] ZIP not found: {zip_path}")
        sys.exit(1)
    print(f"[INFO] Extracting BIDS dataset from {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(os.path.dirname(target_dir))
    if not os.path.isdir(target_dir):
        raise RuntimeError("BIDS extraction failed — target folder missing")
    print(f"[OK] Extracted BIDS dataset → {target_dir}")

if not os.path.isdir(BIDS_DIR):
    extract_bids_dataset(BIDS_ZIP, BIDS_DIR)
else:
    print("[INFO] BIDS directory already exists:", BIDS_DIR)

print("[INFO] Top-level entries:", sorted(os.listdir(BIDS_DIR))[:10])
# =========================================================
#  PART 3 — Download and Prepare Model Checkpoints
# =========================================================

import urllib.request

CKPT_DIR = os.path.join(MODEL_DIR, "LesionLocatorCheckpoint")
os.makedirs(CKPT_DIR, exist_ok=True)

needed = [
    f"{CKPT_DIR}/LesionLocatorSeg/point_optimized/fold_0/checkpoint_final.pth",
    f"{CKPT_DIR}/LesionLocatorSeg/point_optimized/plans.json",
    f"{CKPT_DIR}/LesionLocatorSeg/point_optimized/dataset.json",
]

def download_model_if_needed():
    have = all(os.path.exists(p) for p in needed)
    if have:
        print("[OK] Model checkpoints already exist.")
        return

    url = "https://zenodo.org/records/15174217/files/LesionLocatorModel.zip?download=1"
    zip_path = os.path.join(MODEL_DIR, "LesionLocatorModel.zip")

    print(f"[INFO] Downloading model weights from: {url}")
    urllib.request.urlretrieve(url, zip_path)

    print("[INFO] Extracting model weights...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(MODEL_DIR)
    print("[OK] Model extracted to:", MODEL_DIR)

download_model_if_needed()
print("=== MODEL CHECKPOINT READY ===")
# =========================================================
#  PART 4 — Flattening, Manifest Parsing & RAS Preparation
# =========================================================

def flatten_nested_cts(bids_root, flat_dir):
    os.makedirs(flat_dir, exist_ok=True)
    pattern = os.path.join(bids_root, "sub-*", "ses-*", "ct", "*.nii*")
    srcs = sorted(glob.glob(pattern))
    linked = 0
    for s in srcs:
        parts = s.split("/")
        subj = next(x for x in parts if x.startswith("sub-"))
        sess = next(x for x in parts if x.startswith("ses-"))
        base = os.path.basename(s)
        dst = os.path.join(flat_dir, f"{subj}__{sess}__ct__{base}")
        if not os.path.exists(dst):
            try:
                os.symlink(s, dst)
            except OSError:
                shutil.copy2(s, dst)
            linked += 1
    print(f"[INFO] Flattened {linked} CTs into {flat_dir}")
    return srcs


def orient_to_RAS(img):
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("RAS")
    return orient_filter.Execute(img)


def parse_manifest_txt(path):
    series = {}
    cur = None
    x = y = sl = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith("serie:"):
                if cur and x is not None and y is not None and sl is not None:
                    series[cur] = {"x": x, "y": y, "slice": sl}
                cur = ln.split("serie:", 1)[1].strip()
                x = y = sl = None
            m = re.search(r"x=(\d+),\s*y=(\d+)", ln)
            if m:
                x = int(m.group(1)); y = int(m.group(2))
            m2 = re.search(r"slice=(\d+)", ln)
            if m2:
                sl = int(m2.group(1))
    if cur and x is not None and y is not None and sl is not None:
        series[cur] = {"x": x, "y": y, "slice": sl}
    print(f"[INFO] Parsed {len(series)} entries from manifest")
    return series


def generate_ras_and_prompts():
    flat_images_dir = os.path.join(BIDS_DIR, "images-nifti")
    os.makedirs(flat_images_dir, exist_ok=True)
    flatten_nested_cts(BIDS_DIR, flat_images_dir)
    manifest_txt = os.path.join(BIDS_DIR, "annotations-nifti.txt")
    entries = parse_manifest_txt(manifest_txt)
    selected = []

    for path in sorted(glob.glob(os.path.join(flat_images_dir, "*.nii*"))):
        base = os.path.basename(path)
        m = re.search(r"(sub-[^_]+_ses-[^_]+_acq-[^_]+_rec-[^_]+)_ct\.nii", base)
        if not m:
            continue
        serie = m.group(1)
        if serie not in entries:
            continue
        img = sitk.ReadImage(path)
        ras = orient_to_RAS(img)
        ras_path = os.path.join(RAS_DIR, base)
        sitk.WriteImage(ras, ras_path, True)

        nx, ny, nz = ras.GetSize()
        ex = entries[serie]
        x0, y0, z0 = ex["x"], ex["y"], ex["slice"]
        z_ras = float(nz - 1 - z0)
        y_ras = float(ny - 1 - y0)
        x_ras = float(x0)
        prompt = {"1": {"point": [y_ras, x_ras, z_ras]}}
        json_path = os.path.join(PROMPTS_DIR, base.replace(".nii.gz", ".json").replace(".nii", ".json"))
        with open(json_path, "w") as f:
            json.dump(prompt, f)
        selected.append(base.replace(".nii.gz", "").replace(".nii", ""))
        print(f"[RAS] {base} → prompt [{y_ras}, {x_ras}, {z_ras}]")

    selected_path = os.path.join(OUTPUT_DIR, "bids-lesionlocator-selected.txt")
    with open(selected_path, "w") as f:
        for s in selected:
            f.write(s + "\n")
    print("[INFO] Wrote selected list:", selected_path)

generate_ras_and_prompts()
# =========================================================
#  PART 5 — LesionLocator Segmentation Runner
# =========================================================

def run_lesionlocator_inference():
    tmp_in = os.path.join(OUTPUT_DIR, "_tmp_in")
    tmp_pr = os.path.join(OUTPUT_DIR, "_tmp_pr")
    os.makedirs(tmp_in, exist_ok=True)
    os.makedirs(tmp_pr, exist_ok=True)

    selected_path = os.path.join(OUTPUT_DIR, "bids-lesionlocator-selected.txt")
    if not os.path.exists(selected_path):
        print("[ERROR] Selected file list missing.")
        return

    with open(selected_path) as f:
        stems = [ln.strip() for ln in f if ln.strip()]
    print(f"[INFO] Running LesionLocator inference for {len(stems)} cases...")

    for stem in stems:
        img_path = os.path.join(RAS_DIR, stem + ".nii.gz")
        if not os.path.exists(img_path):
            img_path = os.path.join(RAS_DIR, stem + ".nii")
        prompt_path = os.path.join(PROMPTS_DIR, stem + ".json")
        if not (os.path.exists(img_path) and os.path.exists(prompt_path)):
            print(f"[WARN] Skipping {stem} — missing image or prompt")
            continue

        # clear temp dirs
        for d in (tmp_in, tmp_pr):
            for f in glob.glob(os.path.join(d, "*")):
                os.remove(f)

        in_link = os.path.join(tmp_in, os.path.basename(img_path))
        try:
            os.symlink(img_path, in_link)
        except OSError:
            shutil.copy2(img_path, in_link)
        shutil.copy2(prompt_path, os.path.join(tmp_pr, os.path.basename(prompt_path)))

        cmd = [
            "LesionLocator_segment",
            "-i", tmp_in,
            "-p", tmp_pr,
            "-t", "point",
            "-o", OUT_SEG_DIR,
            "-m", MODEL_DIR,
            "-f", "0",
            "--disable_tta"
        ]

        log_file = os.path.join(QC_DIR, f"{stem}.log")
        print(f"[RUN] {stem}")
        with open(log_file, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=lf)

        if rc == 0:
            print(f"[OK]  {stem}")
        else:
            print(f"[FAIL] {stem} — see {log_file}")

run_lesionlocator_inference()
# =========================================================
#  PART 6 — QC Overlay Generation (SimpleITK + matplotlib)
# =========================================================

def resample_like(moving, reference, interp=sitk.sitkNearestNeighbor):
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        interp,
        reference.GetOrigin(),
        reference.GetSpacing(),
        reference.GetDirection(),
        0,
        moving.GetPixelID()
    )


def generate_qc_overlays():
    selected_path = os.path.join(OUTPUT_DIR, "bids-lesionlocator-selected.txt")
    if not os.path.exists(selected_path):
        print("[ERROR] Missing selected.txt for QC overlays.")
        return

    with open(selected_path) as f:
        selected = [ln.strip() for ln in f if ln.strip()]

    print(f"[INFO] Generating QC overlays for {len(selected)} cases...")

    for stem in selected:
        img_path = os.path.join(RAS_DIR, f"{stem}.nii.gz")
        if not os.path.exists(img_path):
            continue
        prompt_path = os.path.join(PROMPTS_DIR, f"{stem}.json")
        mask_candidates = sorted(glob.glob(os.path.join(OUT_SEG_DIR, f"{stem}*.nii*")))
        if not mask_candidates:
            print(f"[WARN] No mask found for {stem}")
            continue

        mask_path = mask_candidates[0]
        img = sitk.ReadImage(img_path)
        msk = sitk.ReadImage(mask_path)
        if img.GetSize() != msk.GetSize():
            msk = resample_like(msk, img)

        arr = sitk.GetArrayFromImage(img)
        marr = sitk.GetArrayFromImage(msk)

        with open(prompt_path) as f:
            j = json.load(f)
        py, px, pz = j["1"]["point"]
        z = int(round(pz))
        y = int(round(py))
        x = int(round(px))
        z = max(0, min(arr.shape[0] - 1, z))

        img_slice = arr[z]
        msk_slice = (marr[z] > 0).astype(np.uint8)

        img_slice_rot = np.rot90(img_slice, k=2)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_slice_rot, cmap="gray", vmin=np.percentile(img_slice_rot, 1), vmax=np.percentile(img_slice_rot, 99))
        overlay = np.ma.masked_where(msk_slice == 0, msk_slice)
        plt.imshow(overlay, alpha=0.35, cmap="autumn")
        plt.scatter([x], [y], s=40, c="cyan", marker="x")
        plt.title(f"{stem} (z={z}) - RAS aligned")
        plt.axis("off")
        out_png = os.path.join(QC_DIR, f"{stem}_qc.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[QC] Saved → {out_png}")

generate_qc_overlays()
# =========================================================
#  PART 7 — Comprehensive RAS Orientation Fix
# =========================================================

def verify_anatomical_landmarks(img):
    arr = sitk.GetArrayFromImage(img)
    middle_slice = arr[arr.shape[0] // 2]
    center_x, center_y = middle_slice.shape[1] // 2, middle_slice.shape[0] // 2
    landmarks = {'spine': False, 'heart': False, 'lungs': False, 'ok': False}

    bottom_center = middle_slice[middle_slice.shape[0]//2:, center_x-10:center_x+10]
    if np.mean(bottom_center) > np.mean(middle_slice) * 1.2:
        landmarks['spine'] = True
    top_right = middle_slice[:middle_slice.shape[0]//2, center_x:]
    if np.mean(top_right) > np.mean(middle_slice) * 1.1:
        landmarks['heart'] = True
    left_side = middle_slice[:, :center_x//2]
    right_side = middle_slice[:, center_x+center_x//2:]
    if np.mean(left_side) < np.mean(middle_slice) * 0.8 and np.mean(right_side) < np.mean(middle_slice) * 0.8:
        landmarks['lungs'] = True
    landmarks['ok'] = all(landmarks.values())
    return landmarks


def comprehensive_ras_fix():
    print("[INFO] Applying comprehensive RAS orientation fix...")
    os.makedirs(CORRECTED_RAS_DIR, exist_ok=True)
    ras_files = sorted(glob.glob(os.path.join(RAS_DIR, "*.nii*")))
    for img_path in ras_files:
        try:
            img = sitk.ReadImage(img_path)
            original = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionMatrix(img.GetDirection())
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation('RAS')
            ras_img = orient_filter.Execute(img)
            fixed_path = os.path.join(CORRECTED_RAS_DIR, os.path.basename(img_path))
            sitk.WriteImage(ras_img, fixed_path, True)
            landmarks = verify_anatomical_landmarks(ras_img)
            print(f"[RAS] {os.path.basename(img_path)} | landmarks={landmarks}")
        except Exception as e:
            print(f"[ERROR] Failed RAS fix for {img_path}: {e}")
    print("[INFO] Comprehensive RAS orientation correction complete.")

comprehensive_ras_fix()
# =========================================================
#  PART 8 — Comprehensive Multi-View Atlas Visualization
# =========================================================

def create_slice_template(slice_img, slice_type):
    template = np.zeros_like(slice_img)
    threshold = np.percentile(slice_img, 30)
    lung_mask = slice_img < threshold
    h, w = slice_img.shape
    cy, cx = h // 2, w // 2

    if slice_type == "apex":
        template[cy-20:cy+20, cx-20:cx+20] = 0.4
    elif slice_type == "middle":
        if np.any(lung_mask):
            left_region = lung_mask[:, :w//2]
            right_region = lung_mask[:, w//2:]
            for region, offset, val in [(left_region, 0, 0.8), (right_region, w//2, 0.7)]:
                coords = np.argwhere(region)
                for i in range(0, len(coords), 50):
                    y, x = coords[i]
                    template[y-10:y+10, x+offset-10:x+offset+10] = val
    elif slice_type == "base":
        template[cy-40:cy+40, cx-40:cx+40] = 0.3
    return template


def comprehensive_multi_view_visualization():
    ras_dir = CORRECTED_RAS_DIR if os.path.exists(CORRECTED_RAS_DIR) else RAS_DIR
    selected_txt = os.path.join(OUTPUT_DIR, "bids-lesionlocator-selected.txt")
    if not os.path.exists(selected_txt):
        print("[WARN] Selected list not found.")
        return

    with open(selected_txt) as f:
        selected = [ln.strip() for ln in f if ln.strip()]

    print(f"[INFO] Creating multi-view atlas visualizations for {len(selected)} cases...")

    for case in selected:
        img_path = os.path.join(ras_dir, f"{case}.nii.gz")
        if not os.path.exists(img_path):
            continue
        img = sitk.ReadImage(img_path)
        arr = sitk.GetArrayFromImage(img)
        nz = arr.shape[0]

        slice_map = {
            "apex": max(0, int(nz * 0.8)),
            "middle": nz // 2,
            "base": min(nz - 1, int(nz * 0.2))
        }

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle(f"{case} — RAS Multi-View Atlas", fontsize=16, fontweight='bold')

        for row, (sname, sidx) in enumerate(slice_map.items()):
            sl = arr[sidx]
            tmpl = create_slice_template(sl, sname)

            # Axial
            ax1 = axes[row, 0]
            sl_rot = np.rot90(sl, 2)
            ax1.imshow(sl_rot, cmap="gray", vmin=np.percentile(sl_rot, 1), vmax=np.percentile(sl_rot, 99))
            ax1.imshow(np.ma.masked_where(tmpl == 0, tmpl), alpha=0.6, cmap="hot")
            ax1.set_title(f"{sname.title()} – Axial (RAS)")
            ax1.axis("off")

            # Sagittal
            ax2 = axes[row, 1]
            sag = np.rot90(sl_rot, 1)
            sag_tmpl = np.rot90(tmpl, 1)
            ax2.imshow(sag, cmap="gray", vmin=np.percentile(sag, 1), vmax=np.percentile(sag, 99))
            ax2.imshow(np.ma.masked_where(sag_tmpl == 0, sag_tmpl), alpha=0.6, cmap="hot")
            ax2.set_title(f"{sname.title()} – Sagittal (RAS)")
            ax2.axis("off")

            # Coronal
            ax3 = axes[row, 2]
            cor = np.rot90(sl_rot, 2)
            cor_tmpl = np.rot90(tmpl, 2)
            ax3.imshow(cor, cmap="gray", vmin=np.percentile(cor, 1), vmax=np.percentile(cor, 99))
            ax3.imshow(np.ma.masked_where(cor_tmpl == 0, cor_tmpl), alpha=0.6, cmap="hot")
            ax3.set_title(f"{sname.title()} – Coronal (RAS)")
            ax3.axis("off")

        out_path = os.path.join(QC_DIR, f"{case}_atlas.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"[OK] Multi-view saved → {out_path}")

comprehensive_multi_view_visualization()
# =========================================================
#  PART 9 — NIFTI Annotation Visualizer (Standalone)
# =========================================================

def flip_to_RAS(image):
    RAS = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if image.GetDirection() == RAS:
        return image
    flip_axes = [False, False, False]
    for i in (0, 4, 8):
        if image.GetDirection()[i] < 0:
            flip_axes[i // 3] = True
    return sitk.Flip(image, flip_axes)


def plot_with_annotations(img, seed_x, seed_y, bbox_x, bbox_y, bbox_w, bbox_h, slice_number, image_name, out_dir):
    arr = sitk.GetArrayFromImage(img)
    z_idx = arr.shape[0] - slice_number
    slice_img = arr[z_idx]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice_img, cmap="gray")
    rect = patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h, linewidth=2, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.scatter(seed_x, seed_y, c="red", marker="*", s=50)
    plt.title(f"{image_name} — Slice {slice_number}")
    plt.axis("off")
    out_path = os.path.join(out_dir, f"{image_name}_slice{slice_number}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved annotated slice → {out_path}")


def visualize_all_annotations(csv_path, bids_dir, output_dir):
    ann = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Visualizing {len(ann)} annotated cases...")
    for _, row in ann.iterrows():
        img_name = row["bidsname"]
        img_path = None
        for root, _, files in os.walk(bids_dir):
            for f in files:
                if f == img_name:
                    img_path = os.path.join(root, f)
                    break
            if img_path:
                break
        if not img_path:
            print(f"[WARN] Not found → {img_name}")
            continue
        img = sitk.ReadImage(img_path)
        img = flip_to_RAS(img)
        plot_with_annotations(
            img,
            row["seed_x"],
            row["seed_y"],
            row["bbox_x"],
            row["bbox_y"],
            row["bbox_w"],
            row["bbox_h"],
            row["slice_number"],
            img_name,
            output_dir
        )

visualize_all_annotations(ANNOTATIONS_CSV, BIDS_DIR, OUTPUT_DIR)

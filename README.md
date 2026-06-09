# A Computed Tomography Atlas of Pulmonary Nodules for Lung Cancer Screening

Master's thesis — University of Bologna, Department of Computer Science and Engineering  
Artificial Intelligence for Medicine M · Academic Year 2024/2025

**Candidate:** Farhad Bayrami  
**Supervisor:** Prof. Stefano Diciotti  
**Co-Supervisor:** Dr. Giulia Raffaella De Luca

---

## Overview

This repository contains the full code and thesis for an automated framework that constructs a probabilistic pulmonary nodule atlas from low-dose CT (LDCT) data. The pipeline integrates two key components: [LesionLocator](https://github.com/MIC-DKFZ/LesionLocator), a zero-shot deep learning model for nodule segmentation, and [ANTs](https://github.com/ANTsX/ANTs), a deformable image registration toolkit. Together they transform individual LDCT scans from the [NLST](https://www.cancer.gov/types/lung/research/nlst) dataset into a shared anatomical template space, enabling voxel-wise population-level analysis of nodule distribution.

The resulting frequency maps confirm known clinical patterns — higher nodule prevalence in the upper lobes and perihilar regions — while establishing a reproducible spatial prior for future risk stratification and detection models.

---

## Pipeline

The workflow runs in three sequential phases:

**Phase 1 — Nodule Segmentation**  
LesionLocator is applied to preprocessed NLST LDCT scans using bounding-box annotations from the [Sybil](https://github.com/reginabarzilaygroup/Sybil) dataset as prompts. Each scan produces a 3D binary nodule mask.

**Phase 2 — Image Registration**  
All CT volumes and nodule masks are registered to a 3D lung template derived from 30 NLST scans using ANTs multi-stage registration: rigid → affine → SyN diffeomorphic. Nearest-neighbour interpolation preserves binary mask integrity.

**Phase 3 — Atlas Construction**  
Warped masks are aggregated voxel-wise across the cohort to produce absolute and normalized frequency maps, heatmaps, and multi-view overlays showing the spatial distribution of nodules across the population.

---

## Repository Structure

```
ct-atlas-pulmonary-nodules/
├── docs/
│   ├── thesis.pdf                        # Full 43-page dissertation
│   └── figures/                          # Exported frequency map figures
│       ├── frequency_map_axial.png
│       ├── frequency_map_sagittal.png
│       ├── frequency_map_coronal.png
│       └── frequency_map_overview.png
├── notebooks/
│   ├── 01_segmentation_phase1.ipynb      # Phase 1: LesionLocator segmentation
│   ├── 03_atlas_analysis_phase3.ipynb    # Phase 3: frequency maps and heatmaps
│   └── 04_figures_reproduction.ipynb    # Reproduce all thesis figures
├── src/
│   ├── segmentation/
│   │   ├── annotation_prompt_generation.py   # Generate LesionLocator JSON prompts
│   │   ├── run_segmentation.sh               # Batch segmentation script
│   │   └── run_segmentation_final.sh         # Final production segmentation script
│   ├── registration/
│   │   └── ants_registration.sh              # ANTs rigid + affine + SyN pipeline
│   └── atlas/
│       ├── frequency_map.py                  # Absolute frequency map from warped masks
│       ├── frequency_map_3view.py            # 3-view heatmap projections
│       ├── heatmap_annotations.py            # Annotation overlays and location heatmaps
│       ├── multiview_overlay.py              # Axial/coronal/sagittal PNG overlays
│       └── overlay_pipeline.py              # Full overlay pipeline with Jet colormap
├── environment.yml                           # Conda environment specification
├── CITATION.cff                              # Citation metadata
├── .gitignore
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/FarhadBayrami/ct-atlas-pulmonary-nodules.git
cd ct-atlas-pulmonary-nodules
```

**2. Create the conda environment**
```bash
conda env create -f environment.yml
conda activate ct-atlas
```

**3. Install ANTs** (required for Phase 2)
```bash
# macOS
brew install ants

# Linux — build from source or use the prebuilt binary
# https://github.com/ANTsX/ANTs/releases
```

**4. Install LesionLocator** (required for Phase 1)
```bash
git clone https://github.com/MIC-DKFZ/LesionLocator.git
cd LesionLocator
pip install -e .
# Download checkpoints from Hugging Face as described in LesionLocator's README
```

---

## Data Access

This project uses a subset of the [National Lung Screening Trial (NLST)](https://www.cancer.gov/types/lung/research/nlst) dataset. Access requires an approved application through the [NCI Cancer Data Access System (CDAS)](https://cdas.cancer.gov/).

The nodule annotations used as LesionLocator prompts are from the [Sybil](https://github.com/reginabarzilaygroup/Sybil) dataset (Mikhael et al., 2023). From the 969 annotated series, 509 baseline (T0) scans were used.

> Data files (`.nii.gz`, `.dcm`) are excluded from this repository via `.gitignore` in compliance with NLST data use agreements.

---

## Usage

### Phase 1 — Segmentation

Generate prompts from Sybil annotations and run LesionLocator:

```bash
# Generate JSON prompt files from bounding box annotations
python src/segmentation/annotation_prompt_generation.py \
  --annotations path/to/sybil_annotations.csv \
  --output_dir path/to/prompts/

# Run batch segmentation
bash src/segmentation/run_segmentation_final.sh \
  path/to/nifti_scans/ \
  path/to/prompts/ \
  path/to/masks_output/
```

### Phase 2 — Registration

Register each scan and its nodule mask to the lung template:

```bash
bash src/registration/ants_registration.sh \
  path/to/BHI_template.nii.gz \
  path/to/moving_scan.nii.gz \
  path/to/nodule_mask.nii.gz \
  path/to/warped_output/
```

### Phase 3 — Atlas Construction

Generate frequency maps and visualizations from the warped masks:

```bash
# Absolute frequency map
python src/atlas/frequency_map.py \
  --input_folder path/to/warped_masks/ \
  --output path/to/frequency_absolute_map.nii.gz

# 3-view heatmap
python src/atlas/frequency_map_3view.py \
  --frequency_map path/to/frequency_absolute_map.nii.gz \
  --output path/to/heatmap_3view.png

# Multi-view overlay on template
python src/atlas/multiview_overlay.py \
  --input_folder path/to/warped_masks/ \
  --template_path path/to/BHI_template.nii.gz \
  --output_folder path/to/outputs/
```

Or run the full pipeline interactively through the notebooks:

```bash
jupyter notebook notebooks/01_segmentation_phase1.ipynb
```

---

## Results

Frequency maps computed from 500 NLST LDCT scans show consistent nodule clustering in the upper lobes and perihilar regions, aligning with clinical observations from large screening cohorts.

| Output | Description |
|--------|-------------|
| `frequency_absolute_map.nii.gz` | Voxel-wise nodule count across all subjects |
| `frequency_map_axial.png` | Axial projection of nodule frequency |
| `frequency_map_sagittal.png` | Sagittal projection |
| `frequency_map_coronal.png` | Coronal projection |
| `frequency_map_overview.png` | Combined 3-view normalized distribution map |

See `docs/figures/` for all exported visualizations.

---

## Citation

If you use this code or thesis in your work, please cite:

```bibtex
@mastersthesis{bayrami2025ctatlas,
  author    = {Farhad Bayrami},
  title     = {A Computed Tomography Atlas of Pulmonary Nodules for Lung Cancer Screening},
  school    = {University of Bologna},
  year      = {2025},
  type      = {Master's Thesis},
  note      = {Artificial Intelligence for Medicine M,
               Supervisor: Prof. Stefano Diciotti,
               Co-Supervisor: Dr. Giulia Raffaella De Luca}
}
```

---

## Acknowledgements

- Prof. Stefano Diciotti and Dr. Giulia Raffaella De Luca for supervision and the BHI lung template
- The [MIC-DKFZ](https://github.com/MIC-DKFZ) team for LesionLocator
- The [ANTsX](https://github.com/ANTsX) team for the ANTs ecosystem
- The National Cancer Institute for NLST data access (CDAS Project: NLST-1175)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

**Contact:** farhad.bayrami@studio.unibo.it

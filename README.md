<div align="center">

# 🫁 CT Atlas of Pulmonary Nodules for Lung Cancer Screening
### Automated Probabilistic Nodule Atlas from NLST Low-Dose CT Data

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://img.shields.io/badge/Thesis-Master's%20Degree-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-NLST%20509%20scans-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Tools-LesionLocator%20%2B%20ANTs-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/University-Bologna-red?style=flat-square"/>
</p>

*An automated pipeline that constructs a probabilistic pulmonary nodule atlas from low-dose CT scans — integrating zero-shot deep learning segmentation with deformable image registration.*

**Candidate:** Farhad Bayrami
**Supervisor:** Prof. Stefano Diciotti
**Co-Supervisor:** Dr. Giulia Raffaella De Luca
**Programme:** Artificial Intelligence for Medicine M — University of Bologna, A.Y. 2024/2025

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline](#-pipeline)
- [Dataset](#-dataset)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

---

## 🔬 Overview

This repository contains the full code and thesis for an automated framework that constructs a **probabilistic pulmonary nodule atlas** from low-dose CT (LDCT) data. The pipeline integrates two key components:

- **[LesionLocator](https://github.com/MIC-DKFZ/LesionLocator)** — a zero-shot deep learning model for nodule segmentation
- **[ANTs](https://github.com/ANTsX/ANTs)** — a deformable image registration toolkit

Together they transform individual LDCT scans from the NLST dataset into a shared anatomical template space, enabling voxel-wise population-level analysis of nodule distribution.

The resulting frequency maps confirm known clinical patterns — **higher nodule prevalence in the upper lobes and perihilar regions** — while establishing a reproducible spatial prior for future risk stratification and detection models.

---

## ⚙️ Pipeline

| Phase | Component | Description |
|-------|-----------|-------------|
| **1 — Segmentation** | LesionLocator (zero-shot) | Applies deep learning nodule segmentation to NLST LDCT scans using Sybil bounding-box prompts |
| **2 — Registration** | ANTs (rigid → affine → SyN) | Registers all CT volumes and nodule masks to a shared 3D lung template |
| **3 — Atlas Construction** | Voxel-wise aggregation | Warped masks are accumulated to produce absolute and normalised frequency maps |

---

## 📦 Dataset

**National Lung Screening Trial (NLST)** — National Cancer Institute

🔗 [cancer.gov/types/lung/research/nlst](https://www.cancer.gov/types/lung/research/nlst)

| Property | Value |
|----------|-------|
| Scans used | 509 baseline (T0) LDCT scans |
| Annotations | Sybil dataset bounding-box prompts |
| Access | Requires NCI CDAS approval (Project: NLST-1175) |

> ⚠️ Raw data files (`.nii.gz`, `.dcm`) are excluded from this repository via `.gitignore` in compliance with NLST data use agreements.

---

## 📊 Results

Frequency maps computed from 500 NLST LDCT scans show consistent nodule clustering in the upper lobes and perihilar regions, aligning with clinical observations from large screening cohorts.

| Output | Description |
|--------|-------------|
| `frequency_absolute_map.nii.gz` | Voxel-wise nodule count across all subjects |
| `frequency_map_axial.png` | Axial projection of nodule frequency |
| `frequency_map_sagittal.png` | Sagittal projection |
| `frequency_map_coronal.png` | Coronal projection |
| `frequency_map_overview.png` | Combined 3-view normalised distribution map |

See `docs/figures/` for all exported visualisations.

---

## 🚀 Getting Started

### Prerequisites

```bash
# 1. Clone the repository
git clone https://github.com/FarhadBayrami/ct-atlas-pulmonary-nodules.git
cd ct-atlas-pulmonary-nodules

# 2. Create conda environment
conda env create -f environment.yml
conda activate ct-atlas

# 3. Install ANTs (required for Phase 2)
# macOS:
brew install ants
# Linux: https://github.com/ANTsX/ANTs/releases

# 4. Install LesionLocator (required for Phase 1)
git clone https://github.com/MIC-DKFZ/LesionLocator.git
cd LesionLocator
pip install -e .
```

---

## 🏃 Usage

### Phase 1 — Segmentation

```bash
python src/segmentation/annotation_prompt_generation.py \
  --annotations path/to/sybil_annotations.csv \
  --output_dir path/to/prompts/

bash src/segmentation/run_segmentation_final.sh \
  path/to/nifti_scans/ path/to/prompts/ path/to/masks_output/
```

### Phase 2 — Registration

```bash
bash src/registration/ants_registration.sh \
  path/to/BHI_template.nii.gz \
  path/to/moving_scan.nii.gz \
  path/to/nodule_mask.nii.gz \
  path/to/warped_output/
```

### Phase 3 — Atlas Construction

```bash
python src/atlas/frequency_map.py \
  --input_folder path/to/warped_masks/ \
  --output path/to/frequency_absolute_map.nii.gz

python src/atlas/frequency_map_3view.py \
  --frequency_map path/to/frequency_absolute_map.nii.gz \
  --output path/to/heatmap_3view.png
```

Or run interactively via notebooks:

```bash
jupyter notebook notebooks/01_segmentation_phase1.ipynb
```

---

## 📁 Project Structure

| Path | Description |
|------|-------------|
| `docs/thesis.pdf` | Full 43-page dissertation |
| `docs/figures/` | Exported frequency map figures |
| `notebooks/01_segmentation_phase1.ipynb` | Phase 1: LesionLocator segmentation |
| `notebooks/03_atlas_analysis_phase3.ipynb` | Phase 3: frequency maps and heatmaps |
| `notebooks/04_figures_reproduction.ipynb` | Reproduce all thesis figures |
| `src/segmentation/` | Prompt generation and batch segmentation scripts |
| `src/registration/ants_registration.sh` | ANTs rigid + affine + SyN pipeline |
| `src/atlas/` | Frequency map, heatmap, and overlay scripts |
| `environment.yml` | Conda environment specification |
| `CITATION.cff` | Citation metadata |
| `LICENSE` | MIT License |
| `README.md` | Project documentation |

---

## 📚 Citation

If you use this code or thesis in your work, please cite:

| Field | Value |
|-------|-------|
| **Author** | Farhad Bayrami |
| **Title** | A Computed Tomography Atlas of Pulmonary Nodules for Lung Cancer Screening |
| **Type** | Master's Thesis |
| **School** | University of Bologna |
| **Year** | 2025 |
| **Supervisor** | Prof. Stefano Diciotti |
| **Co-Supervisor** | Dr. Giulia Raffaella De Luca |
---

## 🙏 Acknowledgements

- Prof. Stefano Diciotti and Dr. Giulia Raffaella De Luca for supervision and the BHI lung template
- The [MIC-DKFZ](https://github.com/MIC-DKFZ) team for LesionLocator
- The [ANTsX](https://github.com/ANTsX) team for the ANTs ecosystem
- The National Cancer Institute for NLST data access (CDAS Project: NLST-1175)

---

## 👤 Author

**Farhad Bayrami**
MSc — Artificial Intelligence for Medicine, University of Bologna
📧 [farhad.bayrami@studio.unibo.it](mailto:farhad.bayrami@studio.unibo.it)
🔗 [GitHub](https://github.com/FarhadBayrami)

---

<div align="center">
  <sub>Master's Thesis · University of Bologna · 2024/2025</sub>
</div>

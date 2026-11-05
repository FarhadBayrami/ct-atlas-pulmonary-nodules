# A Computed Tomography Atlas of Pulmonary Nodules for Lung Cancer Screening

![Thesis](https://img.shields.io/badge/MSc%20Thesis-University%20of%20Bologna-990000)
![Year](https://img.shields.io/badge/Academic%20Year-2024%2F2025-blue)
![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-informational)
![Data](https://img.shields.io/badge/Data-NLST%20LDCT-lightgrey)

**Master's Thesis** in *Artificial Intelligence for Medicine M*
MSc in Artificial Intelligence · Department of Computer Science and Engineering
Alma Mater Studiorum – Università di Bologna · Academic Year 2024/2025

| | |
|---|---|
| **Candidate** | Farhad Bayrami |
| **Supervisor** | Prof. Stefano Diciotti |
| **Co-Supervisor** | Dr. Giulia Raffaella De Luca |

> **Note on code availability.** The implementation (segmentation, registration and atlas-construction scripts and notebooks) is kept in a private repository because it works with NLST data obtained under a data-use agreement (CDAS Project NLST-1175). Access to the code or the full thesis text is available on request. See [Contact](#contact).

![Nodule frequency map overlaid on the lung template](figures/frequency_overlay_template.png)

---

## Table of Contents

- [Abstract](#abstract)
- [Pipeline](#pipeline)
- [Data](#data)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Tools](#tools)
- [Citation](#citation)
- [Contact](#contact)

---

## Abstract

Lung cancer is the leading cause of cancer-related mortality worldwide, and early detection remains critical for improving survival rates. This thesis proposes an integrated framework for automated construction of a probabilistic nodule atlas using low-dose computed tomography (LDCT) data from the National Lung Screening Trial (NLST). The approach leverages two key methodologies: LesionLocator, a deep learning model for zero-shot segmentation, and the Advanced Normalization Tools (ANTs) framework for inter-subject registration. Together, these tools address challenges of anatomical variability, segmentation consistency, and inter-patient comparison in LDCT screening data.

LesionLocator was applied to the NLST dataset to generate nodule masks. Subsequently, these masks were transformed into a standardized anatomical space defined by a previously obtained lung template via Symmetric diffeomorphic (SyN) non-linear registration. A nodule frequency map in the template space is then constructed by accumulating the warped nodule masks. This map highlights high-density nodule regions in the upper lobes and lung periphery, validating both the registration accuracy and the model's capacity to reproduce clinically known nodule distribution patterns in lung cancer screening.

This work establishes a reproducible computational workflow that generates objective spatial priors for population-based analysis. The resulting atlas can serve as a foundational component for future risk stratification tools and detection models, thereby complementing current radiological assessment by automating high-throughput spatial measurements. Overall, this thesis lays the groundwork for next-generation diagnostic systems that integrate machine learning, medical imaging, and population-based analysis to enhance early cancer detection and personalized patient care.

---

## Pipeline

The workflow has three sequential phases. Each phase consumes the outputs of the previous one.

| Phase | Goal | Method | Output |
|---|---|---|---|
| **0 · Pre-processing** | Make every scan geometrically consistent | NLST scans in BIDS / NIfTI; voxel spacing and affine preserved; orientation harmonised (LPS → RAS) with SimpleITK | Standardised NIfTI CT volumes |
| **1 · Nodule segmentation** | Delineate each nodule in 3D without retraining | [LesionLocator](https://github.com/MIC-DKFZ/LesionLocator) (MIC-DKFZ), zero-shot, point-prompted. Prompts generated automatically from the centre of the [Sybil](https://github.com/reginabarzilaygroup/Sybil) bounding-box annotations, converted to RAS and written as JSON | 3D binary nodule mask per scan + QC overlays (axial / sagittal / coronal) |
| **2 · Registration to template** | Map every subject into one anatomical space | Lungs isolated with a U-Net lung mask ([lungmask](https://github.com/JoHof/lungmask)); [ANTs](https://github.com/ANTsX/ANTs) v2.5.4 multi-stage registration to a normative lung template built from 30 NLST scans | Warped CT volumes and warped nodule masks in template space |
| **3 · Atlas construction** | Population-level spatial analysis | Voxel-wise summation of warped masks, normalisation by cohort size, Gaussian smoothing (σ = 1), thresholding of non-zero clusters | Absolute and normalised frequency maps, non-zero region mask, location-distribution map |

### Registration details (Phase 2)

| Stage | Degrees of freedom | Similarity metric | Iterations | Purpose |
|---|---|---|---|---|
| Rigid | 6 | Mutual Information | 100 × 50 | Correct gross patient-positioning differences |
| Affine | 12 | Mutual Information | 100 × 50 × 25 | Correct body size and scanner calibration differences |
| SyN (diffeomorphic) | Non-linear | Cross-Correlation (radius 4) | 100 × 70 × 50 × 10 | Capture local deformations while preserving topology |

Multi-resolution shrink factors 8 × 4 × 2 × 1 were used. The resulting affine and warp transforms were applied with `antsApplyTransforms`: **B-spline** interpolation for CT intensities and **nearest-neighbour** interpolation for binary masks, so that masks stay binary.

---

## Data

| Item | Detail |
|---|---|
| Source | [National Lung Screening Trial (NLST)](https://cdas.cancer.gov/nlst/), U.S. National Cancer Institute |
| Access | De-identified data under approved request, CDAS Project NLST-1175 |
| Subset | NLST series with nodule bounding-box annotations released by the Sybil authors (Mikhael et al., 2023) |
| Cohort used | 509 baseline (T0) LDCT series, one annotation per series used as the segmentation prompt |
| Template | Normative lung template built from 30 NLST scans with `antsMultivariateTemplateConstruction2.sh` |

No imaging data are distributed in this repository.

---

## Results

**Voxel-wise nodule frequency map** in template space (axial, sagittal and coronal projections):

![Frequency map projections](figures/frequency_projections.png)

**Location-distribution map** of nodule positions across the template:

![Location distribution map](figures/location_distribution.png)

### Key findings

- Every subject's nodule mask was mapped into one shared lung template, which makes voxel-wise population analysis possible.
- Nodule density is highest in the **upper lobes**, with additional clustering in **central / perihilar** regions. This is consistent with the distributions reported for large screening cohorts.
- The extreme peripheral regions of the lower lobes are sparsely populated in this cohort.
- Diffeomorphic registration preserved the shape of warped masks, so imperfect segmentations still gave coherent population maps.

---

## Limitations and Future Work

| Limitation | Planned direction |
|---|---|
| Relatively small subset of NLST | Scale the pipeline to the full NLST cohort and to external screening datasets |
| Segmentation quality was not quantitatively validated against expert delineations | Radiologist review and Dice-based evaluation per nodule type (solid, part-solid, GGO) |
| Registration quality was assessed qualitatively | Landmark error, Jacobian-determinant and overlap metrics |
| ANTs SyN is computationally expensive (≈ 20 min per scan) | GPU or learning-based registration for large-scale use |

---

## Tools

| Category | Tools |
|---|---|
| Segmentation | LesionLocator, PyTorch, MONAI |
| Registration | ANTs (antsRegistration, antsApplyTransforms, template construction), lungmask |
| Image I/O and processing | SimpleITK, NiBabel, NumPy |
| Analysis and visualisation | Pandas, Matplotlib, Seaborn |
| Automation | Bash, GNU Parallel |

---

## Citation

```bibtex
@mastersthesis{bayrami2025ctatlas,
  author = {Farhad Bayrami},
  title  = {A Computed Tomography Atlas of Pulmonary Nodules for Lung Cancer Screening},
  school = {Alma Mater Studiorum -- Universit{\`a} di Bologna},
  year   = {2025},
  type   = {Master's Thesis},
  note   = {MSc in Artificial Intelligence. Supervisor: Prof. Stefano Diciotti; Co-Supervisor: Dr. Giulia Raffaella De Luca}
}
```

---

## Acknowledgements

- Prof. Stefano Diciotti and Dr. Giulia Raffaella De Luca for supervision and the lung template.
- The [MIC-DKFZ](https://github.com/MIC-DKFZ) team for LesionLocator, and the [ANTsX](https://github.com/ANTsX) team for ANTs.
- The authors of Sybil for releasing the nodule annotations.
- The National Cancer Institute for access to NLST data through CDAS.

---

## Contact

**Farhad Bayrami**
Email: farhad.bayrami@studio.unibo.it
[LinkedIn](https://www.linkedin.com/in/farhad-bayrami-9410a8279/) · [Google Scholar](https://scholar.google.com/citations?user=zFwyidYAAAAJ) · [GitHub](https://github.com/FarhadBayrami)

To request access to the code repository or the full thesis, please email with a short note on your intended use.

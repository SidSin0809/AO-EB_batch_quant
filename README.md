# AO-EB_batch_quant
Automated, reproducible quantification of acridine-orange / ethidium-bromide (AO/EB) live-dead assays: field‐by‐field, image-agnostic, and ready for high-throughput screening.

The script discovers paired green (AO) and red (EB) TIFFs, segments nuclei, classifies them into Live / Early Apoptosis / Late Apoptosis / Necrotic, and writes CSV summaries plus optional overlays & label masks.

# Key features

| **Dual-channel segmentation** – three modes (`green`, `union`, `fallback`) to cope with pure-red nuclei or failed AO staining.     |

| **Vectorised statistics** – NumPy `bincount` makes v2.16 ≈4× faster & ≈40 % leaner than classic `regionprops` loops.               |

| **Self-tuning thresholds** – multi-Otsu, auto valley-finding, and optional GMM maintain robust AO/EB splits without hand tweaking. |

| **Safety-cap for EB split** – prevents red-only floods from wrecking the Live/Early balance (`--max_r_z_thr`, default 10).         |

| **Rich QC output** – per-field overlays, 16-bit label masks, cell-level CSV, and a JSON meta-file for provenance.                  |

| **Headless batch-ready** – runs equally well on workstations or HPC nodes; fully CLI-driven.                                       |


# Requirements (Installation)
pip install numpy scipy scikit-image scikit-learn tifffile joblib tqdm pandas

Python ≥ 3.9 and scikit-image ≥ 0.19 are supported

# Usage 
Folder layout (any depth):

├── field01_G.tif       # AO   (max-proj or single slice)

├── field01_R.tif       # EB

├── field01_BF.tif      # optional bright-field

├── field02_G.tif

└── ...


# All flags & options
| Flag                 | Default       | Purpose                                                                    |
| -------------------- | ------------- | -------------------------------------------------------------------------- |
| `-i  / --input_dir`  | **required**  | Folder containing \*\_G.tif / \*\_R.tif pairs.                             |
| `-o  / --output`     | `summary.csv` | Summary CSV path; sister files written alongside.                          |
| `-j  / --workers`    | *all CPUs*    | Parallel fields; ≤ number of images.                                       |
| `--seg_mode`         | `union`       | `green` = AO only; `union` = AO ∪ EB; `fallback` = AO then EB if AO fails. |
| `--min_size`         | 80 px²        | Small-object removal before watershed.                                     |
| `--top_hat`          | 15 px         | White-top-hat radius (0 → disabled).                                       |
| `--ratio`            | 0             | Manually force AO/EB ratio split (0 → auto).                               |
| `--ratio_fallback`   | 1.2           | Valley split if auto fails.                                                |
| `--g_z_early`        | **8**         | AO z-score needed for **Early** call.                                      |
| `--ratio_mult_early` | **2.0**       | AO/EB ratio multiplier for **Early**.                                      |
| `--area_factor_late` | 1.25          | Late = EB+ nucleus < median × factor.                                      |
| `--ratio_factor_nec` | 0.30          | EB+ nucleus is necrotic if ratio < factor × split.                         |
| `--r_z_nec`          | **12**        | Red z-score that forces Necrotic.                                          |
| `--g_z_thr_min`      | **2**         | AO z-score floor for building EB threshold pool (union-mode stabiliser).   |
| `--max_r_z_thr`      | **10**        | Upper cap on EB split after Otsu.                                          |
| `--overlay`          | off           | Save RGB overlay (green/BF base + red contours).                           |
| `--masks`            | off           | Save 16-bit label mask (`*_mask.tif`).                                     |
| `--qc`               | off           | Overlay uses BF instead of AO if present.                                  |
| `--no_auto_orient`   | off           | Disable auto 90° rotation of skinny images.                                |
| `--first_channel`    | off           | Use first Z-slice instead of max projection.                               |

# Algorithmic workflow
1. Discovery – walks input_dir, grouping by stem; accepts _G/_R/_BF and dash/numbered variants.
2. Pre-processing – local background removal (white_tophat).
3. Segmentation
   
     multi-Otsu → binary mask
   
     distance-transform & H-maxima seeding
   
     watershed → preliminary labels
   
     artefact filter: size, border-touch, extreme eccentricity, BF texture.
   
   
4. Vector-ised feature extraction – per-label AO/EB sums via np.bincount; compute z-scores & AO/EB ratio.
5. Adaptive thresholds
   
     EB split: Otsu on red-z of AO-positive pool → capped at max_r_z_thr.
   
     AO/EB ratio split: valley-finding + optional 2-component GMM inside EB-negatives.
   
   
6. Gating logic
   
      EB–  & g_z>g_z_early & ratio>mult        → Early

      EB–  otherwise                           → Live

      EB+  & (red-dom)                         → Necrotic

      EB+  & (small area)                      → Late

      EB+  otherwise                           → Necrotic


10. Output
    
     summary.csv – per-field counts & percentages.
    
     summary_cells.csv – per-nucleus metrics & class.
    
     *.meta.json – full CLI + computed thresholds.
    
     Optional PNG overlays & 16-bit masks under qc/.
    


# Example: treat-vs-control batch
python ao_eb_batch_quant_v2.py \
       -i ./treated -o results/treated.csv --seg_mode union --overlay --masks

python ao_eb_batch_quant_v2.py \
       -i ./control -o results/control.csv --seg_mode union

python ao_eb_batch_quant_v2.py -i ./ -o results/summary.csv --overlay --masks --qc --seg_mode union

# Troubleshooting
| Symptom                             | Likely cause & fix                                                                            |
| ----------------------------------- | --------------------------------------------------------------------------------------------- |
| *Too many Early*                    | Tighten `--g_z_early` or `--ratio_mult_early`.                                                |
| *Late stays 0*                      | AO staining too weak in EB-positive nuclei → raise `--area_factor_late` or lower `--r_z_nec`. |
| *Necrotic skyrockets in union‐mode* | Lower `--max_r_z_thr` (default 10) or raise `--g_z_thr_min`.                                  |
| *Red-only nuclei missing*           | Use `--seg_mode union` (or `fallback` if AO sometimes absent altogether).                     |

## Citation
If you use AO/EB Batch Quant in a publication, please cite this repository. Feel free to raise issues or pull requests—contributions are welcome!



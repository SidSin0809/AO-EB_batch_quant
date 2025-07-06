#!/usr/bin/env python3
"""
AO/EB Batch Quant v2.16  –  tuned gates + red-z safety-cap
=========================================================

* Default gates tightened to match the empirical “fallback ≈ union” balance
  you validated on your dataset.

      --g_z_early         8        
      --ratio_mult_early  2.0      
      --r_z_nec           12       
      --g_z_thr_min       2        

* New CLI switch  --max_r_z_thr  (default **10**):
  After Otsu, the EB/“red” z-score threshold is clamped to this value
  so a flood of red-only nuclei can’t push the split sky-high.

"""

from __future__ import annotations
import argparse, json, logging, os, warnings
from pathlib import Path
from typing import Dict, List, Union

import numpy as np, pandas as pd, tifffile as tiff
from joblib import Parallel, delayed
from skimage import exposure, filters, io, measure, morphology, segmentation, util
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
_log = logging.getLogger(__name__)

# ─── regionprops API probe (works 0.19 → 0.25) ──────────────────────
try:
    _REGIONPROPS_SUPPORTS_COOR = "coordinates" in measure.regionprops.__code__.co_varnames
except Exception:
    _REGIONPROPS_SUPPORTS_COOR = False

# ───────────────────────── CLI ───────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("AO/EB batch quant v2.16")
    p.add_argument("-i", "--input_dir", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, default=Path("summary.csv"))
    p.add_argument("-j", "--workers", type=int, default=os.cpu_count())

    # segmentation knobs
    p.add_argument("--min_size", type=int, default=80)
    p.add_argument("--top_hat", type=int, default=15)
    p.add_argument("--seg_mode", choices=("green", "union", "fallback"), default="union")

    # gating knobs (updated defaults)
    p.add_argument("--ratio", type=float, default=0.0)
    p.add_argument("--ratio_fallback", type=float, default=1.2)
    p.add_argument("--g_z_early", type=float, default=8.0)            # ↑
    p.add_argument("--ratio_mult_early", type=float, default=2.0)     # ↑
    p.add_argument("--area_factor_late", type=float, default=1.25)
    p.add_argument("--ratio_factor_nec", type=float, default=0.30)
    p.add_argument("--r_z_nec", type=float, default=12.0)             # ↓
    p.add_argument("--g_z_thr_min", type=float, default=2.0,          # ↑
                   help="AO z-score above which a nucleus is used for Otsu red split")
    p.add_argument("--max_r_z_thr", type=float, default=10.0,
                   help="Maximum allowed red-z threshold after Otsu (union safety cap)")

    # I/O & misc
    p.add_argument("--overlay", action="store_true")
    p.add_argument("--masks", action="store_true")
    p.add_argument("--qc", action="store_true")
    p.add_argument("--no_auto_orient", action="store_true")
    p.add_argument("--first_channel", action="store_true")
    return p.parse_args()

# ─────────── helper functions (unchanged core) ──────────────────────
digit_channel = {"01": "BF", "02": "G", "03": "R", "04": "M"}

def split_numeric(stem: str):
    if "-" in stem: base, suff = stem.split("-", 1)
    elif "_" in stem: base, suff = stem.split("_", 1)
    else: return None
    if "_" not in suff: return None
    num, chan = suff.split("_", 1)
    return (base, digit_channel[num]) if num in digit_channel and chan.upper().startswith(digit_channel[num]) else None

def discover_bases(root: Path):
    bases = set()
    for tif in root.rglob("*.tif"):
        st = tif.stem
        if st.endswith(("_G", "_R", "_BF", "_M")):
            bases.add(tif.with_name(st.rsplit("_", 1)[0])); continue
        parsed = split_numeric(st)
        if parsed: bases.add(tif.with_name(parsed[0]))
    return sorted(bases)

def _reduce_2d(im, first_slice):
    while im.ndim > 2:
        if im.ndim == 3 and im.shape[-1] in (3, 4):
            im = im.mean(axis=-1); break
        im = im[0] if first_slice else im.max(axis=0)
    return im.astype(np.float32)

def _auto_orient(im, enabled=True):
    h, w = im.shape
    return im.T if enabled and min(h, w) < 16 and max(h, w) > 4 * min(h, w) else im

def read_plane(path, first_slice, orient_ok):
    return _auto_orient(_reduce_2d(tiff.imread(str(path)), first_slice), orient_ok)

def load_set(base: Path, A):
    L = lambda p: read_plane(p, A.first_channel, not A.no_auto_orient)
    g = r = bf = None
    for pat in ("{b}_G.tif", "{b}-02_G.tif", "{b}_02_G.tif"):
        p = base.with_name(pat.format(b=base.name))
        if p.exists(): g = L(p); break
    for pat in ("{b}_R.tif", "{b}-03_R.tif", "{b}_03_R.tif"):
        p = base.with_name(pat.format(b=base.name))
        if p.exists(): r = L(p); break
    for pat in ("{b}_BF.tif", "{b}-01_BF.tif", "{b}_01_BF.tif"):
        p = base.with_name(pat.format(b=base.name))
        if p.exists(): bf = L(p); break
    if g is None or r is None:
        raise FileNotFoundError(base.name)
    return g, r, bf

def white_tophat(img, r): return morphology.white_tophat(img, footprint=morphology.disk(r)) if r > 0 else img

def segment(chan, mn):
    thr, _ = filters.threshold_multiotsu(chan, classes=3)[:2]
    mask = morphology.remove_small_objects(chan > thr, mn)
    dist = ndi.distance_transform_edt(mask)
    markers = measure.label(morphology.h_maxima(dist, 0.4))
    return segmentation.watershed(-dist, markers, mask=mask)

def _iter_regions(lbl):
    if _REGIONPROPS_SUPPORTS_COOR: yield from measure.regionprops(lbl, coordinates="rc")
    else: yield from measure.regionprops(lbl)

def _shape_art(r, h, w):
    minr, minc, maxr, maxc = r.bbox
    if minr == 0 or minc == 0 or maxr == h or maxc == w: return True
    return r.eccentricity > 0.97 and r.axis_major_length / (r.axis_minor_length + 1e-3) > 6

def filter_regions(lbl, bf):
    if lbl.max() == 0: return lbl
    keep = np.zeros(lbl.max() + 1, bool); h, w = lbl.shape
    for reg in _iter_regions(lbl):
        if reg.area < 50 or _shape_art(reg, h, w): continue
        if bf is not None and bf[tuple(reg.coords.T)].var() < 50: continue
        keep[reg.label] = True
    return lbl * keep[lbl]

def _do_segmentation(g, r, bf, A):
    if A.seg_mode == "green":
        return filter_regions(segment(g, A.min_size), bf)
    if A.seg_mode == "fallback":
        lbl = filter_regions(segment(g, A.min_size), bf)
        return lbl if lbl.max() > 0 else filter_regions(segment(r, A.min_size), bf)
    # union
    lbl = morphology.label((segment(g, A.min_size) > 0) | (segment(r, A.min_size) > 0))
    return filter_regions(lbl, bf)

def _fast_props(g, r, labels, g_bg, r_bg, g_sd, r_sd):
    flat = labels.ravel(); mask = flat > 0; ids = flat[mask]
    area = np.bincount(ids, minlength=labels.max() + 1).astype(np.float32)[1:]
    g_sum = np.bincount(ids, weights=g.ravel()[mask], minlength=labels.max() + 1)[1:]
    r_sum = np.bincount(ids, weights=r.ravel()[mask], minlength=labels.max() + 1)[1:]
    g_corr, r_corr = g_sum - area * g_bg, r_sum - area * r_bg
    g_z = g_corr / (g_sd * area + 1e-8); r_z = r_corr / (r_sd * area + 1e-8)
    ratio = np.maximum(g_corr / (r_corr + 1e-6), 1e-6)
    return area, g_z, r_z, ratio

def auto_valley_safe(ratios, fallback):
    clean = ratios[np.isfinite(ratios) & (ratios > 0)]
    if clean.size < 30: return fallback
    log = np.log2(clean + 1e-12); hist, edges = np.histogram(log, 128)
    peaks, _ = find_peaks(hist, prominence=hist.max() * 0.05)
    if len(peaks) < 2: return fallback
    p1, p2 = np.sort(peaks[np.argsort(hist[peaks])][-2:])
    valley = np.argmin(hist[p1:p2]) + p1
    return 2 ** ((edges[valley] + edges[valley + 1]) / 2)

# ─────────── per-field analysis ─────────────────────────────────────
def analyse(base: Path, A):
    try: g_raw, r_raw, bf = load_set(base, A)
    except FileNotFoundError as e: _log.error(e); return {}, []
    g, r = white_tophat(g_raw, A.top_hat), white_tophat(r_raw, A.top_hat)
    labels = _do_segmentation(g, r, bf, A)
    if labels.max() == 0:
        _log.warning("%s – no nuclei", base.name); return {}, []
    bg = labels == 0
    g_bg, g_sd = g[bg].mean(), g[bg].std() + 1e-6
    r_bg, r_sd = r[bg].mean(), r[bg].std() + 1e-6

    area, g_z, r_z, ratio = _fast_props(g, r, labels, g_bg, r_bg, g_sd, r_sd)
    idx = np.arange(1, labels.max() + 1); n = idx.size
    if n == 0: return {}, []

    # ----- EB split (uses AO-positive pool, then capped) -----------------
    pool = r_z[g_z > A.g_z_thr_min] if A.seg_mode == "union" else r_z
    thr_r_z = filters.threshold_otsu(pool) if pool.size > 1 else 3.0
    thr_r_z = min(thr_r_z, A.max_r_z_thr)   # safety-cap

    thr_ratio = A.ratio if A.ratio > 0 else auto_valley_safe(ratio, A.ratio_fallback)
    eb_neg = r_z <= thr_r_z
    if eb_neg.sum() >= 50:
        try:
            means = np.sort(GaussianMixture(2, random_state=0)
                            .fit(np.log2(ratio[eb_neg] + 1e-12).reshape(-1, 1))
                            .means_.ravel())
            thr_ratio = max(thr_ratio, 2 ** (means.mean()))
        except Exception as e: _log.debug("GMM fail %s", e)

    med_area = np.median(area)
    cls = np.full(n, "", object)
    early = (g_z > A.g_z_early) & (ratio >= thr_ratio * A.ratio_mult_early)
    cls[eb_neg & early] = "Early Apoptosis"
    cls[eb_neg & ~early] = "Live"
    eb_pos = ~eb_neg
    red_dom = (r_z > A.r_z_nec) | (ratio < A.ratio_factor_nec * thr_ratio)
    cls[eb_pos & red_dom] = "Necrotic"
    late = eb_pos & ~red_dom & (area < A.area_factor_late * med_area)
    cls[late] = "Late Apoptosis"
    cls[(eb_pos) & (cls == "")] = "Necrotic"

    cats = ("Live", "Early Apoptosis", "Late Apoptosis", "Necrotic")
    counts = {k: int((cls == k).sum()) for k in cats}
    summary = {"Image": base.name, "Total": n,
               "thr_ratio": thr_ratio, "thr_r_z": thr_r_z, **counts,
               **{f"{k}_pct": v / n * 100 for k, v in counts.items()}}
    rows = [{"Image": base.name, "Label": int(l), "Area_px": float(a),
             "g_z": float(gz), "r_z": float(rz), "ratio": float(rt), "Class": c}
            for l, a, gz, rz, rt, c in zip(idx, area, g_z, r_z, ratio, cls)]

    # ----- optional overlays / masks (unchanged) ------------------------
    if A.overlay or A.masks:
        qc_dir = A.output.with_suffix("").parent / "qc"; qc_dir.mkdir(parents=True, exist_ok=True)
        if A.overlay and min(g_raw.shape) >= 32:
            base_img = bf if (bf is not None and A.qc) else g_raw
            rgb = np.dstack([util.img_as_ubyte(exposure.rescale_intensity(base_img, out_range=(0, 1)))] * 3)
            for y, x in np.column_stack(np.where(segmentation.find_boundaries(labels))):
                rgb[y, x] = (255, 0, 0)
            io.imsave(qc_dir / f"{base.name}_overlay.png", rgb, check_contrast=False)
        if A.masks:
            tiff.imwrite(qc_dir / f"{base.name}_mask.tif", labels.astype(np.uint16),
                         photometric="minisblack")
    return summary, rows

# ─────────── main driver ─────────────────────────────────────────────
def main():
    A = parse_args(); bases = discover_bases(A.input_dir)
    if not bases: raise SystemExit("No AO/EB image sets found")
    _log.info("Processing %d field(s) [seg_mode=%s]", len(bases), A.seg_mode)
    par = Parallel(n_jobs=max(1, min(A.workers, len(bases))), prefer="threads")
    summaries, cells = [], []
    for s, c in tqdm(par(delayed(analyse)(b, A) for b in bases), total=len(bases)):
        if s: summaries.append(s); cells.extend(c)
    if not summaries: _log.error("Nothing quantified – abort"); return
    A.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).sort_values("Image").to_csv(A.output, index=False)
    pd.DataFrame(cells).to_csv(A.output.with_name(f"{A.output.stem}_cells.csv"), index=False)
    _log.info("Summary saved → %s", A.output)
    meta = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in vars(A).items()}
    meta["n_fields"] = len(summaries)
    A.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()

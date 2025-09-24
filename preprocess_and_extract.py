# preprocess_and_extract.py
"""
Corrected and robust FITS -> features extractor.
Saves: features.npy, labels.npy, files_labels.csv
Run: python preprocess_and_extract.py
"""
import os
import glob
import time
import numpy as np
import pandas as pd
from astropy.io import fits
import cv2
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops, label
from scipy import ndimage as ndi

# Config
IMG_DIR = "sdss_data/images"
OUT_FEATURES = "features.npy"
OUT_LABELS = "labels.npy"
OUT_CSV = "files_labels.csv"
PROGRESS_EVERY = 25  # print progress every N files

# --- I/O helpers ---
def load_fits(fname):
    """Load FITS and return 2D float image with NaNs replaced."""
    hdul = fits.open(fname, memmap=False)
    data = hdul[0].data.astype(float)
    hdul.close()
    # If multi-dim, take the first 2D plane
    if data.ndim > 2:
        data = data[0]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

# --- preprocessing / segmentation ---
def background_subtract(img, boxsize=50):
    """Simple median background subtraction."""
    med = ndi.median_filter(img, size=boxsize)
    img_bs = img - med
    return img_bs

def segment_object(img):
    """
    Segment the brightest object using adaptive thresholding and morphological cleaning.
    Returns a 0/255 uint8 mask of the largest component or None if segmentation fails.
    """
    img_norm = img.copy()
    p1, p99 = np.percentile(img_norm, (1, 99))
    if p99 - p1 > 0:
        img_norm = (img_norm - p1) / (p99 - p1)
    img_norm = np.clip(img_norm, 0, 1)
    img_u8 = (img_norm * 255).astype("uint8")

    # Try adaptive threshold (fallback to Otsu)
    try:
        th = cv2.adaptiveThreshold(img_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, -5)
    except Exception:
        _, th = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Label connected components, choose largest
    lab = label(th // 255)
    props = regionprops(lab)
    if not props:
        return None
    areas = [p.area for p in props]
    largest = props[int(np.argmax(areas))]
    mask = (lab == largest.label).astype("uint8") * 255
    return mask

# --- feature extraction ---
def extract_features_from_image(img, mask):
    """
    img: original float 2D (background-subtracted)
    mask: uint8 0/255 mask for the object
    returns: 1D numpy array of features or None
    """
    lab = label(mask // 255)
    props = regionprops(lab, intensity_image=img)
    if not props:
        return None
    p = props[0]

    area = float(p.area)
    minr, minc, maxr, maxc = p.bbox
    height = float(maxr - minr)
    width = float(maxc - minc)
    aspect = float(width / height) if height > 0 else 0.0
    eccentricity = float(p.eccentricity)
    solidity = float(p.solidity)
    extent = float(p.extent)
    mean_int = float(p.mean_intensity) if p.mean_intensity is not None else 0.0
    max_int = float(p.max_intensity) if p.max_intensity is not None else 0.0
    min_int = float(p.min_intensity) if p.min_intensity is not None else 0.0

    # Hu moments from contour (log-scaled)
    cnts, _ = cv2.findContours((mask > 0).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hu = np.zeros(7, dtype=float)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        moments = cv2.moments(c)
        hu_raw = cv2.HuMoments(moments).flatten()
        # log-transform for numeric stability:
        hu = -np.sign(hu_raw) * np.log10(np.abs(hu_raw) + 1e-40)

    # Texture: LBP on uint8 crop with fixed-length histogram (10 bins)
    crop = img[minr:maxr, minc:maxc]
    if crop.size == 0:
        lbp_hist = np.zeros(10, dtype=float)
    else:
        p1c, p99c = np.percentile(crop, (1, 99))
        if p99c - p1c > 0:
            crop_u8 = np.clip((crop - p1c) / (p99c - p1c), 0, 1)
        else:
            crop_u8 = np.clip(crop - p1c, 0, 1)
        crop_u8 = (crop_u8 * 255).astype("uint8")
        lbp = local_binary_pattern(crop_u8, P=8, R=1, method="uniform")
        nbins = 10
        hist, _ = np.histogram(lbp.ravel(), bins=nbins, range=(0, nbins))
        hist = hist.astype("float")
        if hist.sum() > 0:
            hist /= hist.sum()
        lbp_hist = hist

    # Radial profile: mean intensity in concentric rings (8 bins)
    cy, cx = p.centroid
    yy, xx = np.indices(img.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = max(img.shape) / 2.0
    nbins = 8
    bins = np.linspace(0, rmax, nbins + 1)
    radial = []
    for i in range(nbins):
        mask_ring = (r >= bins[i]) & (r < bins[i + 1]) & (mask > 0)
        vals = img[mask_ring]
        radial.append(float(vals.mean()) if vals.size > 0 else 0.0)
    radial = np.array(radial, dtype=float)

    feats = np.concatenate([
        [area, width, height, aspect, eccentricity, solidity, extent, mean_int, max_int, min_int],
        hu,
        lbp_hist,
        radial
    ]).astype(float)
    return feats

# --- main processing loop ---
def main():
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.fits")))
    if not files:
        print("No FITS files found in", IMG_DIR)
        return

    features = []
    labels = []
    rows = []
    start_time = time.time()

    for idx, f in enumerate(files):
        if idx % PROGRESS_EVERY == 0:
            print(f"[{idx}/{len(files)}] Processing... ({time.strftime('%H:%M:%S')})")
        fname = os.path.basename(f)
        print(" ->", fname)
        label_str = fname.split("_")[0].upper()  # expects GALAXY_123.fits etc.

        try:
            img = load_fits(f)
        except Exception as e:
            print("   [ERROR] Failed to load:", fname, "->", repr(e))
            continue

        try:
            img_bs = background_subtract(img, boxsize=40)
            mask = segment_object(img_bs)
            if mask is None:
                print("   [WARN] Segmentation failed for", fname)
                continue

            feats = extract_features_from_image(img_bs, mask)
            if feats is None:
                print("   [WARN] Feature extraction failed for", fname)
                continue

            features.append(feats)
            labels.append(label_str)
            rows.append({"file": fname, "label": label_str})
        except Exception as e:
            print("   [ERROR] Processing failed for", fname, "->", repr(e))
            continue

    elapsed = time.time() - start_time
    if not features:
        print("No features extracted. Exiting.")
        return

    X = np.vstack(features)
    y = np.array(labels)
    np.save(OUT_FEATURES, X)
    np.save(OUT_LABELS, y)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"Saved {OUT_FEATURES}, {OUT_LABELS}, {OUT_CSV}")
    print("Feature matrix shape:", X.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))
    print("Elapsed time: {:.1f}s".format(elapsed))

if __name__ == "__main__":
    main()

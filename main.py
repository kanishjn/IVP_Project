from astroquery.sdss import SDSS
from astropy import coordinates as coords
import numpy as np
import pandas as pd
import os
import time
from astropy.table import vstack

# Output folders
os.makedirs("sdss_data/images", exist_ok=True)

all_results = []
counter = {"STAR": 0, "QSO": 0, "GALAXY": 0}
target_per_class = 2
radius = "3m"
pause_time = 2

print("📡 Collecting 2 STAR, 2 QSO, and 2 GALAXY images...")

while any(c < target_per_class for c in counter.values()):
    ra = np.random.uniform(0, 360)
    dec = np.random.uniform(-10, 70)
    pos = coords.SkyCoord(f"{ra}d {dec}d", frame="icrs")
    print(f"➡️ Querying RA={ra:.2f}, Dec={dec:.2f}")

    try:
        results = SDSS.query_region(
            pos,
            spectro=True,
            radius=radius,
            fields=["ra", "dec", "class", "z", "specobjid", "run", "rerun", "camcol", "field"]
        )
    except Exception as e:
        print(f"⚠️ Query failed: {e}")
        continue

    if results is None or len(results) == 0:
        continue

    # Filter only STAR, GALAXY, QSO
    filtered = results[np.isin(results["class"], ["STAR", "GALAXY", "QSO"])]
    if len(filtered) == 0:
        continue

    all_results.append(filtered)

    try:
        images = SDSS.get_images(matches=filtered, band="r")
    except Exception as e:
        print(f"⚠️ Image download failed: {e}")
        continue

    if images is None:
        continue

    for row, img in zip(filtered, images):
        if img is None:
            continue

        label = row["class"]
        # only save if we still need this class
        if counter[label] < target_per_class:
            try:
                hdu = img[0]
                fname = f"sdss_data/images/{label}_{counter[label]+1}.fits"
                hdu.writeto(fname, overwrite=True)
                counter[label] += 1
                print(f"✅ Saved {label} ({counter[label]}/{target_per_class})")
            except Exception as e:
                print(f"⚠️ Could not save {label}: {e}")

        # stop early if all targets are met
        if all(c == target_per_class for c in counter.values()):
            break

    time.sleep(pause_time)

# Save metadata
if all_results:
    combined = vstack(all_results)
    df = combined.to_pandas()
    df.to_csv("sdss_data/metadata.csv", index=False)
    print(f"🎉 Finished! Saved {sum(counter.values())} images "
          f"({counter['STAR']} STAR, {counter['QSO']} QSO, {counter['GALAXY']} GALAXY)")
else:
    print("❌ No objects found")

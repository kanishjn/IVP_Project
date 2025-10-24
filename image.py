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
counter = 0
target_images = 1000   # how many images you want
radius = "3m"          # search radius
pause_time = 2         # polite delay

print(f"📡 Collecting at least {target_images} images...")

while counter < target_images:
    # Pick a random sky position
    ra = np.random.uniform(0, 360)
    dec = np.random.uniform(-10, 70)  # avoid poles
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

    # Filter for STAR, GALAXY, QSO
    filtered = results[np.isin(results["class"], ["STAR", "GALAXY", "QSO"])]
    if len(filtered) == 0:
        continue

    all_results.append(filtered)

    # Download images
    try:
        images = SDSS.get_images(matches=filtered, band="r")
    except Exception as e:
        print(f"⚠️ Image download failed: {e}")
        continue

    if images is None:
        continue

    for row, img in zip(filtered, images):
        if img is None or counter >= target_images:
            continue
        try:
            hdu = img[0]
            label = row["class"]
            fname = f"sdss_data/images/{label}_{counter}.fits"
            hdu.writeto(fname, overwrite=True)
            counter += 1
            if counter % 100 == 0:
                print(f"✅ Saved {counter} images so far...")
        except Exception as e:
            print(f"⚠️ Could not save image {counter}: {e}")

    # polite pause
    time.sleep(pause_time)

# Save metadata
if all_results:
    combined = vstack(all_results)
    df = combined.to_pandas()
    df.to_csv("sdss_data/metadata.csv", index=False)
    print(f"🎉 Finished! Saved {counter} images and {len(df)} metadata rows")
else:
    print("❌ No objects found")
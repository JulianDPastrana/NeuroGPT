#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import mne

SUBJECTS = [f"A{str(i).zfill(2)}" for i in range(1, 10)]
SESSIONS = ["T", "E"]

RAW_DIR = Path("/home/usuarioutp/Documents/NeuroGPT/bci2a_egg_npz/raw_gdf")
OUT_DIR = Path("/home/usuarioutp/Documents/NeuroGPT/bci2a_egg_npz")


def convert_one(gdf_path: Path, out_dir: Path):
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])  # 250 Hz expected

    # Get data and store as (samples, channels)
    data = raw.get_data()  # (channels, samples)
    s = data.T  # (samples, channels)

    # Build event arrays from annotations (to keep durations)
    ann = raw.annotations
    desc = ann.description
    onset = ann.onset
    dur = ann.duration

    # Build mapping from description to numeric code if possible
    # mne.events_from_annotations returns a map when possible
    _, event_id = mne.events_from_annotations(raw)

    etyp_list = []
    epos_list = []
    edur_list = []

    for d, o, u in zip(desc, onset, dur):
        code = None
        # Convert numeric strings like '768', '769', etc.
        try:
            code = int(d)
        except Exception:
            # Try mapped event id (e.g., 'Stimulus/768')
            if d in event_id:
                code = int(event_id[d])
        if code is None:
            # Skip unrecognized annotations
            continue
        start_samp = int(round(o * sfreq))
        dur_samp = int(round(u * sfreq))
        etyp_list.append(code)
        epos_list.append(start_samp)
        edur_list.append(dur_samp)

    if len(etyp_list) == 0:
        print(f"[WARN] No recognizable events in {gdf_path.name}. Skipping file.")
        return False

    # Shape arrays as (N, 1) to match loader expectations (it transposes to (1, N))
    etyp = np.array(etyp_list, dtype=np.int32).reshape(-1, 1)
    epos = np.array(epos_list, dtype=np.int32).reshape(-1, 1)
    edur = np.array(edur_list, dtype=np.int32).reshape(-1, 1)
    artifacts = np.zeros_like(epos, dtype=np.int32)

    out_path = out_dir / (gdf_path.stem + ".npz")
    np.savez(out_path, s=s, etyp=etyp, epos=epos, edur=edur, artifacts=artifacts)
    print(f"[OK] Wrote {out_path} with s={s.shape}, events={len(etyp_list)}")
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    total = 0
    for subj in SUBJECTS:
        for sess in SESSIONS:
            gdf_path = RAW_DIR / f"{subj}{sess}.gdf"
            total += 1
            if not gdf_path.exists():
                print(f"[MISS] {gdf_path} not found")
                continue
            ok = convert_one(gdf_path, OUT_DIR)
            success += int(bool(ok))
    print(f"[DONE] Converted {success}/{total} files")


if __name__ == "__main__":
    main()

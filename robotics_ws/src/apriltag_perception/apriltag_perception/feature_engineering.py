"""
feature_engineering.py  —  Phase 2: Raw landmark → joint angles
================================================================
Loads every CSV from ~/gesture_data/, computes geometric joint
angles from the (x, y, z) landmark positions, and writes a new
features_TIMESTAMP.csv ready for LSTM training.

Run standalone:
    python3 feature_engineering.py

Output columns:
    timestamp, label, session_id,
    thumb_curl, idx_curl, mid_curl, rng_curl, pky_curl,
    idx_mid_splay, mid_rng_splay, rng_pky_splay,
    elbow_angle
"""

import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
GESTURE_DATA_DIR = os.path.expanduser("~/gesture_data")
TIMESTAMP        = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH      = os.path.join(GESTURE_DATA_DIR, f"features_{TIMESTAMP}.csv")

# ── Output column order ───────────────────────────────────────────────────────
FEATURE_COLS = [
    "thumb_curl",
    "idx_curl",
    "mid_curl",
    "rng_curl",
    "pky_curl",
    "idx_mid_splay",
    "mid_rng_splay",
    "rng_pky_splay",
    "elbow_angle",
]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def pt(row: pd.Series, landmark: str) -> np.ndarray:
    """
    Pull the (x, y, z) position of one landmark out of a DataFrame row.

    landmark must match the CSV prefix exactly, e.g. 'IDX_MCP', 'shoulder'.
    Returns a 1-D numpy array [x, y, z].
    """
    return np.array([row[f"{landmark}_x"],
                     row[f"{landmark}_y"],
                     row[f"{landmark}_z"]], dtype=float)


def angle_at_vertex(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """
    Return the angle (in degrees) formed at 'vertex' by the two bones
    vertex→a  and  vertex→b.

    How it works:
      1. Build two vectors pointing AWAY from the vertex toward a and b.
      2. Normalise them (make them unit length) so the dot product gives
         the cosine of the angle between them (cos θ = u·v for unit vectors).
      3. Clamp the value to [-1, 1] before arccos to avoid NaN from
         tiny floating-point overshoots.
      4. Convert from radians to degrees.

    Returns 0.0 if either bone has zero length (degenerate pose).
    """
    va = a - vertex          # vector from vertex to point a
    vb = b - vertex          # vector from vertex to point b

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0           # can't compute angle for a collapsed bone

    va = va / norm_a         # unit vector
    vb = vb / norm_b         # unit vector

    cos_theta = np.dot(va, vb)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)   # guard against float errors

    return float(np.degrees(np.arccos(cos_theta)))


# ── Per-row feature extraction ────────────────────────────────────────────────

def compute_features(row: pd.Series) -> dict:
    """
    Compute all joint angles for a single frame (one CSV row).

    Curl angles
    -----------
    A 'curl' angle is measured at the MIDDLE joint of a three-joint chain.
    0° = the finger is perfectly straight.
    ~90° = the finger is bent into a fist.

    Example for the index finger:
        IDX_MCP (knuckle) is the vertex.
        Bone 1: WRIST → IDX_MCP  (palm bone pointing to the knuckle)
        Bone 2: IDX_MCP → IDX_PIP (proximal finger bone)
    We measure the angle between those two bones at the MCP knuckle.

    Splay angles
    ------------
    A 'splay' angle is how far apart two neighbouring fingers are.
    Both vectors originate at WRIST and point toward the respective MCP knuckles.
    0° = fingers pressed together, larger = fingers spread apart.

    Arm elbow angle
    ---------------
    Same dot-product method: shoulder→elbow→wrist chain.
    ~180° = arm fully extended, ~90° = elbow bent to a right angle.
    """

    # ── Finger curl angles ────────────────────────────────────────
    # Each angle is at the MIDDLE joint of the 3-point chain shown.

    # Thumb: THUMB_CMC → THUMB_MCP → THUMB_IP
    thumb_curl = angle_at_vertex(
        pt(row, "THUMB_CMC"),   # proximal side
        pt(row, "THUMB_MCP"),   # vertex (the angle is HERE)
        pt(row, "THUMB_IP"),    # distal side
    )

    # Index: IDX_MCP → IDX_PIP → IDX_DIP
    idx_curl = angle_at_vertex(
        pt(row, "IDX_MCP"),
        pt(row, "IDX_PIP"),
        pt(row, "IDX_DIP"),
    )

    # Middle: MID_MCP → MID_PIP → MID_DIP
    mid_curl = angle_at_vertex(
        pt(row, "MID_MCP"),
        pt(row, "MID_PIP"),
        pt(row, "MID_DIP"),
    )

    # Ring: RNG_MCP → RNG_PIP → RNG_DIP
    rng_curl = angle_at_vertex(
        pt(row, "RNG_MCP"),
        pt(row, "RNG_PIP"),
        pt(row, "RNG_DIP"),
    )

    # Pinky: PKY_MCP → PKY_PIP → PKY_DIP
    pky_curl = angle_at_vertex(
        pt(row, "PKY_MCP"),
        pt(row, "PKY_PIP"),
        pt(row, "PKY_DIP"),
    )

    # ── Finger splay angles ───────────────────────────────────────
    # Origin is the WRIST; vectors point toward each finger's MCP knuckle.

    wrist = pt(row, "WRIST")

    # Index vs Middle splay
    idx_mid_splay = angle_at_vertex(
        pt(row, "IDX_MCP"),   # tip of one splay ray
        wrist,                 # vertex (origin of both rays)
        pt(row, "MID_MCP"),   # tip of the other splay ray
    )

    # Middle vs Ring splay
    mid_rng_splay = angle_at_vertex(
        pt(row, "MID_MCP"),
        wrist,
        pt(row, "RNG_MCP"),
    )

    # Ring vs Pinky splay
    rng_pky_splay = angle_at_vertex(
        pt(row, "RNG_MCP"),
        wrist,
        pt(row, "PKY_MCP"),
    )

    # ── Elbow angle ───────────────────────────────────────────────
    # Chain: shoulder → elbow → wrist (arm joints, not hand WRIST landmark)
    elbow_angle = angle_at_vertex(
        pt(row, "shoulder"),   # proximal side
        pt(row, "elbow"),      # vertex
        pt(row, "wrist"),      # distal side  (arm wrist, lower-case)
    )

    return {
        "thumb_curl":    round(thumb_curl,    4),
        "idx_curl":      round(idx_curl,      4),
        "mid_curl":      round(mid_curl,      4),
        "rng_curl":      round(rng_curl,      4),
        "pky_curl":      round(pky_curl,      4),
        "idx_mid_splay": round(idx_mid_splay, 4),
        "mid_rng_splay": round(mid_rng_splay, 4),
        "rng_pky_splay": round(rng_pky_splay, 4),
        "elbow_angle":   round(elbow_angle,   4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load all CSVs ──────────────────────────────────────────
    pattern = os.path.join(GESTURE_DATA_DIR, "gestures_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No gesture CSVs found in {GESTURE_DATA_DIR}")
        return

    print(f"Loading {len(csv_files)} CSV file(s)...")
    frames = [pd.read_csv(f) for f in csv_files]
    raw = pd.concat(frames, ignore_index=True)
    print(f"  Total rows loaded: {len(raw)}")

    # ── 2. Compute features row-by-row ────────────────────────────
    print("Computing joint angles...")
    feature_rows = [compute_features(row) for _, row in raw.iterrows()]
    features_df  = pd.DataFrame(feature_rows, columns=FEATURE_COLS)

    # ── 3. Assemble output dataframe ──────────────────────────────
    out = pd.concat(
        [raw[["timestamp", "label", "session_id"]].reset_index(drop=True),
         features_df],
        axis=1,
    )

    # ── 4. Write output CSV ───────────────────────────────────────
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Columns: {list(out.columns)}\n")

    # ── 5. Summary: row counts per label ─────────────────────────
    print("=" * 40)
    print("  Frames per gesture label")
    print("=" * 40)
    counts = out["label"].value_counts().sort_index()
    for label, count in counts.items():
        bar = "#" * (count // 10)      # rough visual bar (1 # per 10 frames)
        print(f"  {label:<12} {count:>5}  {bar}")
    print("=" * 40)

    total = counts.sum()
    print(f"  {'TOTAL':<12} {total:>5}")

    if len(counts) > 1:
        min_c, max_c = counts.min(), counts.max()
        imbalance = max_c / min_c if min_c > 0 else float("inf")
        if imbalance > 2.0:
            print(
                f"\n  WARNING: largest class is {imbalance:.1f}x the smallest. "
                "Consider collecting more data for under-represented gestures."
            )
        else:
            print("\n  Dataset looks balanced.")
    print()


if __name__ == "__main__":
    main()

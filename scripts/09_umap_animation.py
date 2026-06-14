"""
Step 6: UMAP Animation Across DIV — Unit-Level
===============================================
Animates individual unit UMAP positions across recording dates (DIV).
Uses pre-computed UMAP coordinates (no re-fitting) so the coordinate space
is fixed and frames are directly comparable.

Two animation modes:
  --color condition  → units colored by FA condition (default)
  --color hdbscan    → units colored by HDBSCAN cluster label

Per frame: all units shown in gray background; units recorded at the current
date highlighted in color. Frames sweep through the 7 recording dates.

Usage:
  python 06_umap_animation.py [--fps 1] [--format gif] [--color condition|hdbscan] [--no-trail]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─── Arguments ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--fps",       type=int,   default=1)
parser.add_argument("--format",    type=str,   default="gif", choices=["gif", "mp4"])
parser.add_argument("--color",     type=str,   default="condition",
                    choices=["condition", "hdbscan"],
                    help="Color units by condition (default) or HDBSCAN cluster")
parser.add_argument("--no-trail",  action="store_true",
                    help="Disable faded trail of previous dates")
args = parser.parse_args()

# ─── Paths (from config.py) ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import METADATA_CSV_T3 as METADATA_CSV, results_hdbscan_dir

HDBSCAN_DIR = results_hdbscan_dir(100)
META_PATH  = METADATA_CSV
OUT_DIR    = HDBSCAN_DIR / "animations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
DATE_TO_DIV = {
    241028: 6, 241030: 8, 241104: 13, 241106: 8, 241107: 16,
    241112: 21, 241113: 15, 241115: 24, 241119: 28, 241122: 31,
    241126: 28, 241127: 36,
}

CONDITIONS = [
    "2mg_control", "0mg_deficient", "10mg_excess",
    "20mg_super_excess", "folinic_acid_excess",
]

CONDITION_COLORS = {
    "2mg_control":          "#2196F3",
    "0mg_deficient":        "#F44336",
    "10mg_excess":          "#4CAF50",
    "20mg_super_excess":    "#FF9800",
    "folinic_acid_excess":  "#9C27B0",
}

# ─── Load data ────────────────────────────────────────────────────────────────
print(f"Loading metadata from {META_PATH}")
meta = pd.read_csv(META_PATH)
print(f"  {len(meta)} units")

print(f"Loading pre-computed UMAP coords from {HDBSCAN_DIR}/umap_coords.npy")
coords = np.load(HDBSCAN_DIR / "umap_coords.npy")
print(f"  umap_coords shape: {coords.shape}")

print(f"Loading HDBSCAN labels from {HDBSCAN_DIR}/hdbscan_labels.npy")
hdb_labels = np.load(HDBSCAN_DIR / "hdbscan_labels.npy")
print(f"  unique clusters: {np.unique(hdb_labels)}")

assert len(meta) == len(coords), \
    f"Row mismatch: metadata={len(meta)}, coords={len(coords)}"

meta["umap_x"] = coords[:, 0]
meta["umap_y"] = coords[:, 1]
meta["hdbscan"] = hdb_labels

DATES_SORTED = sorted(meta["date"].unique())

# Fixed axis limits across all frames
X_PAD = (coords[:, 0].max() - coords[:, 0].min()) * 0.04
Y_PAD = (coords[:, 1].max() - coords[:, 1].min()) * 0.04
XLIM  = (coords[:, 0].min() - X_PAD, coords[:, 0].max() + X_PAD)
YLIM  = (coords[:, 1].min() - Y_PAD, coords[:, 1].max() + Y_PAD)

# ─── Color helpers ────────────────────────────────────────────────────────────
if args.color == "hdbscan":
    unique_labels = sorted(hdb_labels[hdb_labels >= 0])
    n_clusters = len(np.unique(unique_labels))
    cluster_cmap = cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    CLUSTER_COLORS = {int(l): cluster_cmap[i] for i, l in
                      enumerate(sorted(set(hdb_labels[hdb_labels >= 0])))}
    CLUSTER_COLORS[-1] = (0.6, 0.6, 0.6, 0.4)   # noise → gray

    def unit_colors(df_sub):
        return [CLUSTER_COLORS.get(int(l), (0.5, 0.5, 0.5, 0.5))
                for l in df_sub["hdbscan"]]

    def legend_handles():
        handles = []
        for l in sorted(set(hdb_labels)):
            label = f"noise" if l == -1 else f"cluster {l}"
            handles.append(mpatches.Patch(color=CLUSTER_COLORS[int(l)], label=label))
        return handles

else:
    def unit_colors(df_sub):
        return [CONDITION_COLORS.get(c, "#888888") for c in df_sub["condition"]]

    def legend_handles():
        return [mpatches.Patch(color=CONDITION_COLORS[c], label=c)
                for c in CONDITIONS if (meta["condition"] == c).any()]


# ─── Render a single frame ────────────────────────────────────────────────────
def render_frame(current_date, trail_dates):
    div = DATE_TO_DIV.get(current_date, "?")
    cur_mask  = meta["date"] == current_date
    n_cur     = cur_mask.sum()

    fig, ax = plt.subplots(figsize=(9, 8))

    # Background: ALL units, tiny gray dots
    ax.scatter(meta["umap_x"], meta["umap_y"],
               s=0.3, c="lightgray", alpha=0.25, linewidths=0,
               zorder=1, rasterized=True)

    # Trail: previous dates, faded, small
    if not args.no_trail and trail_dates:
        n_trail = len(trail_dates)
        for ti, td in enumerate(trail_dates):
            alpha = 0.05 + 0.20 * (ti + 1) / n_trail
            trail_mask = meta["date"] == td
            sub = meta[trail_mask]
            ax.scatter(sub["umap_x"], sub["umap_y"],
                       s=0.8, c=unit_colors(sub), alpha=alpha,
                       linewidths=0, zorder=2, rasterized=True)

    # Current date: highlighted units
    cur = meta[cur_mask]
    ax.scatter(cur["umap_x"], cur["umap_y"],
               s=2.0, c=unit_colors(cur), alpha=0.7,
               linewidths=0, zorder=3, rasterized=True)

    handles = legend_handles()
    ax.legend(handles=handles, fontsize=6, loc="upper right",
              framealpha=0.85, markerscale=2, ncol=1)

    color_label = "FA condition" if args.color == "condition" else "HDBSCAN cluster"
    ax.set_title(
        f"Date: {current_date}  |  DIV: {div}  |  n={n_cur:,} units  |  colored by {color_label}",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        arr = buf.reshape(h, w, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        arr = buf.reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return arr


# ─── Save animation ───────────────────────────────────────────────────────────
def save_animation(frames, out_path, fps):
    ext = out_path.suffix.lower()
    if ext == ".gif":
        try:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                out_path, save_all=True, append_images=pil_frames[1:],
                duration=int(1000 / fps), loop=0,
            )
            print(f"  Saved: {out_path}")
            return
        except ImportError:
            pass
        import matplotlib.animation as animation
        fig, ax = plt.subplots(); ax.axis("off")
        im = ax.imshow(frames[0]); plt.tight_layout(pad=0)
        ani = animation.FuncAnimation(fig, lambda i: [im.set_data(frames[i])],
                                       frames=len(frames), interval=int(1000/fps), blit=True)
        ani.save(out_path, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  Saved: {out_path}")

    elif ext == ".mp4":
        import matplotlib.animation as animation
        fig, ax = plt.subplots(); ax.axis("off")
        im = ax.imshow(frames[0]); plt.tight_layout(pad=0)
        ani = animation.FuncAnimation(fig, lambda i: [im.set_data(frames[i])],
                                       frames=len(frames), interval=int(1000/fps), blit=True)
        try:
            ani.save(out_path, writer="ffmpeg", fps=fps)
            print(f"  Saved: {out_path}")
        except Exception as e:
            gif_path = out_path.with_suffix(".gif")
            print(f"  ffmpeg failed ({e}) — saving GIF to {gif_path}")
            ani.save(gif_path, writer=animation.PillowWriter(fps=fps))
            print(f"  Saved: {gif_path}")
        plt.close(fig)


# ─── Render per-condition animation ──────────────────────────────────────────
def render_frame_condition(focal_cond, current_date, trail_dates):
    div   = DATE_TO_DIV.get(current_date, "?")
    color = CONDITION_COLORS.get(focal_cond, "#888888")

    cur_mask  = (meta["date"] == current_date) & (meta["condition"] == focal_cond)
    n_cur     = cur_mask.sum()

    fig, ax = plt.subplots(figsize=(9, 8))

    # Background: all units, tiny gray
    ax.scatter(meta["umap_x"], meta["umap_y"],
               s=0.3, c="lightgray", alpha=0.25, linewidths=0,
               zorder=1, rasterized=True)

    # Trail: same condition at previous dates, faded
    if not args.no_trail and trail_dates:
        n_trail = len(trail_dates)
        for ti, td in enumerate(trail_dates):
            alpha = 0.05 + 0.20 * (ti + 1) / n_trail
            sub = meta[(meta["date"] == td) & (meta["condition"] == focal_cond)]
            if len(sub):
                ax.scatter(sub["umap_x"], sub["umap_y"],
                           s=0.8, c=color, alpha=alpha,
                           linewidths=0, zorder=2, rasterized=True)

    # Current date, focal condition
    cur = meta[cur_mask]
    if len(cur):
        ax.scatter(cur["umap_x"], cur["umap_y"],
                   s=2.0, c=color, alpha=0.75,
                   linewidths=0, zorder=3, rasterized=True)

    ax.set_title(
        f"{focal_cond}  |  Date: {current_date}  |  DIV: {div}  |  n={n_cur:,} units",
        fontsize=11, fontweight="bold"
    )
    ax.legend(handles=[mpatches.Patch(color=color, label=focal_cond)],
              fontsize=8, loc="upper right", framealpha=0.85)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_xlabel("UMAP 1", fontsize=10); ax.set_ylabel("UMAP 2", fontsize=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()

    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        arr = buf.reshape(h, w, 3)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        arr = buf.reshape(h, w, 4)[..., :3]
    plt.close(fig)
    return arr


# ─── Build and save animations ────────────────────────────────────────────────

if args.color == "condition":
    # One animation per condition
    for cond in CONDITIONS:
        if not (meta["condition"] == cond).any():
            print(f"Skipping {cond} — not found in metadata")
            continue
        print(f"\nRendering condition: {cond}")
        frames = []
        for di, date in enumerate(DATES_SORTED):
            trail = DATES_SORTED[:di] if not args.no_trail else []
            frames.append(render_frame_condition(cond, date, trail))
            n = ((meta["date"] == date) & (meta["condition"] == cond)).sum()
            print(f"  Frame {di+1}/{len(DATES_SORTED)}: date={date} DIV={DATE_TO_DIV.get(date,'?')}  n={n:,} units")
        fname = f"umap_units_{cond}.{args.format}"
        save_animation(frames, OUT_DIR / fname, fps=args.fps)

else:
    # Single animation colored by HDBSCAN
    print(f"\nRendering {len(DATES_SORTED)} frames (hdbscan coloring)…")
    frames = []
    for di, date in enumerate(DATES_SORTED):
        trail = DATES_SORTED[:di] if not args.no_trail else []
        frames.append(render_frame(date, trail))
        n = (meta["date"] == date).sum()
        print(f"  Frame {di+1}/{len(DATES_SORTED)}: date={date} DIV={DATE_TO_DIV.get(date,'?')}  n={n:,} units")
    save_animation(frames, OUT_DIR / f"umap_units_by_hdbscan.{args.format}", fps=args.fps)

print(f"\nDone. Output dir: {OUT_DIR}")


# ─── Multi-panel grid: conditions × DIV ──────────────────────────────────────
print("\nRendering conditions × DIV panel grid…")

n_conds = len(CONDITIONS)
n_dates = len(DATES_SORTED)

fig, axes = plt.subplots(
    n_conds, n_dates,
    figsize=(2.5 * n_dates, 2.5 * n_conds),
    sharex=True, sharey=True,
)

for ci, cond in enumerate(CONDITIONS):
    color = CONDITION_COLORS.get(cond, "#888888")
    for di, date in enumerate(DATES_SORTED):
        ax = axes[ci, di]
        div = DATE_TO_DIV.get(date, "?")

        # Background: all units
        ax.scatter(meta["umap_x"], meta["umap_y"],
                   s=0.1, c="lightgray", alpha=0.2, linewidths=0,
                   rasterized=True)

        # Focal: this condition at this date
        mask = (meta["date"] == date) & (meta["condition"] == cond)
        sub  = meta[mask]
        if len(sub):
            ax.scatter(sub["umap_x"], sub["umap_y"],
                       s=0.4, c=color, alpha=0.6, linewidths=0,
                       rasterized=True)

        ax.set_xlim(XLIM); ax.set_ylim(YLIM)
        ax.set_xticks([]); ax.set_yticks([])

        # Column headers (DIV) on top row
        if ci == 0:
            ax.set_title(f"DIV {div}", fontsize=9, fontweight="bold")

        # Row labels (condition) on left column
        if di == 0:
            ax.set_ylabel(cond.replace("_", "\n"), fontsize=7.5,
                          rotation=0, ha="right", va="center",
                          labelpad=4)

# Add unit count annotations in each panel
for ci, cond in enumerate(CONDITIONS):
    for di, date in enumerate(DATES_SORTED):
        ax = axes[ci, di]
        n = ((meta["date"] == date) & (meta["condition"] == cond)).sum()
        ax.text(0.97, 0.03, f"n={n:,}", transform=ax.transAxes,
                fontsize=5.5, ha="right", va="bottom", color="black",
                alpha=0.7)

plt.suptitle("HIPPIE UMAP — units by condition × DIV (epochs=100)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()

grid_path = HDBSCAN_DIR / "umap_grid_condition_x_div.png"
fig.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {grid_path}")

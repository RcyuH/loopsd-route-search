"""Export route-search tables and a stage-one landscape."""

import csv
import json
from pathlib import Path

import numpy as np

from loopsd_search.routes import Route


def export_report(root: Path) -> None:
    # Matplotlib is only needed to export figures, not to run inference workers.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((Path(manifest["config"]["model"]["path"]) / "config.json").read_text())
    depth = config["num_hidden_layers"]
    rows = []
    for stage in ("stage1", "stage2", "validation"):
        path = root / f"{stage}_summary.json"
        if path.exists():
            results = json.loads(path.read_text())
            baseline = results["standard"]["accuracy"]
            for route, value in results.items():
                rows.append(
                    {
                        "stage": stage,
                        "route": route,
                        "delta_pp": 100 * (value["accuracy"] - baseline),
                        **value,
                    }
                )
    if rows:
        with (root / "route_scores.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    path = root / "stage1_summary.json"
    if not path.exists():
        return
    scores = json.loads(path.read_text())
    landscape = np.full((depth, depth), np.nan)
    for name, value in scores.items():
        route = Route.parse(name)
        if route.start is not None:
            landscape[route.start - 1, route.end - 1] = 100 * (
                value["accuracy"] - scores["standard"]["accuracy"]
            )
    if not np.isfinite(landscape).any():
        return
    bound = max(1.0, float(np.nanmax(np.abs(landscape))))
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e5e7eb")
    im = ax.imshow(
        landscape,
        cmap=cmap,
        vmin=-bound,
        vmax=bound,
        origin="upper",
        extent=(0.5, depth + 0.5, depth + 0.5, 0.5),
    )
    ax.set(
        xlabel="Last repeated layer j (inclusive)",
        ylabel="First repeated layer i (inclusive)",
        title=(
            f"Search stage 1, n={scores['standard']['n']}\n"
            "Accuracy change vs standard; not test performance"
        ),
    )
    fig.colorbar(im, ax=ax, label="Accuracy difference (percentage points)")
    fig.tight_layout()
    fig.savefig(root / "route_landscape.png", dpi=180)
    plt.close(fig)

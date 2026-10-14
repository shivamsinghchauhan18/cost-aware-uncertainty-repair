"""Figure for E2 from results/e2_iris_rows.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = [r for r in json.loads((ROOT / "results" / "e2_iris_rows.json").read_text()) if r["feasible"]]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for off, mode in ((-0.2, "overlap"), (0.2, "containment")):
    ids = range(1, 13)
    vals = [100 * np.mean([r["main_differs"] for r in rows if r["mode"] == mode and r["sid"] == i]) for i in ids]
    ax[0].bar(np.array(ids) + off, vals, width=0.4, label=mode)
    ov = np.array([r["overpay"] for r in rows if r["mode"] == mode and r["overpay"] is not None and np.isfinite(r["overpay"])])
    ax[1].hist(np.log10(np.clip(ov, 1, None)), bins=40, alpha=0.6, label=f"{mode} (median {np.median(ov):.2f}x)")
ax[0].set(xlabel="IRIS-CPS engineering scenario", ylabel="% configurations", xticks=range(1, 13), ylim=(0, 100),
          title="Cheapest repair puts the main burden on a different team")
ax[1].set(xlabel="log10(cost of smallest repair / cost of cheapest repair)", ylabel="configurations",
          title="How much the current method overpays")
for a in ax:
    a.legend(); a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "results" / "e2_iris.png", dpi=160)
print("saved results/e2_iris.png")

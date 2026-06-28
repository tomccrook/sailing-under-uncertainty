"""Export standalone scientific figures from a verified recorded study report.

No fitting, selection or simulator execution. Usage: python scripts/plot_learning.py
--study artifacts/studies/m4 --output artifacts/figures/m4
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sail_lab.experiments import file_hash, write_json
from sail_lab.locked_evaluation import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slice", default="nominal")
    args = parser.parse_args()
    directory = args.study / "evaluation" if (args.study / "lock.json").exists() else args.study
    verify_manifest(directory)
    report = json.loads((directory / "report.json").read_text())
    selected = report.get("slices", {}).get(args.slice, report)
    groups = selected["learned_groups"]
    names = list(groups)
    if not names:
        raise ValueError("no learned groups in selected report")
    args.output.mkdir(parents=True, exist_ok=False)
    means = np.array([groups[n]["paired_mean_difference_s"] for n in names])
    intervals = np.array([groups[n]["hierarchical_paired_mean_95"] for n in names])
    labels = [n.replace("PPO_", "").replace("_", " / ") for n in names]
    stage = "Locked" if report["stage"] == "locked_learning" else "Development validation"
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axis = plt.subplots(figsize=(9, max(3.5, len(names) * 0.8)))
        y = np.arange(len(names))
        axis.axvline(0, color="#617b7e", linewidth=1, linestyle="--")
        axis.hlines(y, intervals[:, 0], intervals[:, 1], color="#14776e", linewidth=2)
        axis.scatter(means, y, color="#14776e", s=45, zorder=3)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Mean cost difference vs tuned belief B1 (s); lower is better")
        axis.set_title(
            f"{stage}: {selected['independent_worlds']} independent worlds\n"
            "Joint 95% intervals resample training seeds and paired whole worlds"
        )
        axis.grid(axis="x", alpha=0.15)
        figure.tight_layout()
        for extension in ("svg", "pdf", "png"):
            figure.savefig(args.output / f"paired-learning.{extension}", dpi=180)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for name, label in zip(names, labels, strict=True):
            group = groups[name]
            axis.scatter(group["mean_cost_s"], group["cvar95_mixture_s"], s=45)
            axis.annotate(
                label,
                (group["mean_cost_s"], group["cvar95_mixture_s"]),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=9,
            )
        baseline = selected["policies"][selected["reference"]]
        axis.scatter(baseline["mean_cost_s"], baseline["cvar95_s"], marker="x", s=70, color="#333")
        axis.annotate(
            "tuned B1",
            (baseline["mean_cost_s"], baseline["cvar95_s"]),
            xytext=(7, -12),
            textcoords="offset points",
        )
        axis.set(
            xlabel="Mean unconditional cost (s)",
            ylabel="CVaR95 cost (s)",
            title=f"{stage}: point estimates, not a fitted efficient frontier",
        )
        axis.grid(alpha=0.15)
        figure.tight_layout()
        for extension in ("svg", "pdf", "png"):
            figure.savefig(args.output / f"mean-tail.{extension}", dpi=180)
        plt.close(figure)
    write_json(
        args.output / "manifest.json",
        {
            "version": 1,
            "source_report_hash": file_hash(directory / "report.json"),
            "stage": report["stage"],
            "slice": args.slice if "slices" in report else "validation",
            "artifacts": {
                p.name: file_hash(p) for p in sorted(args.output.iterdir()) if p.is_file()
            },
        },
    )


if __name__ == "__main__":
    main()

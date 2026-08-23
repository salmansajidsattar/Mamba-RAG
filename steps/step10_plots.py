"""Step 10 — Generate all paper-ready figures.

Outputs (PDF + PNG):
  outputs/figures/scaling_curve.pdf      — encoding time vs. sequence length
  outputs/figures/ablation_heatmap.pdf   — λ × τ_c heatmap (from step 8)
  outputs/figures/results_bar.pdf        — main results bar chart across datasets
"""
import subprocess, sys, os
from pathlib import Path


def run(work_dir: str, skip: bool = False):
    if skip:
        print("  [skip] plots (skip_plots=True)")
        return

    out_dir = Path(work_dir) / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    scripts = [
        # efficiency / scaling curve
        (["experiments/run_efficiency.py",
          "--config", "configs/default.yaml", "--device", "cuda"], 1800),
        # scaling plot
        (["plotting/plot_scaling.py",
          "--results", "outputs/efficiency/efficiency_results.json",
          "--output-dir", "outputs/figures"], 300),
        # main results bar chart
        (["plotting/plot_results.py",
          "--results-dir", "outputs",
          "--output-dir",  "outputs/figures"], 300),
    ]

    for args, timeout in scripts:
        script = args[0]
        if not (Path(work_dir) / script).exists():
            print(f"  ⚠ {script} not found — skipping.")
            continue
        cmd = [sys.executable] + args
        result = subprocess.run(cmd, cwd=work_dir, capture_output=True,
                                text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"  ⚠ {script} failed:")
            print(result.stderr[-400:])
        else:
            print(f"  ✓ {script}")

    # Convert PDFs → PNG for inline viewing
    _convert_pdfs(out_dir)
    print(f"  ✓ All figures → {out_dir}")


def _convert_pdfs(fig_dir: Path):
    """Convert PDF figures to PNG using ImageMagick (if available)."""
    for pdf in sorted(fig_dir.glob("*.pdf")):
        png = pdf.with_suffix(".png")
        if png.exists():
            continue
        r = subprocess.run(
            ["convert", "-density", "150", str(pdf), str(png)],
            capture_output=True,
        )
        if r.returncode != 0:
            # Try pdftoppm as fallback
            subprocess.run(
                ["pdftoppm", "-r", "150", "-png", str(pdf),
                 str(fig_dir / pdf.stem)],
                capture_output=True,
            )


def display_figures(work_dir: str):
    """Display all generated PNG figures using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig_dir = Path(work_dir) / "outputs" / "figures"
    pngs = sorted(fig_dir.glob("*.png"))
    if not pngs:
        print("  No PNG figures found. Run step 10 first.")
        return

    for png in pngs:
        print(f"\n  --- {png.name} ---")
        img = mpimg.imread(str(png))
        plt.figure(figsize=(11, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

"""
run_all.py — recompute the entire paper from raw EEG and build every figure.

    python run_all.py --n_jobs -1 --n_rand 20 --n_boot 1000

Steps:
  1. Part 1 — Spearman/MTF/QG small-world metrics  -> cache/trial_metrics_*.csv
  2. Part 2 — synchrony metrics                    -> cache/trial_synchrony_metrics.csv
  3. Part 3 — representations + TDA + embedding + bootstrap (spearman, mtf, qg)
  4. Render Figures 1, 2, 3 (+ standalone Part-3 panels)
Each step is cached/resumable; delete the relevant cache file to force a redo.
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "part3_embedding_topology"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--n_rand", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--tags", nargs="+", default=["spearman", "mtf", "qg"])
    a = ap.parse_args()

    # 1. Part 1 metrics
    from lib.metrics import compute_trial_metrics
    compute_trial_metrics(n_jobs=a.n_jobs, n_rand=a.n_rand)

    # 2. Part 2 synchrony
    from lib.synchrony import compute_synchrony
    compute_synchrony(n_jobs=a.n_jobs)

    # 2b. Part 4 regional metrics (n_rand smaller; region means average over subjects)
    from lib.regions import compute_region_metrics
    compute_region_metrics(n_jobs=a.n_jobs, n_rand=max(5, a.n_rand // 4))

    # 3. Part 3 per representation
    from part3_embedding_topology.compute_features import run as compute_part3
    from lib.config import CACHE
    for tag in a.tags:
        if not (CACHE / f"tda_{tag}.csv").exists():
            compute_part3(tag=tag, n_jobs=a.n_jobs, n_boot=a.n_boot)

    # 4. figures
    import importlib.util
    def run_script(p, **kw):
        spec = importlib.util.spec_from_file_location(Path(p).stem, p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.main(**kw)
    run_script(ROOT / "part1_topology" / "run_part1.py")
    run_script(ROOT / "part2_synchrony" / "run_part2.py")
    run_script(ROOT / "part4_regional" / "run_part4.py")
    from part3_embedding_topology.run_part3 import assemble, standalone_all
    assemble(a.tags[0]); standalone_all(a.tags[0])
    run_script(ROOT / "part3_embedding_topology" / "run_umap.py")
    run_script(ROOT / "part1_topology" / "run_bootstrap_effects.py")
    run_script(ROOT / "part1_topology" / "run_pairwise.py")
    run_script(ROOT / "part5_enrichment" / "run_network_enrichment.py")
    run_script(ROOT / "part5_enrichment" / "run_embedding_enrichment.py")
    for sc in ["run_centrality_topo.py","run_richclub.py","run_splithalf.py",
               "run_crossrep.py","run_nbs.py","run_figure4_spatial.py","run_sample_size.py","run_gudhi_simplices.py","run_figure4_combined.py"]:
        run_script(ROOT / "part6_advanced" / sc)
    print("\nAll figures in", ROOT / "figures")

if __name__ == "__main__":
    main()

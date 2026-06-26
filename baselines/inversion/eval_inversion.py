"""Round-trip inversion evaluation (Task 5, core).

For each test target descriptor vector d:
  1. Generate params p_hat from a method (NN retrieval, CVAE posterior-mean,
     CVAE best-of-N) conditioned on d.
  2. Decode -> clamp_and_validate to plugin-valid ranges.
  3. render_and_describe(controller, p_hat) -> d_hat (re-rendered, same
     normalisation chain as the dataset).
  4. Primary metric: per-descriptor MAE/RMSE/macro on ||d - d_hat||.
  5. Secondary: audibility rate (fraction passing the audibility check) and
     validity rate (fraction needing no clamping).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from baselines.common.data import build_split_matrices, per_descriptor_metrics
from baselines.common.encoding import ParamCodec
from baselines.common.io import (
    TIMBRAL_KEYS,
    canonical_param_keys,
    load_dataset,
    load_param_spec,
    project_params_to_spec,
)
from baselines.common.render import (
    SYLENTH1_PATH_DEFAULT,
    Sylenth1Controller,
    clamp_and_validate_params,
    render_and_describe,
)
from baselines.inversion.baselines_nn import DescriptorNNRetrieval
from baselines.inversion.model_cvae import BlockLayout, CVAE


_BASELINES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = _BASELINES_ROOT / "artifacts" / "cvae_checkpoint.pt"
DEFAULT_RESULTS = _BASELINES_ROOT / "artifacts" / "results" / "inversion_metrics.csv"
DEFAULT_FIG = _BASELINES_ROOT / "artifacts" / "figures" / "inversion_mae_per_descriptor.png"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--n-test", type=int, default=200,
                    help="number of test presets to evaluate (full test = 1481, ~25 min serial)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--plugin", type=str, default=SYLENTH1_PATH_DEFAULT)
    ap.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--fig", type=Path, default=DEFAULT_FIG)
    ap.add_argument("--methods", nargs="+", default=("nn", "cvae_mean", "cvae_sample"),
                    help="subset of {nn, cvae_mean, cvae_sample}")
    ap.add_argument("--drop-ambiguous", action="store_true")
    return ap.parse_args()


def _project_test_subset(splits_path: Path, dataset_path: Path, n_test: int, seed: int,
                         drop_ambiguous: bool) -> list[dict]:
    """Return up to n_test entries from the TEST split (random subset, seeded)."""
    spec_keys = canonical_param_keys()
    with open(splits_path, "r") as fh:
        splits = json.load(fh)
    test_ids = [pid for pid, v in splits["presets"].items()
                if v["split"] == "test" and (not drop_ambiguous or not v.get("ambiguous"))]
    entries = {e["id"]: e for e in load_dataset(dataset_path)}
    rng = np.random.RandomState(seed)
    if len(test_ids) > n_test:
        idx = rng.choice(len(test_ids), n_test, replace=False)
        test_ids = [test_ids[i] for i in idx]
    return [entries[i] for i in test_ids if i in entries]


def _gen_nn(test_targets: np.ndarray, splits, params_train: list[dict]) -> list[dict]:
    Y_tr = splits["train"][1]
    nn = DescriptorNNRetrieval(Y_tr, params_train)
    return nn.query(test_targets)


def _gen_cvae(test_targets: np.ndarray, codec: ParamCodec, ckpt: dict,
              device: str = "cpu", mode: str = "mean", seed: int = 0) -> list[dict]:
    """Generate CVAE params for each target. ``mode``:
      'mean'   - z=0 (decoder's posterior mean given prior centred at 0).
      'sample' - z ~ N(0, I) (single sample from the prior).
    """
    layout = BlockLayout(**ckpt["layout"])
    model = CVAE(layout, desc_dim=ckpt["desc_dim"], z_dim=ckpt["z_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    d_mean = np.asarray(ckpt["d_mean"], dtype=np.float32)
    d_std = np.asarray(ckpt["d_std"], dtype=np.float32)
    d_std = np.where(d_std > 1e-6, d_std, 1.0)

    D = (np.nan_to_num(test_targets, nan=0.0) - d_mean) / d_std
    D_t = torch.from_numpy(D.astype(np.float32)).to(device)
    with torch.no_grad():
        if mode == "mean":
            z = torch.zeros((D_t.shape[0], ckpt["z_dim"]), device=device)
        elif mode == "sample":
            g = torch.Generator(device=device).manual_seed(seed)
            z = torch.randn((D_t.shape[0], ckpt["z_dim"]), generator=g, device=device)
        else:
            raise ValueError(f"unknown CVAE mode: {mode!r}")
        recon = model.decode(z, D_t)
        x = model.to_codec_vector(recon)
    return [codec.decode(x[i]) for i in range(x.shape[0])]


def _evaluate_method(name: str, gen_params: list[dict], controller, spec,
                     targets: np.ndarray) -> dict:
    """Render each generated preset and return per-descriptor metrics + rates."""
    n = len(gen_params)
    d_hat = np.full_like(targets, np.nan, dtype=np.float32)
    audibility = np.zeros(n, dtype=bool)
    validity = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, p in enumerate(gen_params):
        valid = clamp_and_validate_params(p, spec)
        validity[i] = (len(valid) == len(p))
        try:
            descr = render_and_describe(controller, valid, param_limits=spec)
        except Exception as e:
            print(f"  {name} preset {i} render error: {e}")
            descr = None
        if descr is None:
            continue
        audibility[i] = True
        for j, k in enumerate(TIMBRAL_KEYS):
            v = descr.get(k)
            if v is not None and np.isfinite(v):
                d_hat[i, j] = v
        if (i + 1) % 25 == 0:
            dt = time.time() - t0
            print(f"  {name}: rendered {i+1}/{n} in {dt:.1f}s ({dt/(i+1):.2f}s/preset)")
    metrics = per_descriptor_metrics(targets, d_hat)
    metrics["audibility_rate"] = float(audibility.mean())
    metrics["validity_rate"] = float(validity.mean())
    return {"metrics": metrics, "d_hat": d_hat,
            "audibility": audibility, "validity": validity}


def _save_figure(results: dict, fig_path: Path) -> None:
    import matplotlib.pyplot as plt
    methods = list(results.keys())
    width = 0.8 / max(1, len(methods))
    x = np.arange(len(TIMBRAL_KEYS))
    plt.figure(figsize=(11, 5))
    for i, m in enumerate(methods):
        mae = [results[m]["metrics"][k]["mae"] for k in TIMBRAL_KEYS]
        plt.bar(x + (i - (len(methods) - 1) / 2) * width, mae, width, label=m)
    plt.xticks(x, TIMBRAL_KEYS, rotation=20)
    plt.ylabel("Round-trip test MAE (0-100 units)")
    plt.title("Descriptor inversion: per-descriptor round-trip MAE")
    plt.legend()
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading splits, codec, target test subset...")
    splits, codec = build_split_matrices(drop_ambiguous=args.drop_ambiguous)
    spec = load_param_spec()
    spec_keys = canonical_param_keys(spec)

    splits_path = _BASELINES_ROOT / "artifacts" / "splits.json"
    dataset_path = Path(__file__).resolve().parents[2] / "FINAL_timbral_dataset_audiocommons.json"
    test_entries = _project_test_subset(splits_path, dataset_path, args.n_test, args.seed,
                                        args.drop_ambiguous)
    targets = np.stack([
        np.asarray([(e.get("models") or {}).get(k) for k in TIMBRAL_KEYS], dtype=np.float32)
        for e in test_entries
    ])
    targets = np.where(np.isfinite(targets), targets, np.nan)
    print(f"  evaluating on {len(test_entries)} test presets")

    # Load CVAE checkpoint if any cvae_* method requested.
    ckpt = None
    if any(m.startswith("cvae") for m in args.methods):
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        print(f"  loaded CVAE ckpt: z_dim={ckpt['z_dim']}, encoded_dim={ckpt['layout']['encoded_dim']}")

    # Build train-side artefacts for NN retrieval.
    train_entries = []
    with open(splits_path, "r") as fh:
        splits_data = json.load(fh)
    train_ids = {pid for pid, v in splits_data["presets"].items() if v["split"] == "train"
                 and (not args.drop_ambiguous or not v.get("ambiguous"))}
    by_id = {e["id"]: e for e in load_dataset(dataset_path)}
    for pid in train_ids:
        if pid in by_id:
            train_entries.append(by_id[pid])
    params_train = [project_params_to_spec(e.get("params") or {}, spec_keys) for e in train_entries]
    Y_train = np.stack([
        np.asarray([(e.get("models") or {}).get(k) for k in TIMBRAL_KEYS], dtype=np.float32)
        for e in train_entries
    ])

    print("Generating param candidates per method...")
    gen: dict[str, list[dict]] = {}
    if "nn" in args.methods:
        from baselines.inversion.baselines_nn import DescriptorNNRetrieval
        nn = DescriptorNNRetrieval(Y_train, params_train)
        gen["nn"] = nn.query(targets)
    if "cvae_mean" in args.methods:
        gen["cvae_mean"] = _gen_cvae(targets, codec, ckpt, device=args.device, mode="mean")
    if "cvae_sample" in args.methods:
        gen["cvae_sample"] = _gen_cvae(targets, codec, ckpt, device=args.device,
                                       mode="sample", seed=args.seed)

    print(f"\nLoading Sylenth1 for round-trip rendering: {args.plugin}")
    controller = Sylenth1Controller(args.plugin)

    print("\n--- Round-trip evaluation ---")
    results = {}
    for m in args.methods:
        print(f"\n[{m}]")
        results[m] = _evaluate_method(m, gen[m], controller, spec, targets)

    # Save CSV.
    fieldnames = ["method", "descriptor", "mae", "rmse", "r2", "n",
                  "audibility_rate", "validity_rate"]
    rows = []
    for m, res in results.items():
        for k in TIMBRAL_KEYS + ("macro",):
            row = {"method": m, "descriptor": k, **res["metrics"][k],
                   "audibility_rate": res["metrics"]["audibility_rate"],
                   "validity_rate": res["metrics"]["validity_rate"]}
            row.setdefault("n", "")
            rows.append(row)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out}")
    _save_figure(results, args.fig)
    print(f"Wrote {args.fig}")

    # Console summary.
    print("\n=== Summary (test macro MAE, 0-100 units) ===")
    for m, res in results.items():
        macro = res["metrics"]["macro"]
        rates = (res["metrics"]["audibility_rate"], res["metrics"]["validity_rate"])
        print(f"  {m:>6s}: MAE={macro['mae']:6.3f}  RMSE={macro['rmse']:6.3f}  "
              f"audibility={rates[0]:.2%}  validity={rates[1]:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

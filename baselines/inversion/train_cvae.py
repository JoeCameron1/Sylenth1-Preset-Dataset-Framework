"""Train the CVAE (Task 5).

Standardises descriptors on TRAIN only, anneals beta linearly to its final
value over the first 25% of training, early-stops on validation reconstruction
loss. Saves model + the descriptor standardiser to
``baselines/artifacts/cvae_checkpoint.pt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baselines.common.data import build_split_matrices
from baselines.inversion.model_cvae import BlockLayout, CVAE


_BASELINES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = _BASELINES_ROOT / "artifacts" / "cvae_checkpoint.pt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--beta-anneal-frac", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--drop-ambiguous", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading splits...")
    splits, codec = build_split_matrices(drop_ambiguous=args.drop_ambiguous)
    X_tr, Y_tr, _ = splits["train"]
    X_va, Y_va, _ = splits["val"]
    print(f"  train={X_tr.shape}, val={X_va.shape}")

    # Standardise descriptors on TRAIN only.
    d_mean = Y_tr.mean(axis=0)
    d_std = Y_tr.std(axis=0); d_std = np.where(d_std > 1e-6, d_std, 1.0)
    D_tr = (Y_tr - d_mean) / d_std
    D_va = (Y_va - d_mean) / d_std

    layout = BlockLayout.from_codec(codec)
    device = torch.device(args.device)
    model = CVAE(layout, desc_dim=Y_tr.shape[1], z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    Xtr_t = torch.from_numpy(X_tr.astype(np.float32))
    Dtr_t = torch.from_numpy(D_tr.astype(np.float32))
    Xva_t = torch.from_numpy(X_va.astype(np.float32)).to(device)
    Dva_t = torch.from_numpy(D_va.astype(np.float32)).to(device)
    n_tr = Xtr_t.shape[0]

    best_val = float("inf")
    best_state = None
    stale = 0
    history: list[dict] = []
    anneal_epochs = max(1, int(args.epochs * args.beta_anneal_frac))

    for epoch in range(1, args.epochs + 1):
        # Linear beta anneal 0 -> args.beta over the first anneal_epochs epochs.
        beta = args.beta * min(1.0, (epoch - 1) / anneal_epochs)
        model.train()
        idx = torch.randperm(n_tr)
        tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        for s in range(0, n_tr, args.batch_size):
            j = idx[s:s + args.batch_size]
            xb = Xtr_t[j].to(device)
            db = Dtr_t[j].to(device)
            recon, mu, lv = model(xb, db)
            losses = model.compute_loss(recon, xb, mu, lv, beta=beta)
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            bs = xb.size(0)
            tot["loss"] += float(losses["loss"].item()) * bs
            tot["recon"] += float(losses["recon_loss"].item()) * bs
            tot["kl"] += float(losses["kl"].item()) * bs

        # Validation: use posterior mean (no reparam noise) so the metric is stable.
        model.eval()
        with torch.no_grad():
            mu_v, lv_v = model.encode(Xva_t, Dva_t)
            recon_v = model.decode(mu_v, Dva_t)
            val_losses = model.compute_loss(recon_v, Xva_t, mu_v, lv_v, beta=beta)
        val_recon = float(val_losses["recon_loss"].item())
        history.append({
            "epoch": epoch, "beta": beta,
            "train_loss": tot["loss"] / n_tr,
            "train_recon": tot["recon"] / n_tr,
            "train_kl": tot["kl"] / n_tr,
            "val_recon": val_recon,
        })
        if val_recon < best_val - 1e-4:
            best_val = val_recon
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  beta={beta:.3f}  "
                  f"train_recon={tot['recon']/n_tr:.4f}  val_recon={val_recon:.4f}  "
                  f"kl={tot['kl']/n_tr:.4f}")
        if stale >= args.patience:
            print(f"  early stop at epoch {epoch} (val_recon plateau)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  best val recon = {best_val:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "z_dim": args.z_dim,
        "desc_dim": Y_tr.shape[1],
        "d_mean": d_mean.tolist(),
        "d_std": d_std.tolist(),
        "layout": {
            "numeric_slots": layout.numeric_slots,
            "bool_slots": layout.bool_slots,
            "onehot_ranges": layout.onehot_ranges,
            "encoded_dim": layout.encoded_dim,
        },
        "args": vars(args),
        "history": history,
    }, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

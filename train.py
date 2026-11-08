"""Train MDA-SGT on LiverHccSeg.  Example:
python train.py --data_root /data/LiverHccSeg --manifest manifest.csv --phases pv --out runs/full
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from liverseg.data import SliceDataset, load_fold
from liverseg.engine import evaluate_patients, set_seed, summarize
from liverseg.losses import DiceBCELoss
from liverseg.models import build_model


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--data_root", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--phases", default="pv", help="comma list from pre,art,pv,del (each becomes an input channel)")
    a.add_argument("--tumor_labels", default="2", help="mask label value(s) that mean tumour - CHECK your data")
    a.add_argument("--img_size", type=int, default=256)
    a.add_argument("--epochs", type=int, default=100)
    a.add_argument("--batch_size", type=int, default=16)
    a.add_argument("--lr", type=float, default=1e-4)
    a.add_argument("--wd", type=float, default=1e-4)
    a.add_argument("--neg_ratio", type=float, default=0.2)
    a.add_argument("--n_folds", type=int, default=5)
    a.add_argument("--fold", type=int, default=0)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--num_workers", type=int, default=2, help="use 0 on Windows")
    a.add_argument("--val_every", type=int, default=1)
    a.add_argument("--patience", type=int, default=30)
    a.add_argument("--out", default="runs/exp")
    # model / ablation switches
    a.add_argument("--base", type=int, default=32)
    a.add_argument("--embed_dim", type=int, default=384)
    a.add_argument("--depth", type=int, default=4)
    a.add_argument("--heads", type=int, default=8)
    a.add_argument("--encoder", default="cres2net", choices=["cres2net", "plain"])
    a.add_argument("--skip", default="dense", choices=["dense", "unet"])
    a.add_argument("--lambda_init", type=float, default=0.0)
    a.add_argument("--no_pfwf", action="store_true")
    a.add_argument("--no_transformer", action="store_true")
    a.add_argument("--no_gsa", action="store_true")
    a.add_argument("--no_direction", action="store_true")
    return a.parse_args()


def main():
    args = get_args()
    cfg = vars(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"
    print("device:", dev)

    train_p, val_p, test_p = load_fold(cfg)
    if not train_p or not val_p:
        raise SystemExit("no usable patients - check the manifest, paths and --phases")
    loader = DataLoader(SliceDataset(train_p, args.neg_ratio, augment=True, seed=args.seed),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=True, pin_memory=use_amp)

    model = build_model(cfg).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    crit = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, bad = -1.0, 0
    with open(out / "history.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "train_loss", "val_dice", "seconds"])
        for ep in range(1, args.epochs + 1):
            t0, model_losses = time.time(), []
            model.train()
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, enabled=use_amp):
                    logits = model(x)
                loss = crit(logits.float(), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                model_losses.append(loss.item())
            sched.step()
            vd = float("nan")
            if ep % args.val_every == 0:
                rows, _ = evaluate_patients(model, val_p, dev, surface=False)
                vd = summarize(rows)["dice"]["mean"]
                if vd > best:
                    best, bad = vd, 0
                    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "val_dice": vd}, out / "best.pt")
                else:
                    bad += args.val_every
            tl = sum(model_losses) / max(len(model_losses), 1)
            wr.writerow([ep, f"{tl:.4f}", f"{vd:.4f}", f"{time.time() - t0:.1f}"])
            f.flush()
            print(f"ep {ep:3d}  loss {tl:.4f}  val dice {vd:.4f}  best {best:.4f}")
            if bad >= args.patience:
                print("early stopping")
                break

    ck = torch.load(out / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    if test_p:
        rows, dg = evaluate_patients(model, test_p, dev, surface=True)
        summ = summarize(rows, dg)
        (out / "test_summary.json").write_text(json.dumps(summ, indent=2))
        with open(out / "test_per_patient.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print("TEST:", json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()

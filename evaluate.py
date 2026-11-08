"""Re-evaluate a trained run on its held-out test fold (same split as training).
python evaluate.py --run_dir runs/full [--tta] [--viz 5]"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from liverseg.data import load_fold
from liverseg.engine import evaluate_patients, predict_volume, summarize
from liverseg.models import build_model


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run_dir", required=True)
    a.add_argument("--tta", action="store_true")
    a.add_argument("--viz", type=int, default=0, help="save N overlay PNGs of the largest-tumour slices")
    args = a.parse_args()
    run = Path(args.run_dir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run / "best.pt", map_location=dev)
    cfg = ck["cfg"]
    model = build_model(cfg).to(dev)
    model.load_state_dict(ck["model"])
    _, _, test_p = load_fold(cfg)
    rows, dg = evaluate_patients(model, test_p, dev, surface=True, tta=args.tta)
    summ = summarize(rows, dg)
    print(json.dumps(summ, indent=2))
    (run / "test_summary_eval.json").write_text(json.dumps(summ, indent=2))

    if args.viz:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (run / "viz").mkdir(exist_ok=True)
        for p in test_p[:args.viz]:
            prob = predict_volume(model, p["images"], dev, tta=args.tta).numpy()
            z = int(p["masks"].reshape(len(p["masks"]), -1).sum(1).argmax())
            fig, ax = plt.subplots(1, 3, figsize=(9, 3))
            img = p["images"][z, 0].float().numpy()
            ax[0].imshow(img, cmap="gray"); ax[0].set_title("input")
            ax[1].imshow(p["masks"][z].numpy(), cmap="gray"); ax[1].set_title("ground truth")
            ax[2].imshow(prob[z] > 0.5, cmap="gray"); ax[2].set_title("prediction")
            for x in ax: x.axis("off")
            fig.savefig(run / "viz" / f"{p['id']}_z{z}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()

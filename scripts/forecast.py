import argparse
import json

import _path
import torch

from clima.data.store import Era5Store
from clima.ensemble import EnsembleConfig
from clima.forecast import buenos_aires_report, forecast
from clima.train import load_members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data", default="data/era5_64x32")
    parser.add_argument("--init", default="2020-12-28T00")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--members", type=int, default=50)
    parser.add_argument("--ic-scale", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=30.0)
    parser.add_argument("--out", default="runs/pronostico")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = Era5Store(args.data)
    models = load_members(args.checkpoints, device)
    ens = EnsembleConfig(members=args.members, ic_scale=args.ic_scale)
    t, phys = forecast(models, store, args.init, args.steps, ens, device)
    report = buenos_aires_report(store, t, phys, args.out, args.threshold)
    print(json.dumps({k: report[k] for k in ("init", "members", "prob_above")}, indent=2))


if __name__ == "__main__":
    main()

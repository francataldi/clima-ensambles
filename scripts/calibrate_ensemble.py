import argparse

import _path
import torch

from clima.compare import save_comparison, summary_lines, sweep_ic_scale
from clima.data.store import Era5Store
from clima.ensemble import EnsembleConfig
from clima.plots import plot_strategy_scores
from clima.train import load_members


def main() -> None:
    parser = argparse.ArgumentParser(description="barre el tamano de la perturbacion de condiciones iniciales y mide CRPS y dispersion/error por plazo, sobre los anios de validacion")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data", default="data/era5_64x32")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2019-12-31T18")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--reference", type=float, default=0.5, help="escala contra la que se mide la mejora de CRPS")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--every", type=int, default=12)
    parser.add_argument("--max-inits", type=int, default=None)
    parser.add_argument("--members", type=int, default=30)
    parser.add_argument("--length-scale", type=float, default=6.0)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--variables", nargs="+", default=["z500", "t850", "t2m", "u10"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/calibracion")
    args = parser.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    store = Era5Store(args.data)
    models = load_members(args.checkpoints, device)
    base = EnsembleConfig(members=args.members, ic_scale=args.reference, length_scale=args.length_scale, batch=args.batch, model_noise=all(m.stochastic for m in models))
    sweep = sweep_ic_scale(models, store, args.start, args.end, tuple(args.scales), base, args.steps, args.every, args.max_inits, device, args.seed)
    out = save_comparison(sweep, args.out)
    variables = tuple(v for v in args.variables if v in store.names)
    plot_strategy_scores(sweep, out, variables, title="Cuánto perturbar el arranque", filename="calibracion.png")
    for line in summary_lines(sweep, variables[0] if variables else "z500"):
        print(line)
    print(f"resultados en {out}")


if __name__ == "__main__":
    main()

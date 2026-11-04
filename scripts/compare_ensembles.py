import argparse

import _path
import torch

from clima.compare import compare_strategies, save_comparison, summary_lines
from clima.data.store import Era5Store
from clima.ensemble import STRATEGIES, EnsembleConfig
from clima.plots import plot_strategy_rank_histograms, plot_strategy_scores
from clima.train import load_members


def main() -> None:
    parser = argparse.ArgumentParser(description="compara estrategias de ensamble con CRPS, dispersion/skill e histogramas de rangos por plazo")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data", default="data/era5_64x32")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2020-12-31T18")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument("--max-inits", type=int, default=None)
    parser.add_argument("--members", type=int, default=50)
    parser.add_argument("--ic-scale", type=float, default=0.5)
    parser.add_argument("--length-scale", type=float, default=6.0)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    parser.add_argument("--variables", nargs="+", default=["z500", "t850", "t2m", "u10"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/comparacion")
    args = parser.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    store = Era5Store(args.data)
    models = load_members(args.checkpoints, device)
    base = EnsembleConfig(members=args.members, ic_scale=args.ic_scale, length_scale=args.length_scale, batch=args.batch)
    comparison = compare_strategies(models, store, args.start, args.end, tuple(args.strategies), base, args.steps, args.every, args.max_inits, device, args.seed)
    if not comparison["strategies"]:
        raise SystemExit("ninguna estrategia disponible para este modelo")
    out = save_comparison(comparison, args.out)
    variables = tuple(v for v in args.variables if v in store.names)
    plot_strategy_scores(comparison, out, variables)
    for var in variables:
        plot_strategy_rank_histograms(comparison, out, var)
    for line in summary_lines(comparison, variables[0] if variables else "z500"):
        print(line)
    print(f"resultados en {out}")


if __name__ == "__main__":
    main()

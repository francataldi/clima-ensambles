import argparse

import _path
import torch

from clima.data.store import Era5Store
from clima.ensemble import EnsembleConfig
from clima.evaluate import evaluate, save_report
from clima.plots import plot_event_scores, plot_rank_histograms, plot_reliability, plot_roc, plot_scorecard, plot_skill
from clima.scorecard import save_scorecard, scorecard
from clima.train import load_members


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--mc-dropout", action="store_true")
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--no-events", action="store_true", help="omite Brier, confiabilidad y ROC de eventos extremos")
    parser.add_argument("--out", default="runs/evaluacion")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = Era5Store(args.data)
    models = load_members(args.checkpoints, device)
    ens = EnsembleConfig(args.members, args.ic_scale, args.length_scale, args.mc_dropout, args.batch)
    result = evaluate(models, store, args.start, args.end, args.steps, args.every, ens, args.max_inits, device, event_metrics=not args.no_events)
    out = save_report(result, args.out)
    plot_skill(result, out)
    plot_rank_histograms(result, out)
    card = scorecard(result)
    save_scorecard(card, out)
    plot_scorecard(card, out)
    if "events" in result:
        plot_event_scores(result, out)
        plot_reliability(result, out)
        plot_roc(result, out)
    print(f"resultados en {out}")


if __name__ == "__main__":
    main()

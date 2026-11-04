import argparse
import json
from pathlib import Path

import _path

from clima.plots import plot_scorecard
from clima.scorecard import save_scorecard, scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description="scorecard estilo WeatherBench a partir de un metrics.json de scripts/evaluate.py")
    parser.add_argument("--metrics", default="runs/evaluacion/metrics.json")
    parser.add_argument("--variables", nargs="+", default=None)
    parser.add_argument("--leads", nargs="+", type=int, default=None, help="plazos en horas")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    metrics = Path(args.metrics)
    result = json.loads(metrics.read_text(encoding="utf-8"))
    card = scorecard(result, args.variables, args.leads)
    out = Path(args.out) if args.out else metrics.parent
    print(save_scorecard(card, out))
    print(plot_scorecard(card, out))


if __name__ == "__main__":
    main()

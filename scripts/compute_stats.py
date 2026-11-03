import argparse

import _path

from clima.data.stats import compute_stats
from clima.data.store import Era5Store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/era5_64x32")
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2017-12-31T18")
    args = parser.parse_args()
    stats = compute_stats(Era5Store(args.data), args.start, args.end)
    for name, m, s, d in zip(stats["names"], stats["mean"], stats["std"], stats["diff_std"]):
        print(f"{name:>6}  media {m:12.4f}  desvio {s:10.4f}  desvio 6h {d:10.4f}")


if __name__ == "__main__":
    main()

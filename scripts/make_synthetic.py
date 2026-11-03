import argparse

import _path

from clima.data.convert import convert
from clima.data.synthetic import make_synthetic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sintetico")
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2019-12-31T18")
    args = parser.parse_args()
    ds = make_synthetic(args.start, args.end)
    convert(ds, args.out, int(args.start[:4]), int(args.end[:4]))


if __name__ == "__main__":
    main()

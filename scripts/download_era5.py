import argparse

import _path

from clima.data.convert import WB2_ERA5_64x32, convert, open_wb2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=WB2_ERA5_64x32)
    parser.add_argument("--out", default="data/era5_64x32")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2022)
    args = parser.parse_args()
    ds = open_wb2(args.source)
    print(ds)
    convert(ds, args.out, args.start, args.end)


if __name__ == "__main__":
    main()

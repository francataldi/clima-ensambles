import argparse

import _path

from clima.train import TrainConfig, train, train_members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtx3060.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--until-stage", type=int, default=None, help="entrena hasta esta etapa inclusive (0 es la primera) y se detiene; despues se retoma desde state.pt")
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    if args.fresh:
        cfg.resume = False
    if args.until_stage is not None:
        cfg.until_stage = args.until_stage
    cfg.last_stage()
    if args.seeds:
        print(train_members(cfg, args.seeds))
    else:
        print(train(cfg))


if __name__ == "__main__":
    main()

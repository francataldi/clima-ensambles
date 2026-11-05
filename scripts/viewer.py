import argparse
from pathlib import Path

import _path

from clima.ensemble import STRATEGIES
from clima.viewer import DEFAULT_VARIABLES, VARIABLES, forecast_viewer, rebuild_viewer

ABOUT = (
    "Una red AFNO, de la familia de FourCastNet, aprendió de 28 años de datos reales de la atmósfera (ERA5, 1990-2017) "
    "cómo cambia el clima cada 6 horas, y pronostica los próximos días encadenando pasos de a uno."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="genera un visor HTML autocontenido (un solo archivo) del pronostico por ensamble")
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--data", default="data/era5_64x32")
    parser.add_argument("--init", default="2020-12-28T00")
    parser.add_argument("--steps", type=int, default=40, help="pasos de 6 h a pronosticar (40 = 10 dias)")
    parser.add_argument("--every", type=int, default=2, help="se guarda un cuadro cada tantos pasos de 6 h")
    parser.add_argument("--members", type=int, default=20)
    parser.add_argument("--strategy", default="auto", choices=["auto", *STRATEGIES])
    parser.add_argument("--ic-scale", type=float, default=0.5)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES), choices=list(VARIABLES))
    parser.add_argument("--title", default="Pronóstico por ensamble")
    parser.add_argument("--truth-label", default="ERA5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-gzip", action="store_true")
    parser.add_argument("--repo", default="https://github.com/francataldi/clima-ensambles", help="link al repositorio que aparece en el visor")
    parser.add_argument("--about", default=ABOUT, help="texto que explica el modelo en la presentacion del visor")
    parser.add_argument("--rebuild", default=None, help="visor ya generado: lo vuelve a armar con la plantilla actual sin correr el modelo")
    parser.add_argument("--out", default="runs/visor/visor.html")
    args = parser.parse_args()
    extra = {"repo": args.repo or None, "model_text": args.about or None}
    if args.rebuild:
        html = rebuild_viewer(Path(args.rebuild).read_text(encoding="utf-8"), **extra)
        write(html, Path(args.out))
        return
    if not args.checkpoints:
        parser.error("hace falta --checkpoints o --rebuild")
    import torch

    from clima.data.store import Era5Store
    from clima.ensemble import STRATEGY_LABELS, EnsembleConfig, strategy_available, strategy_config
    from clima.train import load_members

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    store = Era5Store(args.data)
    models = load_members(args.checkpoints, device)
    name = args.strategy
    if name == "auto":
        name = "ci+ruido" if strategy_available("ci+ruido", models) is None else "ci"
    reason = strategy_available(name, models)
    if reason:
        raise SystemExit(f"la estrategia {name} no esta disponible: {reason}")
    ens = strategy_config(name, EnsembleConfig(members=args.members, ic_scale=args.ic_scale, batch=min(args.members, 25)))
    subtitle = f"AFNO · {args.members} miembros · {STRATEGY_LABELS[name]} · grilla {len(store.lon)}×{len(store.lat)}"
    html = forecast_viewer(models, store, args.init, args.steps, args.every, ens, tuple(args.variables), device, args.seed, args.title, subtitle, args.truth_label, not args.no_gzip, extra=extra)
    write(html, Path(args.out))


def write(html: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"visor en {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

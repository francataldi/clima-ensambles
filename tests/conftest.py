import pytest

from clima.data.convert import convert
from clima.data.stats import compute_stats
from clima.data.store import Era5Store
from clima.data.synthetic import make_synthetic


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("era5")
    ds = make_synthetic("2018-01-01", "2018-03-31T18", seed=1)
    convert(ds, root, 2018, 2018, verbose=False)
    s = Era5Store(root)
    compute_stats(s, "2018-01-01", "2018-02-28T18", verbose=False)
    return s

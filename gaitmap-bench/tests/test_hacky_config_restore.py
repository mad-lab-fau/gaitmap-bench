import joblib
from gaitmap_bench import config, set_config
from tpcp.parallel import delayed


def test_config_restore():
    set_config()
    # We just check that this does not throw an error.
    # This indicates that `set_config` was called correctly in the parallel process.
    joblib.Parallel(n_jobs=2)(delayed(config)() for _ in range(2))

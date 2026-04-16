import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
FTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

KPPAX_PATH = Path("KPPax")
sys.path.insert(0, str(KPPAX_PATH))
from stoicm import StoicmSplitSP, parse_kpp_dump
from rate_law import log_rate_law_SP
import model


pollu_C0 = {
    "NO": 0.2,
    "O3": 0.04,
    "HCHO": 0.1,
    "CO": 0.3,
    "ALD": 0.01,
    "SO2": 0.007,
}
pollu_ts = jnp.linspace(0, 60, 20, dtype=FTYPE)

toy_C0 = {
    "NO": 2e7,
    "O3": 5e11,
    "OH": 1e6,
    "TOY": 1e11,
    "HO2": 1e8,
}
toy_ts = jnp.arange(0, 5.1, 0.1, dtype=FTYPE)


def get_scheme(sch_name: str):
    global rates, ro2
    if sch_name == "pollu":
        import schemes.pollu.rates as rates
        import schemes.pollu.ro2 as ro2

        PARSE_CFG_PATH = KPPAX_PATH / "schemes/pollu/parse.yaml"
        C0 = pollu_C0
        ts = pollu_ts
    elif sch_name == "toy":
        import schemes.toy_autoxidation.rates as rates
        import schemes.toy_autoxidation.ro2 as ro2

        PARSE_CFG_PATH = KPPAX_PATH / "schemes/toy_autoxidation/parse.yaml"
        C0 = toy_C0
        ts = toy_ts

    kpp_dump = parse_kpp_dump(PARSE_CFG_PATH)
    stoicm_split = StoicmSplitSP.from_stoicm(kpp_dump["STOICM"])

    y0 = jnp.zeros(kpp_dump["STOICM"].nspec, dtype=FTYPE)
    for k, v in C0.items():
        y0 = y0.at[kpp_dump["SPC_NAMES"].index(k)].set(v)
    # for robertson and pollu, ro2 is redundant
    k_static, ro2_coef = rates.update_rconst(TEMP=jnp.array(288., dtype=FTYPE))

    kinetics = {
        "stoicm": stoicm_split,
        "k_static": k_static,
        "ro2_coef": ro2_coef,
        "update_RO2_fn": ro2.update_RO2,
        "update_rconst_RO2_fn": rates.update_rconst_RO2,
        "rate_law_fn": log_rate_law_SP,
    }

    return kpp_dump, kinetics, ts, y0


def combine_static_ro2(tree):
    combined = jnp.zeros(rates.NREACT)
    combined = combined.at[rates._STATIC_DYN_INDICES].set(
        tree["k_static"][rates._STATIC_DYN_INDICES]
    )
    combined = combined.at[rates._RO2_INDICES].set(tree["ro2_coef"])
    return combined


def get_ys(params, ts, y0):
    return model.solve(params, ts, y0, model.kinetic_ode)


TOY_DATASET_DIR = "dataset1-10/"


def load_toy_dataset(target_spc_names):
    folder = Path(TOY_DATASET_DIR)
    with open(folder / "compounds_data.txt") as f:
        spc_names = f.read().strip().strip("[]").replace("'", "").split(", ")

    data_list = []
    for i in range(10):
        data_list.append(np.loadtxt(folder / f"dataset{i + 1}.csv", delimiter=","))

    dataset = np.stack(data_list, 0)  # [B, spc, time]

    # Reorder speies to match given species order
    name_idx = {name: i for i, name in enumerate(spc_names)}
    reorder = [name_idx[name] for name in target_spc_names]
    dataset = dataset[:, reorder]

    return jnp.transpose(jnp.asarray(dataset), (0, 2, 1))  # [B, time, spc]


def add_normal_noise(a: jax.Array, factor, key):
    return a * (1.0 + jax.random.normal(key, a.shape) * factor)

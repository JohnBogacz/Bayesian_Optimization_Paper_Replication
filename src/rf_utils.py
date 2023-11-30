import pandas as pd
import os, sys

global PATH
PATH = os.getcwd()
PATH = PATH.split(os.path.sep)
if "src" in PATH:
    PATH.remove("src")
PATH = os.path.sep.join(PATH)


def dataset(name: str = None) -> set[str, pd.DataFrame]:
    print(name)
    if (
        name not in ["AgNP", "AutoAM", "Crossed barrel", "P3HT", "Perovskite"]
        and name is not None
    ):
        raise Exception(
            f"Input must be...\n'AgNp'\n'AutoAM'\n'Crossed barrel'\n'P3HT'\n'Perovskite'\n"
        )

    if name is None:
        set_df = {
            "AgNP": pd.read_csv(os.path.join(PATH, "datasets", "AgNP_dataset.csv")),
            "AutoAM": pd.read_csv(os.path.join(PATH, "datasets", "AutoAM_dataset.csv")),
            "Crossed barrel": pd.read_csv(
                os.path.join(PATH, "datasets", "Crossed barrel_dataset.csv")
            ),
            "P3HT": pd.read_csv(os.path.join(PATH, "datasets", "P3HT_dataset.csv")),
            "Perovskite": pd.read_csv(
                os.path.join(PATH, "datasets", "Perovskite_dataset.csv")
            ),
        }
    else:
        set_df = {
            name: pd.read_csv(os.path.join(PATH, "datasets", name + "_dataset.csv"))
        }

    for key, val in set_df.items():
        features = list(val.columns)[:-1]
        obj_metric = list(val.columns)[-1]

        # (A) Set optimization metric to be negative, only 3 need to be flipped
        if key in ["P3HT", "Crossed barrel", "AutoAM"]:
            val[obj_metric] = -val[obj_metric].values

        # (B) Some identical measurements have been made more then once
        # So we are going to take the mean
        val = val.groupby(features)[obj_metric].agg(lambda x: x.unique().mean())
        val = (val.to_frame()).reset_index()

        set_df[key] = val

    return set_df


def rf_param_fixer(rf_params: dict) -> dict:
    pass


if __name__ == "__main__":
    """
    import unittest

    class GenericTests(unittest.TestCase):
        def dataset_test(self):
            pass

    unittest.main()
    """

    # dataset()
    # dataset("Crossed barrel")

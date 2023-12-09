import multiprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import math, os, random, copy, time, itertools, gc, time, warnings


def df_preprocessing(data_name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PATH, "datasets", data_name + "_dataset.csv"))
    # (A) There are multiple of the same experiments, so average them!
    features = list(df.columns)[:-1]
    obj_metric = list(df.columns)[-1]
    df = df.groupby(features)[obj_metric].agg(lambda x: x.unique().mean())
    df = (df.to_frame()).reset_index()
    # (B) Change all NECESSARY df to a minimization problem
    if data_name in ["P3HT", "Crossed barrel", "AutoAM"]:
        # aka AgNP & Perovskite
        df[obj_metric] = -df[obj_metric].values
    return df


class Model:
    def __init__(
        self, seeds, df, df_name, n_ensemble, n_initial, ac_type, ratio, pth
    ) -> None:
        self.metric: float = 0.0
        self.metric_history: list[float] = [0.0]
        self.seeds = seeds
        self.df = df
        self.N = len(df)
        self.df_name = df_name
        self.n_ensemble = n_ensemble
        self.n_initial = n_initial
        self.ac_type = ac_type
        self.lcb_ratio = ratio
        self.pth = pth
        self.identity = "INSERT MODEL NAME"

    def worker(self, tasks, results):
        lock = multiprocessing.Lock()  # IDK if needed
        while True:
            time.sleep(random.randint(0, 10) / 10)
            with lock:
                seed = tasks.get()
                if seed is None:
                    break
                result = self.main(seed)
                results.put(result)

    def run(self):
        NUM_CORES = multiprocessing.cpu_count()
        tasks = multiprocessing.Queue()

        for seed in self.seeds:
            tasks.put(seed)
        for _ in range(NUM_CORES):
            tasks.put(None)

        processes = list()
        manager = multiprocessing.Manager()
        results = manager.Queue()

        for _ in range(NUM_CORES):
            process = multiprocessing.Process(target=self.worker, args=(tasks, results))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        all_together = {
            "index_collection": list(),
            "X_collection": list(),
            "y_collection": list(),
            "TopCount_collection": list(),
            "total_time": list(),
        }
        while not results.empty():
            result = results.get()
            all_together["index_collection"].append(result["index_collection"])
            all_together["X_collection"].append(result["X_collection"])
            all_together["y_collection"].append(result["y_collection"])
            all_together["TopCount_collection"].append(result["TopCount_collection"])
            all_together["total_time"].append(result["total_time"])

        np.savez_compressed(
            os.path.join(self.pth, "results", self.identity),
            index_collection=all_together["index_collection"],
            X_collection=all_together["X_collection"],
            y_collection=all_together["y_collection"],
            TopCount_collection=all_together["TopCount_collection"],
            total_time=all_together["total_time"],
            identity=self.identity,
        )
        return None


class RF(Model):
    def __init__(
        self, seeds, df, df_name, n_ensemble, n_initial, ac_type, ratio, pth
    ) -> None:
        super().__init__(seeds, df, df_name, n_ensemble, n_initial, ac_type, ratio, pth)
        self.identity = f"RF_{self.ac_type}_{self.df_name}"

    def acquisition_function(self, X_j, RF_model, y_best):
        tree_predictions = []
        for j in np.arange(self.n_ensemble):
            tree_predictions.append(
                (RF_model.estimators_[j].predict(np.array([X_j]))).tolist()
            )
        mean = np.mean(np.array(tree_predictions), axis=0)[0]
        std = np.std(np.array(tree_predictions), axis=0)[0]

        if self.ac_type == "EI":
            if abs(std) > 1e-6:
                z = (y_best - mean) / std
            else:
                z = 0.0
            return (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)
        elif self.ac_type == "PI":
            if abs(std) > 1e-6:
                z = (y_best - mean) / std
                return norm.cdf(z)
            else:
                return 0.0
        else:  # LCB
            return -mean + self.lcb_ratio * std

    def main(self, input_seed):
        feature_name = list(self.df.columns)[:-1]
        objective_name = list(self.df.columns)[-1]
        X_feature = self.df[feature_name].values
        y = np.array(self.df[objective_name].values)

        n_top = int(math.ceil(len(self.df[list(self.df.columns)[-1]].values) * 0.05))
        top_indices = list(
            self.df.sort_values(list(self.df.columns)[-1]).head(n_top).index
        )

        # Initialization
        start_time = time.time()
        index_collection = []
        X_collection = []
        y_collection = []
        TopCount_collection = []
        random.seed(input_seed)
        indices = list(np.arange(self.N))
        index_learn = indices.copy()
        index_ = random.sample(index_learn, self.n_initial)
        X_ = []
        y_ = []
        c = 0
        TopCount_ = []
        for i in index_:
            X_.append(X_feature[i])
            y_.append(y[i])
            if i in top_indices:
                c += 1
            TopCount_.append(c)
            index_learn.remove(i)

        for i in np.arange(len(index_learn)):
            y_best = np.min(y_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s_scaler = preprocessing.StandardScaler()
                X_train = s_scaler.fit_transform(X_)
                y_train = s_scaler.fit_transform([[i] for i in y_])
                RF_model = RandomForestRegressor(
                    n_estimators=self.n_ensemble, n_jobs=-1
                )
                RF_model.fit(X_train, y_train)

            next_index = None
            max_ac = -(10**10)
            for j in index_learn:
                X_j = X_feature[j]
                y_j = y[j]
                ac_value = self.acquisition_function(X_j, RF_model, y_best)

                if max_ac <= ac_value:
                    max_ac = ac_value
                    next_index = j

            X_.append(X_feature[next_index])
            y_.append(y[next_index])

            if next_index in top_indices:
                c += 1

            TopCount_.append(c)

            index_learn.remove(next_index)
            index_.append(next_index)

            index_collection.append(index_)
            X_collection.append(X_)
            y_collection.append(y_)
            TopCount_collection.append(TopCount_)

        total_time = time.time() - start_time
        result = {
            "index_collection": index_collection,
            "X_collection": X_collection,
            "y_collection": y_collection,
            "TopCount_collection": TopCount_collection,
            "total_time": total_time,
        }
        return result


if __name__ == "__main__":
    # Constants
    PATH = os.getcwd()
    PATH = PATH.split(os.path.sep)
    if "src" in PATH:
        PATH.remove("src")
    PATH = os.path.sep.join(PATH)

    DATASET = ["AutoAM"]
    df = dict()
    for dataset in DATASET:
        df[dataset] = df_preprocessing(data_name=dataset)
    DATASET = df
    del df

    NUM_MODELS = 50
    # Create tasks
    models = multiprocessing.Queue()
    SEED_LIST = [random.randint(0, 9999) for _ in range(NUM_MODELS)]
    for name, df in DATASET.items():
        for acquisition in ["EI", "PI", "LCB"]:
            for type in ["RF", "GP"]:
                if type == "RF":
                    model = RF(
                        seeds=SEED_LIST,
                        df=df,
                        df_name=name,
                        n_ensemble=100,
                        n_initial=2,
                        ac_type=acquisition,
                        ratio=10,
                        pth=PATH,
                    )

                else:  # GP
                    continue
                models.put(model)

    while not models.empty():
        model = models.get()
        x = model.run()

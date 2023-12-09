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


def worker_calc_data(tasks: multiprocessing.Queue, results: multiprocessing.Queue, pth):
    while not tasks.empty():
        model: RF = tasks.get()
        identity: str = model.identity()
        result: dict = model.run()
        result["identity"] = identity

        np.savez_compressed(
            os.path.join(pth, "results", identity),
            index_collection=result["index_collection"],
            X_collection=result["X_collection"],
            y_collection=result["y_collection"],
            TopCount_collection=result["TopCount_collection"],
            total_time=result["total_time"],
            identity=result["identity"],
        )
        results.put(result)
        time.sleep(random.randint(0, 10) / 10)  # not to go too fast
    return None


class Model:
    def __init__(self) -> None:
        self.metric: float = 0.0
        self.metric_history: list[float] = [0.0]
        self.identity


class RF(Model):
    def __init__(
        self, seed, df, df_name, n_ensemble, n_initial, ac_type, ratio
    ) -> None:
        super().__init__()
        self.seed = seed
        self.df = df
        self.df_name = df_name
        self.N = len(df)
        self.n_ensemble = n_ensemble
        self.n_initial = n_initial
        self.ac_type = ac_type
        self.lcb_ratio = ratio

    def _AC(self, X_j, RF_model, y_best):
        tree_predictions = []
        for j in np.arange(self.n_ensemble):
            tree_predictions.append(
                (RF_model.estimators_[j].predict(np.array([X_j]))).tolist()
            )
        mean = np.mean(np.array(tree_predictions), axis=0)[0]
        std = np.std(np.array(tree_predictions), axis=0)[0]

        if self.ac_type == "EI":
            z = (y_best - mean) / std
            return (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)
        elif self.ac_type == "PI":
            if std != 0.0:
                z = (y_best - mean) / std
                return norm.cdf(z)
            else:
                return 0.0
        else:  # LCB
            return -mean + self.lcb_ratio * std

    def run(self):
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
        random.seed(self.seed)
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
                #             #TODO: select Acquisiton Function for BO
                ac_value = self._AC(X_j, RF_model, y_best)

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

    def identity(self):
        s = f"RF_{self.ac_type}_{self.df_name}_{self.seed}"
        return s


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

    NUM_CORES = multiprocessing.cpu_count()
    NUM_MODELS = 50

    # Create tasks
    tasks = multiprocessing.Queue()
    random.seed(5853)
    NUM_MODELS = [random.randint(0, 9999) for _ in range(NUM_MODELS)]
    for name, df in DATASET.items():
        for seed in NUM_MODELS:
            for acquisition in ["EI", "PI", "LCB"]:
                for type in ["RF", "GP"]:
                    if type == "RF":
                        model = RF(
                            seed=seed,
                            df=df,
                            df_name=name,
                            n_ensemble=100,
                            n_initial=2,
                            ac_type=acquisition,
                            ratio=10,
                        )

                    else:  # GP
                        continue
                    tasks.put(model)

    # Setting up & Running Multiprocessing on worker_calc_data()
    processes = list()
    results = multiprocessing.Queue()
    print(
        f"""
PATH = {PATH}
DATASET = {DATASET.keys()}
NUM_CORES = {NUM_CORES}
NUM_MODELS = {len(NUM_MODELS)}
        """
    )
    for i in range(0, NUM_CORES):
        process = multiprocessing.Process(
            target=worker_calc_data, args=(tasks, results, PATH)
        )
        processes.append(process)
        process.start()
    for process in processes:
        process.join()

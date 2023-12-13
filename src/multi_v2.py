import multiprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import GPy
import GPyOpt
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
        self.NUM_CORES = multiprocessing.cpu_count()

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
        manager = multiprocessing.Manager()
        tasks = manager.Queue()

        for seed in self.seeds:
            tasks.put(seed)
        for _ in range(self.NUM_CORES):
            tasks.put(None)

        processes = list()
        results = manager.Queue()

        for _ in range(self.NUM_CORES):
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

        total_time = time.time() - start_time
        result = {
            "index_collection": index_,
            "X_collection": X_,
            "y_collection": y_,
            "TopCount_collection": TopCount_,
            "total_time": total_time,
        }
        return result


class GP(Model):
    def __init__(
        self, seeds, df, df_name, n_ensemble, n_initial, ac_type, ratio, pth, kernel
    ) -> None:
        super().__init__(seeds, df, df_name, n_ensemble, n_initial, ac_type, ratio, pth)

        self.df_X_feature = self.df[list(self.df.columns)[:-1]].values
        Bias = GPy.kern.Bias(self.df_X_feature.shape[1], variance=1.0)
        k = {
            "Matern52": GPy.kern.Matern52(
                self.df_X_feature.shape[1], variance=1.0, ARD=False
            )
            + Bias,
            "Matern52_ARD": GPy.kern.Matern52(
                self.df_X_feature.shape[1], variance=1.0, ARD=True
            )
            + Bias,
            "Matern32": GPy.kern.Matern32(
                self.df_X_feature.shape[1], variance=1.0, ARD=False
            )
            + Bias,
            "Matern32_ARD": GPy.kern.Matern32(
                self.df_X_feature.shape[1], variance=1.0, ARD=True
            )
            + Bias,
            "Matern12": GPy.kern.Exponential(
                self.df_X_feature.shape[1], variance=1.0, ARD=False
            )
            + Bias,
            "Matern12_ARD": GPy.kern.Exponential(
                self.df_X_feature.shape[1], variance=1.0, ARD=True
            )
            + Bias,
            "RBF": GPy.kern.RBF(self.df_X_feature.shape[1], variance=1.0, ARD=False)
            + Bias,
            "RBF_ARD": GPy.kern.RBF(self.df_X_feature.shape[1], variance=1.0, ARD=True)
            + Bias,
            "MLP": GPy.kern.MLP(self.df_X_feature.shape[1], variance=1.0, ARD=False)
            + Bias,
            "MLP_ARD": GPy.kern.MLP(self.df_X_feature.shape[1], variance=1.0, ARD=True)
            + Bias,
        }
        if kernel not in [value for value in k.keys()]:
            raise Exception
        self.kernel = k[kernel]
        self.identity = f"GP_{kernel}_{self.ac_type}_{self.df_name}"
        self.NUM_CORES = max(multiprocessing.cpu_count() // 4, 1)

    def acquisition_function(self, X_j, GP_model, y_best):
        X_j = X_j.reshape([1, self.df_X_feature.shape[1]])
        mean, std = GP_model.predict(X_j)[0][0][0], GP_model.predict(X_j)[1][0][0]
        std = np.sqrt(std)

        if self.ac_type == "EI":
            xi = 0
            z = (y_best - mean - xi) / std
            return (y_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
        elif self.ac_type == "PI":
            xi = 0
            z = (y_best - mean - xi) / std
            return norm.cdf(z)
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

        # START
        start_time = time.time()
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
            s_scaler = preprocessing.StandardScaler()
            X_train = s_scaler.fit_transform(X_)
            y_train = s_scaler.fit_transform([[i] for i in y_])
            try:
                GP_learn = GPy.models.GPRegression(
                    X=X_train, Y=y_train, kernel=self.kernel, noise_var=0.01
                )
                GP_learn.optimize_restarts(
                    num_restarts=10,
                    parallel=True,
                    robust=True,
                    optimizer="bfgs",
                    max_iters=100,
                    verbose=False,
                )
            except:
                break
            next_index = None
            max_ac = -(10**10)
            for j in index_learn:
                X_j = X_feature[j]
                y_j = y[j]
                ac_value = self.ac_type(X_j, GP_learn, y_best)
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
        total_time = time.time() - start_time
        # END

        result = {
            "index_collection": index_,
            "X_collection": X_,
            "y_collection": y_,
            "TopCount_collection": TopCount_,
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

    NUM_MODELS = 10
    NUM_ENSEMBLES = 10
    NUM_INITIAL = 2
    LCB_RATIO = 2

    # Create tasks
    models = multiprocessing.Queue()
    random.seed(5853)
    SEED_LIST = [random.randint(0, 9999) for _ in range(NUM_MODELS)]
    for name, df in DATASET.items():
        for acquisition in ["EI", "PI", "LCB"]:
            for type in ["RF", "GP"]:
                if type == "RF":
                    model = RF(
                        seeds=SEED_LIST,
                        df=df,
                        df_name=name,
                        n_ensemble=NUM_ENSEMBLES,
                        n_initial=NUM_INITIAL,
                        ac_type=acquisition,
                        ratio=LCB_RATIO,
                        pth=PATH,
                    )
                    models.put(model)
                else:  # GP
                    for k in [
                        "Matern52",
                        "Matern52_ARD",
                        "Matern32",
                        "Matern32_ARD",
                        "Matern12",
                        "Matern12_ARD",
                        "RBF",
                        "RBF_ARD",
                        "MLP",
                        "MLP_ARD",
                    ]:
                        model = GP(
                            seeds=SEED_LIST,
                            df=df,
                            df_name=name,
                            n_ensemble=NUM_ENSEMBLES,
                            n_initial=NUM_INITIAL,
                            ac_type=acquisition,
                            ratio=LCB_RATIO,
                            pth=PATH,
                            kernel=k,
                        )
                        models.put(model)
    start = time.time()

    while not models.empty():
        model = models.get()
        print(f"Running: {model.identity}")
        model.run()

    total = time.time() - start
    print(f"total time: {total}")

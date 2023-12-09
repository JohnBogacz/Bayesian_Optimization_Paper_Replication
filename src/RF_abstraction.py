import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import os, math
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from bayes_opt import UtilityFunction, BayesianOptimization
import warnings
from joblib import Parallel, delayed
import random
import copy


class TopMetric:
    def __init__(self, df, n_top_candidates: float = 0.05) -> None:
        if df is None:
            raise Exception(f"Please give 'TopPercentMetric' a dataframe")
        top_n = int(math.ceil(len(df) * n_top_candidates))
        self.__candidates_indexes = list(
            df.sort_values(list(df.columns)[-1]).head(top_n).index
        )
        self.numerator = 0
        self.metric_history = list()

    def metric(self) -> float:
        return float(self.numerator / len(self.__candidates_indexes))

    def update_metric(self, idx: int) -> None:
        if idx in self.__candidates_indexes:
            self.numerator += 1
        metric = self.numerator / len(self.__candidates_indexes)
        self.metric_history.append(metric)
        return metric


class Model(TopMetric, UtilityFunction):
    def __init__(
        self,
        df,
        seed: int = 5835,
        n_top_candidates: float = 0.05,
    ) -> None:
        super().__init__(df=df, n_top_candidates=n_top_candidates)
        self.seed = seed
        self.pbounds = dict()
        self.df = df

        self.pool_candidates = list(df.index.tolist())
        self.observed_candidates = list()

        # (1) Running some initialization stuff
        if isinstance(df, pd.DataFrame) and not df.empty:
            self._set_pbounds(df)

    def _set_pbounds(self, df: pd.DataFrame) -> None:
        """
        This is done as we can't have symbols such as (,),%
        in the varible name when BayesianOptimization passes the column name
        into the black-box/f/surrogate-model
        """
        remove = {
            "(": "",
            ")": "",
            "%": "",
            "uL/min": "",
            " ": "",
            "(measured)": "",
            "(S/cm)": "",
        }
        for col in list(df.columns)[:-1]:
            smallest = df[col].min()
            largest = df[col].max()

            for key, val in remove.items():
                col = col.replace(key, val)
            self.pbounds[col] = (smallest, largest)

    def f_surrogate_model(self, **args):
        """
        For the template class this will server as the basic black-box function
        where we will will return the loss/objective_metric for the closest data-point
        that the BayesianOptimization predicts.
        Note: In the github repository the BayesianOptimization can not work with discrete
        values so this the similar to what example they gave and for our purpose.
        Note: Using *args as the input will be a tuple of columns, each datasets has different
        number of columns.
        """

        objective_metric = self.df.values[-1]

        col = dict()
        for c in list(self.df.columns)[:-1]:
            col[c] = self.df[c]

        for c, val in args.items():
            col[c] = np.exp2(col[c] - val)

        combined_column = np.vstack(tuple(col.values()))
        combined_column = np.sum(combined_column, axis=1)
        combined_column = np.sqrt(combined_column)
        smallest_dist = np.argmin(combined_column)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            idx = np.argwhere(
                combined_column == combined_column[smallest_dist]
            ).flatten()
            idx = idx[0]
        return objective_metric[idx]

    def add_candidate(self, **args) -> None:
        """
        In order to implement like research paper we will need list of candidates
        """
        col = dict()
        for c in list(self.df.columns)[:-1]:
            col[c] = self.df[c]

        for c, val in args.items():
            col[c] = np.exp2(col[c] - val)

        combined_column = np.vstack(tuple(col.values()))
        combined_column = np.sum(combined_column, axis=1)
        combined_column = np.sqrt(combined_column)
        smallest_dist = np.argmin(combined_column)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            idx = np.argwhere(
                combined_column == combined_column[smallest_dist]
            ).flatten()
            idx = idx[0]

        if idx in self.observed_candidates:
            return None
        else:
            self.observed_candidates.append(idx)
            self.pool_candidates.remove(idx)
            self.update_metric(idx=idx)


class RF(Model):
    def __init__(
        self,
        df,
        acquisition_type: str = "EI",
        n_est: int = 100,
        ucb_ratio: int = 10,
        n_top_candidates=0.05,
        seed=5835,
    ) -> None:
        super().__init__(df=df, n_top_candidates=n_top_candidates, seed=seed)
        if acquisition_type not in ["EI", "PI", "UCB2"]:
            raise Exception(
                "Only 3 options for 'acquisition'\n[1] 'EI'\n[2] 'PI'\n[3] 'UCB2'"
            )
        self.aqu_type = acquisition_type
        self.n_est = n_est
        self.ucb_ratio = ucb_ratio

    # ------------------- Acquisition -------------------
    """
    class UtilityFunction:
        gp: gaussian processed fitted to relevant data
        y_max: The current maximum known value of the target function
        self.kappa: Parameter to indicate how closed are the next parameters sampled.
        self.xi: mean for gp???
    """

    def utility(self, x, gp, y_max):
        if self.aqu_type == "UCB2":
            return self._ucb(x, gp, 1)
        if self.aqu_type == "EI":
            return self._ei(x, gp, y_max, 1)
        if self.aqu_type == "PI":
            return self._poi(x, gp, y_max, 1)

    def _helper(self, x):
        # (1) Get X and y from pool
        df_pool = self.df.loc[self.pool_candidates]
        # X = df_pool[list(df_pool.columns)[:-1]].values
        # y = df_pool[list(df_pool.columns)[-1]].values

        # (2) Processing
        # y_max = np.max(y)
        s_scaler = StandardScaler()
        X_train = s_scaler.fit_transform(df_pool[list(df_pool.columns)[:-1]].values)
        ###!!!y_train = s_scaler.fit_transform([[i] for i in y])!!!###
        y_reshape = np.array(df_pool[list(df_pool.columns)[-1]].values).reshape(-1, 1)
        y_train = s_scaler.fit_transform(y_reshape).flatten()
        ###

        RF_model = RandomForestRegressor(n_estimators=self.n_est, n_jobs=-1)
        RF_model.fit(X_train, y_train)

        # Fast way
        # tree_predictions = list()

        def predict_est(estimator, x):
            return estimator.predict(x)

        tree_predictions = Parallel(n_jobs=-1)(
            delayed(predict_est)(RF_model.estimators_[j], x)
            for j in np.arange(self.n_est)
        )

        mean = np.mean(np.array(tree_predictions), axis=0)[0]
        std = np.std(np.array(tree_predictions), axis=0)[0]

        return mean, std

    # @staticmethod
    def _ucb(self, x, gp, kappa):
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std
        """
        mean, std = self._helper(x)
        # (4) Acquisition
        return mean + (self.ucb_ratio * std)

    # @staticmethod
    def _ei(self, x, gp, y_max, xi):
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = mean - y_max - xi
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)
        """
        """
        # (1) Get X and y from pool
        df_pool = self.df.loc[self.pool_candidates]
        X = df_pool[list(df_pool.columns)[:-1]].values
        y = df_pool[list(df_pool.columns)[-1]].values

        # (2) Processing
        y_max = np.max(y)
        s_scaler = StandardScaler()
        X_train = s_scaler.fit_transform(X)
        ###y_train = s_scaler.fit_transform([[i] for i in y])
        y_reshape = np.array(y).reshape(-1, 1)
        y_train = s_scaler.fit_transform(y_reshape).flatten()
        ###

        RF_model = RandomForestRegressor(n_estimators=self.n_est, n_jobs=-1)
        RF_model.fit(X_train, y_train)

        ###!!! I'm deciding to calc based on input X rather then loop through all candidates!!!
        ###!!! Fix for other Acquisition functions!!!
        # (3) Loop through candidates and find smallest acquisition value

        # Fast way
        tree_predictions = list()

        def predict_est(estimator, x):
            return estimator.predict(x)

        tree_predictions = Parallel(n_jobs=-1)(
            delayed(predict_est)(RF_model.estimators_[j], x)
            for j in np.arange(self.n_est)
        )

        mean = np.mean(np.array(tree_predictions), axis=0)[0]
        std = np.std(np.array(tree_predictions), axis=0)[0]
        """
        mean, std = self._helper(x)

        # (4) Acquisition
        if std != 0.0:
            z = (y_max - mean) / std
        else:
            z = 0
        val = (y_max - mean) * norm.cdf(z) + std * norm.pdf(z)
        return val

    # @staticmethod
    def _poi(self, x, gp, y_max, xi):
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return norm.cdf(z)
        """

        mean, std = self._helper(x)

        # (4) Acquisition
        z = (y_max - mean) / std
        return norm.cdf(z)

    # ------------------- Acquisition -------------------


if __name__ == "__main__":
    PATH = os.getcwd()
    PATH = PATH.split(os.path.sep)
    if "src" in PATH:
        PATH.remove("src")
    PATH = os.path.sep.join(PATH)

    def df_preprocessing(data_name: str) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(PATH, "datasets", data_name + "_dataset.csv"))
        # (A) There are multiple of the same experiments, so average them!
        features = list(df.columns)[:-1]
        obj_metric = list(df.columns)[-1]
        df = df.groupby(features)[obj_metric].agg(lambda x: x.unique().mean())
        df = (df.to_frame()).reset_index()
        # (B) Change all NECESSARY df to a maximization problem
        if data_name not in ["P3HT", "Crossed barrel", "AutoAM"]:
            # aka AgNP & Perovskite
            df[obj_metric] = -df[obj_metric].values
        return df

    dataset_df = df_preprocessing("Crossed barrel")

    template_model = RF(
        n_est=10, ucb_ratio=10, df=dataset_df, n_top_candidates=0.05
    )  # 100

    template_dict = {
        "EI": copy.deepcopy(template_model),
        "POI": copy.deepcopy(template_model),
        "UCB2": copy.deepcopy(template_model),
    }
    template_dict["EI"].acquisition_type = "EI"
    template_dict["POI"].acquisition_type = "POI"
    template_dict["UCB2"].acquisition_type = "UCB2"

    random.seed(5853)
    n_models = 50
    n_models = [random.randint(0, 9999) for _ in range(n_models)]

    for seed in n_models:
        for key, val in template_dict.items():
            Model = copy.copy(val)
            Model.seed = seed

            optimizer = BayesianOptimization(
                f=Model.f_surrogate_model,
                pbounds=Model.pbounds,
                random_state=Model.seed,
            )

            while Model.metric() < 0.1:  # 1.0
                next_point = optimizer.suggest(utility_function=Model)
                Model.add_candidate(**next_point)
                value = Model.f_surrogate_model(**next_point)
                optimizer.register(params=next_point, target=value)

                print(f"metric: {Model.metric()}")
                print(f"next_point: {next_point}")
                print(f"value: {value}")
                # print(f"Model.pool_candidates: {Model.pool_candidates}")
                print(f"Model.observed_candidates: {Model.observed_candidates}")

            with open(f"{seed}_{key}.txt", "w") as f:
                f.write("\n".join(map(str, Model.metric_history)))

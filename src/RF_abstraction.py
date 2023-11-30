import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import os, math
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from bayes_opt import UtilityFunction, BayesianOptimization
import warnings


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
        return self.numerator / len(self.__candidates_indexes)

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
        self.pbounds = dict[str, tuple]
        self.df = df

        self.pool_candidates = df.index.tolist()
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
        for col in list(df.columns)[:-1]:
            smallest = df[col].min()
            largest = df[col].max()
            remove = {
                "(": "",
                ")": "",
                "%": "",
                "uL/min": "",
                " ": "",
                "(measured)": "",
                "(S/cm)": "",
            }
            for key, val in remove.items():
                col = col.replace(key, val)
            self.pbounds[col] = (smallest, largest)

    def f_surrogate_model(self, *args):
        """
        For the template class this will server as the basic black-box function
        where we will will return the loss/objective_metric for the closest data-point
        that the BayesianOptimization predicts.
        Note: In the github repository the BayesianOptimization can not work with discrete
        values so this the similar to what example they gave and for our purpose.
        Note: Using *args as the input will be a tuple of columns, each datasets has different
        number of columns.
        """
        features = self.df.values[:-1]
        objective_metric = self.df.values[-1]

        if len(args) != features.shape[0]:
            raise Exception("Error")

        temp_matrix = np.zeros_like(features)
        for arg_i in range(args):
            # matrix_col = (matrix_col - arg)^2
            temp_matrix[:, arg_i] = np.exp2(features[:, arg_i] - args[arg_i])
        row_sums = np.sum(temp_matrix, axis=1)
        row_sums = np.sqrt(row_sums)
        smallest_dist = np.argmin(row_sums)

        closest_objective_metric = objective_metric[smallest_dist]
        return closest_objective_metric

    def add_candidate(self, *args) -> None:
        """
        In order to implement like research paper we will need list of candidates
        """
        features = self.df.values[:-1]

        if len(args) != features.shape[0]:
            raise Exception("Error")

        temp_matrix = np.zeros_like(features)
        for arg_i in range(args):
            # matrix_col = (matrix_col - arg)^2
            temp_matrix[:, arg_i] = np.exp2(features[:, arg_i] - args[arg_i])
        row_sums = np.sum(temp_matrix, axis=1)
        row_sums = np.sqrt(row_sums)
        smallest_dist_idx = np.argmin(row_sums)

        if smallest_dist_idx in self.observed_candidates:
            return None
        else:
            self.observed_candidates.append(smallest_dist_idx)
            self.pool_candidates.remove(smallest_dist_idx)
            self.update_metric(idx=smallest_dist_idx)


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
            return self._ucb(x, gp, self.kappa)
        if self.aqu_type == "EI":
            return self._ei(x, gp, y_max, self.xi)
        if self.aqu_type == "PI":
            return self._poi(x, gp, y_max, self.xi)

    # @staticmethod
    def _ucb(self, x, gp, kappa):  # Return numpy array of 1 element, x=input?
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std
        """
        # (1) Get X and y from pool
        df_pool = self.df.loc[self.pool_candidates]
        X = df_pool[list(df_pool.columns)[:-1]].values
        y = df_pool[list(df_pool.columns)[-1]].values

        # (2) Processing
        y_max = y.argmax
        s_scaler = StandardScaler()
        X_train = s_scaler.fit_transform(X)
        y_train = s_scaler.fit_transform([[i] for i in y])
        RF_model = RandomForestRegressor(n_estimators=self.n_est, n_jobs=-1)
        RF_model.fit(X_train, y_train)

        # (3) Loop through candidates and find smallest acquisition value
        for idx in self.pool_candidates:
            X_j = X[idx]
            # Mean & STD
            tree_predictions = []
            for j in np.arange(self.n_est):
                tree_predictions.append(
                    (RF_model.estimators_[j].predict(np.array([X_j]))).tolist()
                )
            mean = np.mean(np.array(tree_predictions), axis=0)[0]
            std = np.std(np.array(tree_predictions), axis=0)[0]

        # (4) Acquisition
        return mean + (self.ucb_ratio * std)

    # @staticmethod
    def _ei(self, x, gp, y_max, xi):  # Return numpy array of 1 element, x=input?
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = mean - y_max - xi
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)
        """
        # (1) Get X and y from pool
        df_pool = self.df.loc[self.pool_candidates]
        X = df_pool[list(df_pool.columns)[:-1]].values
        y = df_pool[list(df_pool.columns)[-1]].values

        # (2) Processing
        y_max = y.argmax
        s_scaler = StandardScaler()
        X_train = s_scaler.fit_transform(X)
        y_train = s_scaler.fit_transform([[i] for i in y])
        RF_model = RandomForestRegressor(n_estimators=self.n_est, n_jobs=-1)
        RF_model.fit(X_train, y_train)

        # (3) Loop through candidates and find smallest acquisition value
        for idx in self.pool_candidates:
            X_j = X[idx]
            # Mean & STD
            tree_predictions = []
            for j in np.arange(self.n_est):
                tree_predictions.append(
                    (RF_model.estimators_[j].predict(np.array([X_j]))).tolist()
                )
            mean = np.mean(np.array(tree_predictions), axis=0)[0]
            std = np.std(np.array(tree_predictions), axis=0)[0]

        # (4) Acquisition
        z = (y_max - mean) / std
        return (y_my_maxin - mean) * norm.cdf(z) + std * norm.pdf(z)

    # @staticmethod
    def _poi(self, x, gp, y_max, xi):  # Return numpy array of 1 element, x=input?
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return norm.cdf(z)
        """
        # (1) Get X and y from pool
        df_pool = self.df.loc[self.pool_candidates]
        X = df_pool[list(df_pool.columns)[:-1]].values
        y = df_pool[list(df_pool.columns)[-1]].values

        # (2) Processing
        y_max = y.argmax
        s_scaler = StandardScaler()
        X_train = s_scaler.fit_transform(X)
        y_train = s_scaler.fit_transform([[i] for i in y])
        RF_model = RandomForestRegressor(n_estimators=self.n_est, n_jobs=-1)
        RF_model.fit(X_train, y_train)

        # (3) Loop through candidates and find smallest acquisition value
        for idx in self.pool_candidates:
            X_j = X[idx]
            # Mean & STD
            tree_predictions = []
            for j in np.arange(self.n_est):
                tree_predictions.append(
                    (RF_model.estimators_[j].predict(np.array([X_j]))).tolist()
                )
            mean = np.mean(np.array(tree_predictions), axis=0)[0]
            std = np.std(np.array(tree_predictions), axis=0)[0]

        # (4) Acquisition
        z = (y_max - mean) / std
        return norm.cdf(z)

    # ------------------- Acquisition -------------------

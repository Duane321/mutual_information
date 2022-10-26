import numpy as np
import pandas as pd
import altair as alt
from tqdm import tqdm
import os
from itertools import product
from hashlib import sha256


class RandomWalkTDvsMC:

    """
    This class runs many simulations of the Monte Carlo and Temporal Difference algorithms on the Random Walk evaluation
    task, for a variety of values of the parameters alpha and n_steps. The results are stored in the self.MC_runs and
    self.TD_runs attributes, which are dataframes. Since running all these simulations may take a long time, this class
    stores results in the /data folder. So if you run the simulations, close the notebook and re-run, you'll begin
    wherever you left off.

    The class attributes of mc_alphas, td_alphas and n_steps determine the set of hyperparameters over which we will
    run these algorithms (for TD, it's every combination of td_alphas and n_steps).

    """

    mc_alphas = [
        0.001,
        0.01,
        0.03,
        0.05,
        0.08,
        0.1,
    ]
    td_alphas = [
        0.001,
        0.01,
        0.03,
        0.05,
        0.08,
        0.1,
        0.16,
        0.2,
        0.6,
        1.0,
    ]
    n_steps = [1, 2, 4, 16, 64]
    initial_v = 0.5

    def __init__(self, n_states: int, n_algo_runs: int, M: int, seed: int = 0):
        """

        Parameters
        ----------
        n_states: number of states
        n_algo_runs: number of runs of each algorithm (Monte Carlo and Temporal Difference)
        M: number of episodes per run of each algorithm.
        """

        assert not ((n_states - 1) % 2), f"n_states ({n_states}) needs to be odd."
        assert n_states > 2, f"n_states ({n_states}) needs to be greater than 2."

        self.n_states = n_states
        self.n_algo_runs = n_algo_runs
        self.M = M
        self.seed = seed

        np.random.seed(seed)
        self.max_distance = (self.n_states - 1) // 2
        self.episode_runs = self.generate_all_episodes()
        self.Vs_true = {
            s: i / (self.n_states - 1)
            for i, s in enumerate(range(-self.max_distance, self.max_distance + 1))
        }
        self.MC_runs = None
        self.TD_runs = None

        # Create the data directory if it doesn't already exist.
        if not os.path.exists("data"):
            os.mkdir("data")

    def get_init_V(self):
        """Creates the initial value function table with all zeros."""
        max_dist_less_1 = self.max_distance - 1
        return pd.DataFrame(
            dict(
                s=[-max_dist_less_1 + float(i) for i in range(max_dist_less_1 * 2 + 1)],
                v=[self.initial_v] * (self.n_states - 2),
            )
        ).set_index("s")

    def sample_ep(self):
        """Sample one episode"""
        position = 0
        episode = [position]
        while abs(position) != self.max_distance:
            move = np.random.choice([-1, 1])
            position += move
            episode.append(position)
        return episode

    def generate_all_episodes(self):
        """
        Generates all episode data for applying MC and TD. The object returned, 'episode_runs', is a list of list of
        lists, where:

            episode_runs[k][m]

        is the m-th episode from the k-th run of episodes. Each episode is a list of positions.

        """

        episode_runs = []
        for _ in range(self.n_algo_runs):
            episode_runs.append([self.sample_ep() for _ in range(self.M)])

        return episode_runs

    def _apply_mc_update(self, Vs, episode, alpha):
        """
        Applies the update to the value function table.
        """
        Vs = Vs.copy()
        return_val = 1 if episode[-1] == self.max_distance else 0
        for s in episode[:-1]:
            Vs.loc[s, "v"] = Vs.loc[s, "v"] + alpha * (return_val - Vs.loc[s, "v"])
        return Vs

    def _apply_td_update(self, Vs, episode, alpha, n_steps):
        """
        Note: TD is applied as the episode is being learned. Since this class is to simply explore the behavior of the
        algorithms, it's not necessary that we actually generate the episode while applying the update rule.

        Hm... need a good way to TEST THIS!
        """
        Vs = Vs.copy()

        # There are len(episode) - 1 rewards
        rewards = [0] * (len(episode) - 1)  # The time index of these are 1, 2, ...
        rewards[-1] = 1 if episode[-1] == self.max_distance else 0

        for t, s in enumerate(episode[:-1]):
            obs_reward_sum = sum(rewards[t : t + n_steps])
            if (t + n_steps) < (len(episode) - 1):
                Vst = Vs.loc[episode[t + n_steps], "v"]
            else:
                Vst = 0
            target = obs_reward_sum + Vst
            Vs.loc[s, "v"] = Vs.loc[s, "v"] + alpha * (target - Vs.loc[s, "v"])

        return Vs

    def _get_run_hash(self, run_idx, algo_method_name, **algo_kwargs):
        """This determines the equivalence class of processing the run_idx-th run of the algo_method_name algorithm.

        If one processing of episode is done with the same:
            - run_idx
            - algo_method_name (either MC or TD)
            - arguments to the algo (alpha and/or n_steps)
            - seed
            - M
            - n_states
        We can say that the result of the algorithm will be the same (and therefore, we don't need to recompute it if
        we've already done so in the past)
        """
        string = f"{run_idx}_{algo_method_name}_{algo_kwargs}_{self.seed}_{self.M}_{self.n_states}"
        return sha256(string.encode("utf-8")).hexdigest()

    def _run_algo(self, run_idx, algo_method_name, **algo_kwargs):
        """
        Runs the algorithm. Returns a wide table giving the value function following the processing of each
        episode from the run_idx-th run of episodes. If the algo has been run before, it'll load the data from the
        data/ folder.

        Parameters
        ----------
        run_idx: index of the run of episodes to use.
        algo_method_name: name of the method to use for the algorithm.
        algo_kwargs: keyword arguments to pass to the algorithm.

        """
        hash_str = self._get_run_hash(run_idx, algo_method_name, **algo_kwargs)
        filename = "data\\" + hash_str

        if os.path.exists(filename):
            print(
                f"Loading data for: run_idx = {run_idx}, algo = {algo_method_name}, seed = {self.seed}, M = {self.M}, n_states = {self.n_states}, "
                + ", ".join([f"{k} = {v}" for k, v in algo_kwargs.items()])
            )
            Vs_by_m = pd.read_pickle(filename)
        else:
            episode_run = self.episode_runs[run_idx]
            # episode_run = self.sample_M_episodes()
            Vs_by_m = [self.get_init_V()]
            for episode in episode_run:
                Vs_by_m.append(
                    getattr(self, algo_method_name)(Vs_by_m[-1], episode, **algo_kwargs)
                )

            Vs_by_m = pd.concat(
                [Vs.rename(columns=dict(v=m)) for m, Vs in enumerate(Vs_by_m)], axis=1
            )
            Vs_by_m.to_pickle(filename)

        return Vs_by_m, hash_str

    def run_mc(self, run_idx, alpha):
        """
        Runs the Monte Carlo algorithm. Returns a wide table giving the value function following the processing of each
        episode from the run_idx-th run of episodes.

        Parameters
        ----------
        run_idx: index of the run of episodes to use.
        alpha: the stepsize used for the MC update rule.

        """
        Vs_by_m, hash_str = self._run_algo(run_idx, "_apply_mc_update", alpha=alpha)
        return Vs_by_m, hash_str

    def run_td(self, run_idx, alpha, n_steps):
        """
        Runs the Temporal Difference algorithm. Returns a wide table giving the value function following the processing
        of each episode from the run_idx-th run of episodes.

        Parameters
        ----------
        run_idx: index of the run of episodes to use.
        alpha: the stepsize used for the TD update rule.
        n_steps: number of steps to use for the TD update rule.

        """
        Vs_by_m, hash_str = self._run_algo(
            run_idx, "_apply_td_update", alpha=alpha, n_steps=n_steps
        )
        return Vs_by_m, hash_str

    def _run_algo_all(self, algo_method_name, **algo_kwargs):
        runs = []
        for run_idx in range(self.n_algo_runs):
            Vs_by_m, hash_str = self._run_algo(run_idx, algo_method_name, **algo_kwargs)
            Vs_over_run = (
                Vs_by_m.reset_index()
                .melt(id_vars=["s"])
                .rename(columns=dict(variable="m", value="v"))
                .assign(
                    run_idx=run_idx, seed=self.seed, hash_str=hash_str, **algo_kwargs
                )
            )
            runs.append(Vs_over_run)
        runs = pd.concat(runs, axis=0)
        runs["v_true"] = runs["s"].map(self.Vs_true)
        runs["abs_error"] = (runs["v"] - runs["v_true"]).abs()
        return runs

    def _append_runs(self, which: str, runs: pd.DataFrame):
        assert which in ["TD", "MC"]
        self_runs = getattr(self, f"{which}_runs")
        if self_runs is None:
            setattr(self, f"{which}_runs", runs)
        else:
            setattr(self, f"{which}_runs", pd.concat([self_runs, runs], axis=0))

    def run_td_multi_run(self, alpha, n_steps):
        print(
            f"Running TD for alpha = {alpha} and n_steps = {n_steps} on {self.n_algo_runs} runs."
        )
        TD_runs = self._run_algo_all("_apply_td_update", alpha=alpha, n_steps=n_steps)
        self._append_runs("TD", TD_runs)

    def run_mc_multi_run(self, alpha):
        print(f"Running MC for alpha = {alpha} on {self.n_algo_runs} runs.")
        MC_runs = self._run_algo_all("_apply_mc_update", alpha=alpha)
        self._append_runs("MC", MC_runs)

    def _get_all_hash(self, which):
        assert which in ["TD", "MC"]

        to_hash = [self.initial_v, self.seed, self.M, self.n_states, self.n_algo_runs]
        string_to_hash = ""
        if which == "TD":
            to_hash.extend([self.td_alphas, self.n_steps])
        else:
            to_hash.append(self.mc_alphas)

        for arg in to_hash:
            string_to_hash += str(arg)

        return sha256(string_to_hash.encode("utf-8")).hexdigest()

    def run_td_all(self):
        hash_str = self._get_all_hash("TD")
        filename = "data\\" + hash_str
        if os.path.exists(filename):
            print("Loading all runs for TD")
            self.TD_runs = pd.read_pickle(filename)
        else:
            for (alpha, n_steps) in tqdm(product(self.td_alphas, self.n_steps)):
                self.run_td_multi_run(alpha=alpha, n_steps=n_steps)
            self.TD_runs.to_pickle(filename)

    def run_mc_all(self):
        hash_str = self._get_all_hash("MC")
        filename = "data\\" + hash_str
        if os.path.exists(filename):
            print("Loading all runs for MC")
            self.MC_runs = pd.read_pickle(filename)
        else:
            for alpha in tqdm(self.mc_alphas):
                self.run_mc_multi_run(alpha=alpha)
            self.MC_runs.to_pickle(filename)

    def _get_errors(self, which: str):
        """
        Get absolute errors averaged over states.
        """

        assert which in ["TD", "MC"]
        conditionals = ["m", "run_idx", "alpha"]
        if which == "TD":
            conditionals.append("n_steps")

        if getattr(self, f"{which}_runs") is None:
            if which == "TD":
                self.run_td_all()
            else:
                self.run_mc_all()

        return (
            getattr(self, f"{which}_runs")
            .groupby(conditionals)[["abs_error"]]
            .mean()
            .reset_index()
        )

    def get_errors(self, which):
        hash_str = f"ave_errors_{which}" + self._get_all_hash("MC")
        filename = "data\\" + hash_str

        if os.path.exists(filename):
            print(f"Loading average errors for {which}")
            ave_errors = pd.read_pickle(filename)
        else:
            ave_errors = self._get_errors(which)

        setattr(self, f"{which}_ave_errors", ave_errors)
        return getattr(self, f"{which}_ave_errors")

    def get_error_summary(
        self, errors_filtered: pd.DataFrame,
    ):
        """
        Given a dataframe filtered down to particular hyperparameters (alphas and n_steps) for either TD or MC, this
        will return a dataframe giving the average error after processing m episode, along with lower_percentile-th to
        upper_percentile-th percentile performance.
        """

        error_std = (
            errors_filtered.groupby(["m"])[["abs_error"]]
            .std()
            .rename(columns=dict(abs_error="abs_error_std"))
        )

        error_mean = (
            errors_filtered.groupby(["m"])[["abs_error"]]
            .mean()
            .rename(columns=dict(abs_error="abs_error_mean"))
        )
        return (
            pd.concat([error_mean, error_std], axis=1)
            .assign(
                lower_1_std=lambda df: df["abs_error_mean"] - df["abs_error_std"],
                upper_1_std=lambda df: df["abs_error_mean"] + df["abs_error_std"],
            )
            .reset_index()
        )

    def plot_errors_td_vs_mc(
        self, alpha_mc, alpha_td, n_steps, width=300, height=300,
    ):
        """As seen in the video, creates the TD vs MC plot for a given set of hyperparameters."""

        errors_TD = (
            self.get_errors("TD")
            .query(f"alpha == {alpha_td} & n_steps == {n_steps}")
            .pipe(self.get_error_summary)
            .assign(which="TD")
        )

        assert errors_TD.shape[0] > 0, f"No TD runs found for alpha_td = {alpha_td} and n_steps = {n_steps}."

        errors_MC = (
            self.get_errors("MC")
            .query(f"alpha == {alpha_mc}")
            .pipe(self.get_error_summary)
            .assign(which="MC")
        )

        assert errors_MC.shape[0] > 0, f"No MC runs found for alpha_mc = {alpha_mc}."

        errors = pd.concat([errors_TD, errors_MC], axis=0)

        chart_means = (
            alt.Chart(errors)
            .mark_line(strokeWidth=1)
            .encode(x="m", y=alt.Y("abs_error_mean", title="Ave. Abs. Error"), color="which")
        )
        chart_bands = (
            alt.Chart(errors)
            .mark_area(opacity=0.5)
            .encode(x="m", y="lower_1_std", y2="upper_1_std", color="which")
        )
        return (chart_bands + chart_means).properties(width=width, height=height)

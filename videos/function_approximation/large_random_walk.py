import numpy as np
import pandas as pd
import altair as alt
from tqdm import tqdm
import os
from itertools import product
from hashlib import sha256
from scipy.linalg import solve


class LargeRandomWalk:

    """
    N-Step Semi-gradient TD applied to the 1000-state random walk from examples 9.1 and 9.2 from Sutton/Barto.
    """

    def __init__(
        self,
        n_steps: int,
        alpha: float,
        seed: int = 0,
    ):
        """

        Parameters
        ----------
        n_steps: number of steps for the TD target. n = infinity for MC.
        alpha: step size
        seed: random seed
        """

        assert 0 < alpha < 1, f"alpha ({alpha}) needs to be between 0 and 1."

        self.n_states = 1000
        self.n_steps = n_steps
        self.alpha = alpha
        self.num_buckets = 10
        self.seed = seed

        np.random.seed(seed)
        self.bucket_size = self.n_states // self.num_buckets
        possible_moves = np.arange(1, self.bucket_size + 1)
        self.possible_moves = np.concatenate([-possible_moves[::-1], possible_moves])
        self.states_micro = np.arange(1, self.n_states + 1)

        # Check the bucket_states method creates evenly sized buckets.
        assert (
            pd.Series(self.bucket_states(self.states_micro)).value_counts().nunique()
            == 1
        )

        # Solve for the true value
        self.Vs_true, self.mu_s_true = self.solve_Vs()

        # Initialize value coefficients
        self.w = np.zeros(self.num_buckets)

    def bucket_states(self, states_micro: np.array):
        return (states_micro - 1) // self.bucket_size

    def sample_ep(self):
        """Sample one episode"""

        path = np.array([500])

        chunk_size = self.num_buckets * 4
        while (1 < path.min()) and (path.max() < self.n_states):
            moves = np.random.choice(self.possible_moves, size=chunk_size)
            path = np.concatenate([path, path[-1] + moves.cumsum()])

        idx_of_max_distance = np.where((path <= 1) | (path >= self.n_states))[0][0]
        path = path[: idx_of_max_distance + 1].clip(1, self.n_states)

        return self.bucket_states(path)

    def solve_Vs(self):
        """
        Sets up a system of equations to solve for the true values v(s) and state distribution mu(s).
        """
        Pss = np.zeros(
            (self.n_states, self.n_states)
        )  # Gives Prob(s|s) where s varies along the rows

        for r in range(1, Pss.shape[0] - 1):

            Pss[r, max(0, r - self.bucket_size) : r] = 1 / (2 * self.bucket_size)
            Pss[r, r + 1 : min(r + self.bucket_size + 1, Pss.shape[1])] = 1 / (
                2 * self.bucket_size
            )
            row_sum = Pss[r, :].sum()
            if r < Pss.shape[0] // 2:
                Pss[r, 0] += 1 - row_sum
            else:
                Pss[r, -1] += 1 - row_sum

        Rs = np.zeros(
            self.n_states
        )  # Gives the reward when transitioning into state s.
        Rs[0] = -1
        Rs[-1] = 1

        """
        Vs = Pss x (Rs + Vs) # Bellman equation
        Vs = Pss x Rs + Pss x Vs
        - Pss x Rs = Pss x Vs - I x Vs
        Pss x Rs = (I - Pss) x Vs
        """
        Vs_ = solve(np.eye(self.n_states) - Pss, Pss @ Rs)

        Vs = (
            pd.Series(Vs_[1:-1])
            .to_frame("Vs_true")
            .reset_index()
            .rename(columns=dict(index="micro_state"))
            .assign(micro_state=lambda df: df["micro_state"] + 2)
        )

        """
        See section 9.2, grey box.

        eta_s = hs + Pss * eta_s
        eta_s - Pss * eta_s = hs
        (I - Pss) * eta_s = hs
        """

        hs = np.zeros(self.n_states)  # Initializing distribution over states
        hs[500] = 1.0

        eta_s = solve(np.eye(self.n_states) - Pss, hs)
        mu_s_ = eta_s / eta_s.sum()

        mu_s = (
            pd.Series(mu_s_[1:-1])
            .to_frame("mu_s_true")
            .reset_index()
            .rename(columns=dict(index="micro_state"))
            .assign(micro_state=lambda df: df["micro_state"] + 2)
        )

        return Vs, mu_s

    def get_state_value(self, s: int):
        """
        Returns the estimated value of state s, where s is a bucket state.
        """
        return self.w[
            s
        ]  # Same thing as a dot product it turns out. No need to one-hot encode.

    def update_w(self):

        """
        n-step Semi-Gradient TD. Updates the value coefficient based on sampling one episode.
        """

        episode = self.sample_ep()

        rewards = [0] * (len(episode) - 1)  # The time index of these are 1, 2, ...
        rewards[-1] = -1 if episode[-1] == 0 else 1

        for t, s in enumerate(episode[:-1]):
            obs_reward_sum = sum(rewards[t : t + self.n_steps])
            if (t + self.n_steps) < (len(episode) - 1):
                Vst = self.get_state_value(episode[t + self.n_steps])
            else:
                Vst = 0
            target = obs_reward_sum + Vst

            # Again, dot product-ing with a one-hot vector is the same as indexing.
            self.w[s] += self.alpha * (target - self.get_state_value(s))

    def run(self, num_episodes: int):
        """Updates w over num_episodes-episodes."""
        for _ in tqdm(range(num_episodes)):
            self.update_w()

    def create_values_df(self):
        """

        Creates the dataframe that contains the true values and estimated values for all states.

        It's what's used to create the plot Example 9.1 and 9.2 from sutton and barto.

        +----+---------------+---------------+-----------+-----------+
        |    |   micro_state |   macro_state |   Vs_true |        Vs |
        +====+===============+===============+===========+===========+
        |  0 |             2 |             0 | -0.921813 | -0.819504 |
        +----+---------------+---------------+-----------+-----------+
        |  1 |             3 |             0 | -0.920568 | -0.819504 |
        +----+---------------+---------------+-----------+-----------+
        |  2 |             4 |             0 | -0.919314 | -0.819504 |
        +----+---------------+---------------+-----------+-----------+
        |  3 |             5 |             0 | -0.918051 | -0.819504 |
        +----+---------------+---------------+-----------+-----------+
        |  4 |             6 |             0 | -0.916779 | -0.819504 |
        +----+---------------+---------------+-----------+-----------+
        """

        value_estimates = (
            pd.Series(self.w)
            .to_frame("Vs")
            .reset_index()
            .rename(columns={"index": "macro_state"})
        )
        value_estimates_micro = (
            value_estimates.iloc[self.bucket_states(self.states_micro[1:-1]), :]
            .assign(micro_state=self.states_micro[1:-1])
            .set_index("micro_state")
        )
        return pd.concat(
            [self.Vs_true.set_index("micro_state"), value_estimates_micro], axis=1
        ).reset_index()[["micro_state", "macro_state", "Vs_true", "Vs"]]

    def plot_true_values(self, width: int = 400, height: int = 200):
        """Creates a plot of the true value function"""
        return (
            alt.Chart(self.Vs_true)
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X("micro_state", title="state"), y=alt.Y("Vs_true", title="value")
            )
            .properties(width=width, height=height, title="True values")
        )

    def plot_true_mu_s(self, width: int = 400, height: int = 200):
        """Creates a plot of the true distribution over states"""
        return (
            alt.Chart(self.mu_s_true)
            .mark_line(strokeWidth=1)
            .encode(
                x=alt.X("micro_state", title="state"),
                y=alt.Y("mu_s_true", title="mu(s)"),
            )
            .properties(width=width, height=height, title="state distribution")
        )

    def plot_true_vs_est(self, width: int = 400, height: int = 200):
        """Creates a plot of the true value function and the estimated value function"""
        values_df = self.create_values_df()

        est = (
            alt.Chart(values_df)
            .mark_line(strokeWidth=1.5, color="green")
            .encode(x=alt.X("micro_state", title="state"), y="Vs")
        )
        true = (
            alt.Chart(values_df)
            .mark_line(strokeWidth=1.5, color="blue")
            .encode(
                x=alt.X("micro_state", title="state"), y=alt.Y("Vs_true", title="value")
            )
        )

        return (true + est).properties(
            width=width, height=height, title="True vs. Estimated Values"
        )

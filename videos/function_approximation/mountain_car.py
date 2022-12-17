import numpy as np
import pandas as pd
import altair as alt
from tqdm import tqdm
import os
from hashlib import sha256
from itertools import count, product
from typing import Tuple, List


class MountainCar:

    """
    Applies N-Step Sarsa to the mountain car problem (Example 10.1 from Sutton/Barto) and provides plotting functions.

    Note: This uses a different feature representation than the one used in the book.
    """

    def __init__(
        self,
        alpha: float,
        distance_scaler: float,
        n: int,
        protos_per_dim: int = 35,
        seed: int = 0,
    ):
        """
        Initializes the MountainCar task. Be careful with the algo-inputs. It is easy to set them such that learning
        process becomes slows or unstable.

        Parameters
        ----------
        alpha: step size
        distance_scaler: scales the distances prior to be fed into the Gaussian. A smaller number has a similar effect
        to reducing K in K-nearest neighbors.
        n: number of steps to look ahead.
        protos_per_dim: number of proto points per dimension. There will be protos_per_dim^2 total.
        seed: random seed
        """

        assert 0 < alpha < 1, f"alpha ({alpha}) needs to be between 0 and 1."

        self.alpha = alpha
        self.seed = seed
        self.distance_scaler = distance_scaler
        self.n = n
        np.random.seed(seed)
        self.actions = [-1, 0, 1]
        self.gamma = 1
        self.epsilon = (
            0.0  # B/c 'everything looks bad', exploration is natural, so this can be 0.
        )
        self.protos_per_dim = protos_per_dim  # We'll tile the 2D space with protos_per_dim^2 proto-typical points.
        self.vel_min_max = (-0.07, 0.07)
        self.pos_min_max = (-1.2, 0.5)
        self.proto_points = self.get_proto_points()
        self.n_plot_points = 100

        self.initialize()

        # Create the data directory if it doesn't already exist.
        if not os.path.exists("data"):
            os.mkdir("data")

    def initialize(self):
        """Initializes the weights and reset results"""
        self.w = np.zeros((self.protos_per_dim**2, len(self.actions)))
        self.results = None
        self.cost_to_go = None

    def get_proto_points(self) -> np.ndarray:
        """
        Returns the proto points for the RBF. These are the centers of the Gaussian basis functions.
        """
        proto_vel = np.linspace(*self.vel_min_max, self.protos_per_dim + 2)[1:-1]
        proto_pos = np.linspace(*self.pos_min_max, self.protos_per_dim + 2)[1:-1]
        return np.array(list(product(proto_vel, proto_pos)))

    def get_start_state(self) -> Tuple[float, float]:
        """Returns the starting state (velocity, position)."""
        return 0.0, np.random.uniform(-0.6, -0.4)

    def select_action(self, vel: float, pos: float):
        """Selects an action using epsilon-greedy."""
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(self.actions, key=lambda a: self.q(vel, pos, a))

    def next_reward_state(
        self, vel: float, pos: float, a: int
    ) -> Tuple[float, float, float]:
        """Returns the next state and reward given the current state and action."""
        vel += 0.001 * a - 0.0025 * np.cos(3 * pos)
        vel = np.clip(vel, *self.vel_min_max)

        pos += vel
        pos = np.clip(pos, *self.pos_min_max)
        if pos == self.pos_min_max[0]:
            vel = 0
        r = -1
        return vel, pos, r

    def terminal_condition(self, vel: float, pos: float, r: int) -> bool:
        """Returns True if the episode has terminated."""
        return pos >= 0.5

    def rbf_vec(self, vel: float, pos: float) -> np.ndarray:
        """Returns the normalized RBF vector for the given state."""
        s = np.array([vel, pos])
        vel_scale = self.vel_min_max[1] - self.vel_min_max[0]
        pos_scale = self.pos_min_max[1] - self.pos_min_max[0]
        dist_components = (s - self.proto_points) / (
            np.array([vel_scale, pos_scale]) * self.distance_scaler
        )
        radial_basis = np.exp(-(dist_components**2).sum(1))
        return radial_basis / radial_basis.sum()

    def q(self, vel: float, pos: float, a: int) -> float:
        """Returns the linearly-estimated Q-value for a given state and action."""
        return self.w[:, a + 1].dot(self.rbf_vec(vel, pos))

    def play_episode(self) -> Tuple[List]:
        """Applies N-Step Sarsa over a single episode."""

        vel, pos = self.get_start_state()
        a = self.select_action(vel, pos)
        states = [(vel, pos)]  # S_0
        rewards = []  # reward[4] is the reward at t=5
        actions = [a]  # A_0
        T = np.inf
        for t in count(0, 1):
            if t < T:
                # Note: below refers to the NEXT state/reward pair. Not that of t.
                vel, pos, r = self.next_reward_state(vel, pos, a)
                states.append((vel, pos))
                rewards.append(r)

                if self.terminal_condition(vel, pos, r):  # terminal
                    T = t + 1
                    actions.append(
                        None
                    )  # Otherwise actions and states won't have the same length.
                else:
                    a = self.select_action(vel, pos)
                    actions.append(a)
            tau = t - self.n + 1
            if tau >= 0:
                g = sum([(self.gamma**i) * r for i, r in enumerate(rewards[tau:])])
                if tau + self.n < T:
                    vel_curr, pos_curr = states[tau + self.n]
                    a_curr = actions[tau + self.n]
                    g += (self.gamma**self.n) * self.q(vel_curr, pos_curr, a_curr)
                vel_tau, pos_tau = states[tau]
                a_tau = actions[tau]
                qsa_tau = self.q(vel_tau, pos_tau, a_tau)
                self.w[:, a_tau + 1] += (
                    self.alpha * (g - qsa_tau) * self.rbf_vec(vel_tau, pos_tau)
                )
            if tau == T - 1:
                break
        return states, rewards

    def run(self, num_episodes: int) -> pd.DataFrame:
        """Runs the algorithm for a given number of episodes."""
        self.initialize()
        results = []
        for i in tqdm(range(num_episodes)):
            states, rewards = self.play_episode()
            df = pd.DataFrame(states, columns=["vel", "pos"]).assign(episode=i)
            results.append(df)
        self.results = pd.concat(results, axis=0)
        self.cost_to_go = self.get_cost_to_go()
        return self.results

    def get_vel_pos_plot_vecs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the velocity and position vectors for plotting."""
        vel_vec = np.linspace(*self.vel_min_max, self.n_plot_points)
        pos_vec = np.linspace(*self.pos_min_max, self.n_plot_points)
        return vel_vec, pos_vec

    def get_cost_to_go(self) -> pd.DataFrame:
        """Returns a DataFrame with the cost to go for each state. The cost to go is the negative of the
        max-action-value function. That is, it's the estimated 'cost' of the move to be made."""
        vel_vec, pos_vec = self.get_vel_pos_plot_vecs()
        pos_diff = pos_vec[1] - pos_vec[2] - 0.005
        vel_diff = vel_vec[1] - vel_vec[2] - 0.005
        df = (
            pd.DataFrame(list(product(vel_vec, pos_vec)), columns=["vel", "pos"])
            .assign(pos2=lambda df: df["pos"] - pos_diff)
            .assign(vel2=lambda df: df["vel"] - vel_diff)
        )
        df["cost_to_go"] = df.apply(
            lambda row: -max([self.q(row.vel, row.pos, a) for a in self.actions]),
            axis=1,
        )
        df["action"] = df.apply(
            lambda row: max(self.actions, key=lambda a: self.q(row.vel, row.pos, a)),
            axis=1,
        )
        return df

    def plot_episode_pos_over_time(
        self, episode: int, height: int = 300, width: int = 450
    ) -> alt.Chart:
        """
        Plots the position of the car over time for a given episode.
        """

        assert self.results is not None, "Must run the TD algorithm first."
        assert episode in self.results["episode"].unique(), "Invalid episode number."

        return self.results.query(f"episode=={episode}").pipe(
            lambda df: alt.Chart(df.reset_index())
            .mark_line()
            .encode(
                x=alt.X("index", title="Time Step"), y=alt.Y("pos", title="Position")
            )
            .properties(width=width, height=height, title=f"Episode {episode}")
        )

    def plot_cost_to_go(self, episode_trail_plot: int) -> alt.Chart:
        """
        Plots the cost to go (the negative of the max action value) for each state in the state space.

        Parameters
        ----------
        episode_trail_plot: The episode to plot the trail of the car over. If None, no trail is plotted. If -1, the
        last episode is plotted.

        """
        assert self.cost_to_go is not None, "Must run the TD algorithm first."
        chart = (
            alt.Chart(self.cost_to_go)
            .mark_rect(shape="square", color="None")
            .encode(
                y=alt.Y(
                    "vel",
                    title="Velocity",
                    scale=alt.Scale(
                        domain=[self.vel_min_max[0] - 0.01, self.vel_min_max[1] + 0.01]
                    ),
                ),
                y2="vel2",
                x=alt.X(
                    "pos",
                    title="Position",
                    scale=alt.Scale(
                        domain=[self.pos_min_max[0] - 0.02, self.pos_min_max[1] + 0.02]
                    ),
                ),
                x2="pos2",
                fill=alt.Color(
                    "cost_to_go",
                    title="Cost to Go",
                    scale=alt.Scale(scheme="darkmulti"),
                ),
            )
            .properties(width=500, height=500, title="Cost to Go")
        )
        if episode_trail_plot is not None:
            if episode_trail_plot == -1:
                episode_trail_plot = self.results["episode"].max()
            trail = (
                self.results.query(f"episode=={episode_trail_plot}")
                .reset_index()
                .pipe(
                    lambda df: alt.Chart(df)
                    .mark_trail(color="red")
                    .encode(
                        x="pos",
                        y="vel",
                        size=alt.Size("index", title="Time Step"),
                        order="index",
                    )
                )
            )
            chart = chart + trail
        return chart

    def plot_steps_per_episode(self, width: int = 450, height: int = 300) -> alt.Chart:
        """
        Plots the number of steps per episode. Since the episode ends when the car reaches the goal, this is a
        measure of the algo's performance as episodes are processed.
        """
        return (
            self.results.groupby("episode")
            .size()
            .to_frame("time_steps")
            .reset_index()
            .pipe(
                lambda df: alt.Chart(df)
                .mark_line()
                .encode(x="episode", y=alt.Y("time_steps", scale=alt.Scale(type="log")))
                .properties(width=width, height=height)
            )
        )

import pandas as pd
import numpy as np
import altair as alt
from tqdm import tqdm


def hexagonal_grid(n_points_x, n_points_y, max_x, max_y):
    """
    This will return points distributed over a hexagonal grid.
    """
    ratio = np.sqrt(3) / 2  # cos(60°)
    xv, yv = np.meshgrid(
        np.arange(n_points_x), np.arange(n_points_y), sparse=False, indexing="xy"
    )
    xv = xv * ratio
    xv[::2, :] += ratio / 2
    xv = xv * (max_x / xv.max())
    yv = yv * (max_y / yv.max())

    return np.array(
        [(xvii, yvii) for xvi, yvi in zip(xv, yv) for xvii, yvii in zip(xvi, yvi)]
    )


eval_points = hexagonal_grid(6, 10, 0.9, 0.9)
eval_points[:, 1] += 0.025
eval_points[:, 0] += 0.05


class WindyHighway:
    """
    Applies the REINFORCE algorithm to the Windy Highway game.
    """

    def __init__(
        self,
        alpha: float,
        distance_scaler: float,
        protos_per_dim: int,
        seed: int = 0,
        with_baseline: bool = False,
        reward_shift: float = 0,
        alpha_value: float = 0.1,
        distance_scaler_value: float = 0.01,
        protos_per_dim_value: int = 3,
    ):
        """
        alpha: The learning rate.
        distance_scaler: Scales distances between proto points and state. If this number is large, the distances between
            the state and the protopoints will shrink and the point will be considered close to *all* protopoints.
        protos_per_dim: The number of proto points per dimension. The total number of proto points is protos_per_dim^2.
        seed: The random seed.
        with_baseline: Whether to use a baseline.
        reward_shift: A constant amount to add to all rewards. This is what the baseline counteracts
        alpha_value: The learning rate for the value function (the baseline).
        distance_scaler_value: The distance scaler for the value function (the baseline).
        protos_per_dim_value: The number of proto points per dimension for the value function (the baseline).
        """
        self.alpha = alpha
        self.distance_scaler = distance_scaler
        self.protos_per_dim = protos_per_dim
        self.protos_per_dim_value = protos_per_dim_value
        self.with_baseline = with_baseline
        self.reward_shift = reward_shift
        self.alpha_value = alpha_value
        self.distance_scaler_value = distance_scaler_value

        np.random.seed(seed)

        self.seed = seed
        self.x_min_max = [
            0,
            1,
        ]  # DON"T CHANGE. The code assumes these hardcoded values.
        self.y_min_max = [
            0,
            1,
        ]  # DON"T CHANGE. The code assumes these hardcoded values.
        self.actions = ["left", "right", "up"]

        self.move_min_max = [
            0.04,
            0.06,
        ]  # Min/max component distance of state transition. This can be changed.
        self.max_wind_strength = 0.02  # The maximum wind strength. This can be changed.
        self.G0s = []  # T=0 returns over all episodes.
        self.initialize()

    def initialize(self):
        # It has dimension protos_per_dim^2-by-2. Column 0 is x and column 1 is y.
        self.proto_points = (
            hexagonal_grid(self.protos_per_dim, self.protos_per_dim, 0.95, 0.95) + 0.025
        )
        # There is a length 2 parameter vector for each proto point. The 2 values correspond to the
        # left and right action. The up action is determined by the sum-to-1 constraint.
        self.theta = np.zeros((self.proto_points.shape[0], 2))
        if self.with_baseline:
            self.proto_points_value = (
                hexagonal_grid(
                    self.protos_per_dim_value, self.protos_per_dim_value, 0.95, 0.95
                )
                + 0.025
            )
            self.w = np.zeros((self.proto_points_value.shape[0]))

    def rbf_vec(self, s_xy, baseline=False):
        # s_xy is a (x, y) 1-by-2 tensor/array.
        if baseline:
            proto_pints = self.proto_points_value
            distance_scaler = self.distance_scaler_value
        else:
            proto_pints = self.proto_points
            distance_scaler = self.distance_scaler

        dist_components = (s_xy - proto_pints) / distance_scaler
        radial_basis = np.exp(-(dist_components**2).sum(1))
        return radial_basis / radial_basis.sum()  # Normalize

    def get_reward(self, s_xy: np.ndarray):
        # Note: s_xy is the state we're transitioning to.
        return np.sin(s_xy[0] * 2 * np.pi) + self.reward_shift

    def get_rand_delta(self):
        return np.random.uniform(*self.move_min_max)

    def next_reward_state(self, s_xy, action):
        # Given the current state and action, return the reward and next state.
        x, y = s_xy
        y += self.get_rand_delta()  # We always move up by a random amount.
        if action == "left":
            x -= self.get_rand_delta()
        elif action == "right":
            x += self.get_rand_delta()

        # # Apply leftward wind.
        # When w is positive, wind is leftward, which is always away from the high reward or towards the negative reward.
        w = np.cos(s_xy[0] * 2 * np.pi) * self.max_wind_strength
        x -= w

        s_xy_next = np.array([x, y]).clip(min=0, max=1)

        r = self.get_reward(s_xy_next)
        return s_xy_next, r

    def terminal_condition(self, s_xy):
        return s_xy[1] >= 1.0

    def get_start_state(self):
        x = np.random.uniform() * 0.2 + 0.4
        return np.array([x, 0.0])

    def play_episode(self):
        s_xy = self.get_start_state()
        states = [s_xy]
        actions = []
        rewards = [np.nan]
        while not self.terminal_condition(s_xy):
            action_probs = self.action_probs(s_xy)
            action = np.random.choice(self.actions, size=1, p=action_probs)
            actions.append(action)
            s_xy, r = self.next_reward_state(s_xy, action)
            states.append(s_xy)
            rewards.append(r)

        returns = []
        for i in range(len(rewards) - 1):
            returns.append(sum(rewards[i + 1 :]))

        returns = returns + [np.nan]

        return states, actions + [np.nan], rewards, returns

    def logits(self, s_xy: np.array):
        radial_basis = self.rbf_vec(s_xy)
        logits = (self.theta * radial_basis[..., np.newaxis]).sum(0)
        return np.concatenate([logits, np.array([0.0])])

    def action_probs(self, s_xy: np.array):
        # s_xy is a (x, y) 1-by-2 np.array.
        exp = np.exp(self.logits(s_xy))
        return exp / exp.sum()

    def jacobian_softmax(self, s_xy):
        # Jacobian of the softmax function w.r.t. the logits.
        probs = self.action_probs(s_xy)
        return np.diag(probs) - np.outer(probs, probs)

    def jacobian_logits(self, s_xy):
        # Jacobian of the logits w.r.t. to theta.
        jac_logits = np.zeros((3, *self.theta.shape), dtype=float)
        for i in range(2):
            jac_logits[i, :, i] = self.rbf_vec(s_xy)
        return jac_logits

    def gradient(self, s_xy: np.array, action: str):
        """
        This returns the gradient of the action probability with respect to the parameters, theta.
        """
        jac_logits = self.jacobian_logits(s_xy)
        jac_softmax_wrt_a = self.jacobian_softmax(s_xy)[self.actions.index(action)]
        return (jac_softmax_wrt_a.reshape(3, 1, 1) * jac_logits).sum(
            0
        )  # 3x1x1 * 3x25x2

    def value_estimate(self, s_xy):
        radial_basis = self.rbf_vec(s_xy, baseline=True)
        return (self.w * radial_basis).sum()

    def gradient_value_estimate(self, s_xy):
        return self.rbf_vec(s_xy, baseline=True)

    def update_params(self, states, actions, rewards, returns):

        for s_xy, a, _, Gt in zip(states[:-1], actions, rewards, returns):
            a_prob = self.action_probs(s_xy)[self.actions.index(a)]
            grad = self.gradient(s_xy, a) / a_prob

            if self.with_baseline:
                grad_v = self.gradient_value_estimate(s_xy)
                delta = Gt - self.value_estimate(s_xy)
                self.w += self.alpha_value * delta * grad_v
                self.theta += self.alpha * delta * grad
            else:
                self.theta += self.alpha * Gt * grad

    def episode_as_df(self, states, actions, rewards, returns):
        state_x = [xy[0].item() for xy in states]
        state_y = [xy[1].item() for xy in states]

        return pd.DataFrame(
            dict(
                state_x=state_x,
                state_y=state_y,
                actions=actions,
                rewards=rewards,
                returns=returns,
            )
        )

    def action_probs_over_eval_points(self):

        rows = []
        for x, y in eval_points:
            rows.append((x, y, *self.action_probs(np.array([x, y]))))

        return pd.DataFrame(rows, columns=["x", "y", *self.actions])

    def run(self, num_episodes: int):
        for _ in tqdm(range(num_episodes)):
            states, actions, rewards, returns = self.play_episode()
            self.G0s.append(returns[0])
            self.update_params(states, actions, rewards, returns)

    def get_G0_df(self, window):
        return pd.DataFrame(
            dict(episode=range(len(self.G0s)), returns=self.G0s)
        ).assign(rolling_ave=lambda df: df["returns"].rolling(window=window).mean())

    def plot_returns(self, window, width=900, height=300, y_min=None, y_max=None):
        """
        Plot the G0 returns for each episode over the run of the algorithm.

        window: the window size for the rolling average.
        width: the width of the plot.
        height: the height of the plot.
        y_min: the minimum value of the y-axis.
        y_max: the maximum value of the y-axis.
        """

        df = self.get_G0_df(window=window)
        if y_max is not None and y_min is None:
            y_min = 0

        c1 = (
            alt.Chart(df)
            .mark_line(strokeWidth=0.5, opacity=0.5)
            .encode(
                x="episode",
                y="returns"
                if not y_max
                else alt.Y("returns", scale=alt.Scale(domain=(y_min, y_max))),
            )
            .properties(width=width, height=height)
        )
        c2 = (
            alt.Chart(df)
            .mark_line(strokeWidth=3, color="blue")
            .encode(x="episode", y="rolling_ave")
        )

        return c1 + c2

    def policy_probs_df(self, n_evals):
        xy_eval = hexagonal_grid(n_evals, n_evals, 1, 1)
        action_probs = [self.action_probs(s_xy)[np.newaxis, :] for s_xy in xy_eval]
        df = pd.DataFrame(np.concatenate(action_probs, axis=0), columns=self.actions)
        df["state_x"] = xy_eval[:, 0]
        df["state_y"] = xy_eval[:, 1]
        return df

    def plot_policy_probs(self, n_evals, width=500, height=500):

        """
        Plot the policy probabilities over the state space.

        n_evals: the number of state-points to evaluate along each dimension. The total number of state points plotted
            with be n_evals^2.
        width: the width of the plot.
        height: the height of the plot.
        """

        df = self.policy_probs_df(n_evals)
        max_arrow_length = (df["state_x"].iloc[2] - df["state_x"].iloc[1]) / 2.5
        min_length = max_arrow_length * 0.2

        df = df.assign(
            state_x_left=lambda df: df["state_x"]
            - (df["left"] * max_arrow_length).clip(lower=min_length),
            state_x_right=lambda df: df["state_x"]
            + (df["right"] * max_arrow_length).clip(lower=min_length),
            state_y_up=lambda df: df["state_y"]
            + (df["up"] * max_arrow_length).clip(lower=min_length),
        )

        def get_lines_df(df, which):
            if which in ["left", "right"]:
                df_out = []
                for i, row in df.iterrows():
                    df_out.append([row["state_x"], row["state_y"], "W" + str(i)])
                    df_out.append(
                        [row[f"state_x_{which}"], row["state_y"], "W" + str(i)]
                    )

            else:
                df_out = []
                for i, row in df.iterrows():
                    df_out.append([row["state_x"], row["state_y"], "W" + str(i)])
                    df_out.append([row[f"state_x"], row["state_y_up"], "W" + str(i)])

            return pd.DataFrame(df_out, columns=["state_x", "state_y", "which"])

        df_left = get_lines_df(df, "left")
        df_right = get_lines_df(df, "right")
        df_up = get_lines_df(df, "up")

        c1 = (
            alt.Chart(df_left)
            .mark_line(opacity=1, color="blue")
            .encode(
                x=alt.X("state_x", title="state x"),
                y=alt.Y("state_y", title="state y"),
                detail="which",
            )
            .properties(width=width, height=height)
        )
        c1_end = (
            alt.Chart(df)
            .mark_point(
                color="blue",
                shape="triangle",
                fill="blue",
                strokeWidth=1,
                opacity=1,
                angle=270,
            )
            .encode(x="state_x_left", y="state_y")
        )
        c1 = c1 + c1_end

        c2 = (
            alt.Chart(df_right)
            .mark_line(opacity=1, color="green")
            .encode(x="state_x", y="state_y", detail="which")
        )
        c2_end = (
            alt.Chart(df)
            .mark_point(
                color="green",
                shape="triangle",
                fill="green",
                strokeWidth=1,
                opacity=1,
                angle=90,
            )
            .encode(x="state_x_right", y="state_y")
        )
        c2 = c2 + c2_end

        c3 = (
            alt.Chart(df_up)
            .mark_line(opacity=1, color="red")
            .encode(x="state_x", y="state_y", detail="which")
        )
        c3_end = (
            alt.Chart(df)
            .mark_point(
                color="red",
                shape="triangle",
                fill="red",
                strokeWidth=1,
                opacity=1,
                angle=0,
            )
            .encode(
                x=alt.X("state_x", scale=alt.Scale(domain=[-0.05, 1.05])),
                y=alt.Y("state_y_up", scale=alt.Scale(domain=[-0.05, 1.05])),
            )
        )
        c3 = c3 + c3_end

        center = (
            alt.Chart(df)
            .mark_point(opacity=1, strokeWidth=0, size=25, fill="black")
            .encode(x="state_x", y="state_y")
        )
        return c1 + c2 + c3 + center

    def plot_trajectory(self, width=500, height=500):
        """
        Plot the trajectory of the agent over a fresh episode.

        width: the width of the plot.
        height: the height of the plot.
        """
        states, actions, rewards, returns = self.play_episode()
        df = self.episode_as_df(states, actions, rewards, returns)
        c1 = (
            alt.Chart(df)
            .mark_point(fill="darkblue", size=50, opacity=0.9, strokeWidth=0)
            .encode(x=alt.X("state_x", scale=alt.Scale(domain=[0, 1])), y="state_y")
            .properties(width=width, height=height)
        )
        c2 = (
            alt.Chart(
                df.assign(
                    returns=lambda df: [
                        f"{s:.4}" for s in df["returns"].astype(str).values
                    ]
                )
            )
            .mark_text(color="black", fontSize=14, dx=-20)
            .encode(x="state_x", y="state_y", text="returns")
        )
        return c1 + c2

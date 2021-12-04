from abc import ABC, abstractmethod
import numpy as np
import altair as alt
import pandas as pd
from scipy.stats import binom


class BaseGame(ABC):

    """
    Betting base class to demonstrate the Kelly Criteria. Subclassing this class allows you to explore different
    bettering strategies. Subclasses override the _strategy method to implement whatever strategy they are intended to
    execute.
    """

    def __init__(
        self,
        prob_heads: float,
        payout_ratio: float,
        N_flips: int,
        N_games: int,
        initial_wealth: float = 50,
    ):

        """
        Initialization for all BaseGame-s - don't override it! The game is determined by the probability of heads,
        the payout ratio of payout-to-wager for a bet on heads and the number of flips.

        Note : Without lose of generaltiy, we assume every wager is on heads. This is because the game is such that it's
        always best to bet on one outcome every time.

        Parameters
        ----------
        prob_heads : true probability of heads on each flip
        payout_ratio : if you wager wager-dollars on a flip for heads, you'll receive wager*payout_ratio, otherwise
        you'll lose wager.
        N_flips : The number of flips involved in one play of the game.
        N_games : The number of games you'll play to test your strategy
        initial_wealth : The amount of wealth you begin each game with.
        """

        assert 0 < prob_heads < 1, "Prob(Heads) needs to be within (0, 1)"
        assert 0 < payout_ratio, "Payout ratio must be positive."
        assert 0 < initial_wealth, "Initial wealth must be positive."
        assert 1 <= N_flips and isinstance(
            N_flips, int
        ), "N_flips needs to be a positive integer"
        assert 1 <= N_games and isinstance(
            N_games, int
        ), "N_flips needs to be a positive integer"

        self.prob_heads = prob_heads
        self.payout_ratio = payout_ratio
        self.N_flips = N_flips
        self.N_games = N_games
        self.initial_wealth = initial_wealth

        # Create all flips
        self.flips = self._make_all_flips()

        self.wealth_by_games = (
            None  # Will store the wealth you have over N_flips and N_games
        )

    def _make_all_flips(self) -> np.array:
        """Creates all flips for all games. A value of 1 indicates a heads, 0 indicates tails."""
        return np.random.binomial(
            n=1, p=self.prob_heads, size=self.N_flips * self.N_games
        ).reshape(self.N_games, self.N_flips)

    def kelly_bet(self, prob_heads: float, payout_ratio: float) -> float:
        """
        Returns the optimal kelly bet. It can never be above 1 because that would involve betting more than your
        wealth. (In other set ups, that would be a leveraged bet). It can't be less than 0, because that would indicate
        a bet on tails, and we're not supporting that in this game.

        Note : this is only intended for use in the base classes KellyGame and KellyWithAnEstimate
        """
        return min(max(prob_heads + (prob_heads - 1) / payout_ratio, 0), 1)

    @abstractmethod
    def _strategy(self, flips_so_far: np.array, W: float) -> float:
        """
        Override this to represent your strategy. This should return a positive number which is your dollar wager and
        it's a function of your current wealth (W) and the flips you've seen so far (flips_so_far).

        Parameters
        ----------
        flips_so_far : the outcomes of the flips so far
        W : current wealth after the flips_so_far
        """
        pass

    def strategy(self, flips_so_far: np.array, W: float) -> float:
        """wrapper for _strategy"""
        wager = self._strategy(flips_so_far=flips_so_far, W=W)
        assert 0 <= wager, "Wager must be nonnegative"
        return wager

    def play_game(self, g: int):
        """
        Play the g-th game according to the strategy. Returns an array giving your wealth after each flip.
        """

        flips_g = self.flips[g, :]

        wealth = [self.initial_wealth]
        for i, o in enumerate(flips_g):
            wager = self.strategy(flips_so_far=flips_g[:i], W=wealth[-1])
            if wealth[-1] <= 0:
                # If you have no wealth, you can't wager anything, so you must have no wealth going forward.
                wealth.append(0)
                continue
            if wager > wealth[-1]:
                # If your wager is greater than your current wealth, wager your wealth.
                wager = wealth[-1]
            if o:
                # Heads, you win wager * self.payout_ratio
                wealth.append(wealth[-1] + wager * self.payout_ratio)
            else:
                # Tails, you lose your wager.
                wealth.append(wealth[-1] - wager)

        return np.array(wealth)

    def get_n_games(self, n_games: int) -> np.array:
        """Returns wealth trajectories for n_games. Playing many games can be slow, since it involves for-looping over
        bets/games. So, if you've already played n_games with a previous call, this will return just those games. If you
        have played less than n_games, it'll play whatever the difference of games is, append that to the store of games
        and then return n_games"""
        if self.wealth_by_games is None:
            n_games_so_far = 0
        else:
            n_games_so_far = self.wealth_by_games.shape[0]

        wealths = [self.wealth_by_games] if self.wealth_by_games is not None else []
        for g in range(n_games_so_far, n_games):
            wealths.append(self.play_game(g).reshape(1, -1))

        self.wealth_by_games = np.concatenate(wealths, axis=0)
        return self.wealth_by_games[:n_games, :]

    def get_n_game_df(self, n_games: int):
        """Does the same thing as get_n_games, except returns the result as a long dataframe, which is useful for
        plotting."""
        return (
            pd.DataFrame(self.get_n_games(n_games).T)
            .assign(flip=range(self.N_flips + 1))
            .melt(id_vars=["flip"])
            .rename(columns=dict(variable="game", value="wealth"))
        )

    def get_growth_rates(self, final_wealths: np.array) -> np.array:
        """Calculates growth rates. """
        return (final_wealths / self.initial_wealth) ** (1 / self.N_flips) - 1

    def plot_games(
        self, n_games: int, log: bool = False, opacity: float = 0.1
    ) -> alt.Chart:
        """
        Returns a chart showing wealth trajectories over multiple games

        Parameters
        ----------
        n_games : the name of game trajectories to show.
        log : should the plot be on a log scale? Careful, it can't show 0 values (so they are clipped at a penny)
        opacity : the opacity to apply to the trajectory. Should depend on how many trajectories you are showing.
        """
        games_df = self.get_n_game_df(n_games)
        encode_args = dict(
            x=alt.X("flip", title="Flip"),
            y=alt.Y("wealth", title="Wealth"),
            color="game:N",
        )

        if log:
            encode_args["y"] = alt.Y(
                "wealth", scale=alt.Scale(type="log"), title="Wealth"
            )
            games_df = games_df.assign(wealth=lambda df: df.wealth.clip(lower=0.01))

        return alt.Chart(games_df).mark_line(opacity=opacity).encode(**encode_args)

    def get_binomial_distribution(self, n_games: int = None):
        """For exact calculations on final wealths, the binomial distribution over 100 flips can be useful. We include
        n_games because we'll actually use it to form an 'exact' simulation (which is easier to pass around). This will
        give a count column we can use."""
        df = pd.DataFrame(
            [
                (k, binom.pmf(k, self.N_flips, self.prob_heads))
                for k in range(self.N_flips + 1)
            ],
            columns=["heads_count", "probability"],
        )
        if n_games is not None:
            df = df.assign(counts=lambda df: round(df.probability * n_games))
        return df

    def simulate_growth_rates(self, n_games: int) -> np.array:
        """Returns monte carlo simulations of final growth rates over n_games. Careful, this can be slow for large
        n_games"""
        return self.get_growth_rates(self.get_n_games(n_games)[:, -1])

    def plot_growth_rate_distribution(
        self, n_games: np.array, min_max_growth_rate: tuple, step_size: float = 0.005,
    ) -> np.array:
        """
        Returns a chart showing a normalized histogram over n_games of simulates growth rates.

        Parameters
        ----------
        n_games : number of games to simulate
        min_max_growth_rate : the min/max range of the growth rates to plot
        step_size : the width of the bins in the histogram, in units of the growth rate.
        """

        return (
            alt.Chart(
                pd.DataFrame(
                    self.simulate_growth_rates(n_games), columns=["growth_rates"]
                )
            )
            .transform_joinaggregate(total="count(*)")
            .transform_calculate(pct="1 / datum.total")
            .mark_bar()
            .encode(
                alt.X(
                    "growth_rates",
                    bin=alt.Bin(extent=min_max_growth_rate, step=step_size),
                    title="Growth Rate",
                    axis=alt.Axis(format="%"),
                ),
                alt.Y("sum(pct):Q", axis=alt.Axis(format="%"), title="Probability"),
            )
        )


class Kelly(BaseGame):
    def _strategy(self, flips_so_far: np.array, W: float) -> float:
        return W * self.kelly_bet(self.prob_heads, self.payout_ratio)

    def simulate_growth_rates(self, n_games: int) -> np.array:
        """For this strategy, we can actually create simulations of growth which reflect their true probability exactly."""

        bin_dist = self.get_binomial_distribution(n_games)
        kelly_f = self.kelly_bet(self.prob_heads, self.payout_ratio)
        final_wealths = []
        for _, row in bin_dist.iterrows():
            final_wealths.extend(
                [
                    self.initial_wealth
                    * ((1 + self.payout_ratio * kelly_f) ** row.heads_count)
                    * ((1 - kelly_f) ** (self.N_flips - row.heads_count))
                ]
                * int(row.counts)
            )

        return self.get_growth_rates(np.array(final_wealths))

    def _exp_growth_rates_by_perc_wager(
        self, min_max_percent_wager: tuple, n_evals: int
    ) -> pd.DataFrame:
        """
        Returns a dataframe giving the fixed percent wager and expected growth rate from that.

        Parameters
        ----------
        min_max_percent_wager : tuple giving the min and max of the fixed percent wager
        n_evals : the number of points to determing the expected growth rate between the min and max

        """

        bin_dist_df = self.get_binomial_distribution()
        results = []
        for f in np.linspace(*min_max_percent_wager, n_evals):
            final_wealths = (
                self.initial_wealth
                * ((1 + self.payout_ratio * f) ** bin_dist_df.heads_count)
                * ((1 - f) ** (self.N_flips - bin_dist_df.heads_count))
            )
            growth_rates = self.get_growth_rates(final_wealths)
            results.append((f, (growth_rates * bin_dist_df.probability).sum()))
        return pd.DataFrame(results, columns=["percent_wager", "expected_growth_rate"])

    def plot_exp_growth_rates_by_perc_wager(
        self, min_max_percent_wager: tuple = (0.0, 0.4), n_evals: int = 400
    ) -> alt.Chart:

        exp_growth_rates = self._exp_growth_rates_by_perc_wager(
            min_max_percent_wager=min_max_percent_wager, n_evals=n_evals
        )

        curve = (
            alt.Chart(exp_growth_rates)
            .mark_line(strokeWidth=4)
            .encode(
                x=alt.X("percent_wager", title="Fixed Percent Wager"),
                y=alt.Y("expected_growth_rate", title="Expected Growth Rate"),
            )
        )

        kelly_line_df = pd.DataFrame(
            dict(percent_wager=[self.kelly_bet(self.prob_heads, self.payout_ratio)])
        )
        kelly_line = (
            alt.Chart(kelly_line_df)
            .mark_rule(strokeWidth=2, strokeDash=[4, 4])
            .encode(x="percent_wager")
        )

        return (curve + kelly_line).properties(
            title="Expected Growth Rate vs Fixed Percent Wager, including Kelly Optimal Bet"
        )


class ConstantDollar(BaseGame):

    """Strategy is to bet a constant dollar amount no matter what."""

    amount = 5

    def _strategy(self, flips_so_far: np.array, W: float) -> float:
        """Always bet self.amount"""
        return self.amount


class BetLikeADummy(BaseGame):

    """Strategy is to bet big when you see a string of tails. Not a good strategy, but it is something people do!"""

    wager_big = 50
    wager_small = 5

    def _strategy(self, flips_so_far: np.array, W: float) -> float:

        if len(flips_so_far) > 2 and flips_so_far[-3:].sum() == 0:
            return self.wager_big
        else:
            return self.wager_small


class KellyWithAnEstimate(BaseGame):

    """Employ the kelly strategy, but we will use an estimate of the probability of heads rather than the true value."""

    def _strategy(self, flips_so_far: np.array, W: float) -> float:

        if len(flips_so_far) < 10:
            # For the first 10 flips, don't bet anything. We are just collecting data.
            return 0

        # Form the estimate. We cap it at extra estimates to avoid oversized bet (also something you'd be likely to do in reality)
        p = max(min(flips_so_far.mean(), 0.999), 0.001)

        return W * self.kelly_bet(p, self.payout_ratio)

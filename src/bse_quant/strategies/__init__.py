"""All strategies, numbered after the idea list. Each one is a single function:

    strategy(data, setting=default, ...) -> Signal

The keyword arguments are exactly the settings you can change in config.toml.
"""

from bse_quant.strategies.s01_time_series_momentum import time_series_momentum
from bse_quant.strategies.s02_moving_average_crossover import moving_average_crossover
from bse_quant.strategies.s03_donchian_breakout import donchian_breakout
from bse_quant.strategies.s04_cross_sectional_momentum import cross_sectional_momentum
from bse_quant.strategies.s05_volatility_scaled_trend import volatility_scaled_trend
from bse_quant.strategies.s06_trend_strength_filter import trend_strength_filter
from bse_quant.strategies.s07_short_term_reversal import short_term_reversal
from bse_quant.strategies.s08_rsi_reversion import rsi_reversion
from bse_quant.strategies.s09_bollinger_reversion import bollinger_reversion
from bse_quant.strategies.s10_gap_fade import gap_fade
from bse_quant.strategies.s11_internal_bar_strength import internal_bar_strength
from bse_quant.strategies.s12_pairs_cointegration import pairs_cointegration
from bse_quant.strategies.s13_kalman_pairs import kalman_pairs
from bse_quant.strategies.s14_pca_residual_reversion import pca_residual_reversion
from bse_quant.strategies.s15_lead_lag import lead_lag
from bse_quant.strategies.s16_volatility_regime_switch import volatility_regime_switch
# 17 (volatility targeting) is applied to the whole portfolio, see engine.volatility_scale
from bse_quant.strategies.s18_volume_shock import volume_shock
from bse_quant.strategies.s19_obv_confirmation import obv_confirmation
from bse_quant.strategies.s20_range_compression_breakout import range_compression_breakout
from bse_quant.strategies.s21_overnight_intraday import overnight_intraday
from bse_quant.strategies.s22_calendar_effects import calendar_effects

STRATEGIES = {
    "time_series_momentum": time_series_momentum,
    "moving_average_crossover": moving_average_crossover,
    "donchian_breakout": donchian_breakout,
    "cross_sectional_momentum": cross_sectional_momentum,
    "volatility_scaled_trend": volatility_scaled_trend,
    "trend_strength_filter": trend_strength_filter,
    "short_term_reversal": short_term_reversal,
    "rsi_reversion": rsi_reversion,
    "bollinger_reversion": bollinger_reversion,
    "gap_fade": gap_fade,
    "internal_bar_strength": internal_bar_strength,
    "pairs_cointegration": pairs_cointegration,
    "kalman_pairs": kalman_pairs,
    "pca_residual_reversion": pca_residual_reversion,
    "lead_lag": lead_lag,
    "volatility_regime_switch": volatility_regime_switch,
    "volume_shock": volume_shock,
    "obv_confirmation": obv_confirmation,
    "range_compression_breakout": range_compression_breakout,
    "overnight_intraday": overnight_intraday,
    "calendar_effects": calendar_effects,
}

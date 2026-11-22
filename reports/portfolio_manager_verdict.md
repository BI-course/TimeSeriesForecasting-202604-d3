# Portfolio Manager Verdict: NFLX Multi-Horizon Forecasting

## Executive assessment

The forecasting models tested do not demonstrate an improvement over a
naive no-change forecast for Netflix (NFLX). Across the 1-, 5-, 21- and
63-trading-day horizons, the naive benchmark recorded the lowest price
RMSE. The classical Simple Exponential Smoothing (SES) model was almost
identical to the benchmark, while the Ridge and Gradient Boosting models
generally produced larger errors. The evidence therefore does not support
using the more complex models in preference to the naive forecast.

## Evidence and model comparison

The evaluation used 2,981 NFLX trading-day observations from 2 January 2015
to 9 November 2026, together with seven external predictor series and
47 engineered features. Models were assessed using expanding-window
walk-forward validation, with an initial training period of 1,000
observations and test blocks of up to 63 observations. The comparison
included a naive no-change benchmark, SES, Drift, Ridge regression and
Gradient Boosting.

At the 63-day horizon, the naive model achieved a price RMSE of $12.9058
and MAE of $9.8181. SES was marginally worse (RMSE $12.9078), while Ridge
recorded $13.7767 and Gradient Boosting $17.2119. The same ranking,
with Naive having the lowest RMSE, was observed at all four horizons.
The negative out-of-sample return R-squared values for the learned models
also provide no evidence of useful improvement over the benchmark.

Diebold-Mariano tests compared each candidate's squared price forecast
errors with those of Naive, using a Newey-West HAC variance estimate to
account for serial dependence. All estimated mean loss differences were
positive, indicating higher average squared errors than Naive. SES, Drift
and Ridge were not significantly different from Naive at the 5% level.
Gradient Boosting was significantly worse at the 5- and 21-day horizons.
At 63 days its result was also worse on average, although the p-value
(0.0563) narrowly missed the 5% significance threshold.

## Confidence in the 63-day forecast

Confidence in the 63-day forecast should be low. Its naive benchmark RMSE
was $12.9058, and none of the tested alternatives reduced that error.
The Ridge model's directional accuracy of approximately 66.2% should not
be interpreted in isolation: it did not translate into lower price error
or positive out-of-sample return R-squared. In addition, 63-day forecast
targets overlap, making forecast errors dependent; the HAC adjustment
addresses this dependence approximately, rather than eliminating all
uncertainty.

The models' hyperparameters were carried over rather than optimised
separately for each forecast horizon. Further tuning and independent
validation could be explored, but the current evidence does not justify
relying on the 63-day machine-learning forecast for a portfolio decision.
The naive forecast is the strongest benchmark in this experiment, and
any investment decision should use additional analysis and risk controls.

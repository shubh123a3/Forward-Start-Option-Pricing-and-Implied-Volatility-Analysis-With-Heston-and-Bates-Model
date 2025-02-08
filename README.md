# Forward-Start Option Pricing and Implied Volatility Analysis With Heston and Bates Model

## Overview
This project implements the pricing of forward-start options and analyzes implied volatility using the Heston and Bates models. It provides a Streamlit application for user-friendly interaction with model parameters.

## Features
- **Heston Model:** Captures stochastic volatility effects.
- **Bates Model:** Extends Heston by incorporating jumps.
- **Forward-Start Option Pricing:** Valuation of options that begin at a future date.
- **Implied Volatility Analysis:** Visualizing the effect of model parameters on volatility.

## Heston Model
The Heston model describes the dynamics of an asset price \( S_t \) with stochastic volatility \( v_t \) as:

\[
\begin{aligned}
    dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW_t^S, \\
    dv_t &= \kappa (\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^v,
\end{aligned}
\]

where:
- \( \mu \) is the drift rate.
- \( \kappa \) is the mean reversion speed.
- \( \theta \) is the long-term variance.
- \( \sigma \) is the volatility of volatility.
- \( W_t^S \) and \( W_t^v \) are two correlated Wiener processes with correlation \( \rho \).

## Bates Model
The Bates model extends Heston by adding jump components:

\[
\begin{aligned}
    dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW_t^S + S_t (e^{J} - 1) dN_t, \\
    dv_t &= \kappa (\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^v.
\end{aligned}
\]

where:
- \( J \) represents the jump size, usually normally distributed as \( J \sim \mathcal{N}(\mu_J, \sigma_J^2) \).
- \( N_t \) is a Poisson process with intensity \( \lambda_J \), dictating the frequency of jumps.

## Forward-Start Option Pricing
A forward-start option has a strike set at a future date \( T_0 \), usually defined as:

\[
K = S_{T_0} e^{m}
\]

where \( m \) is a predetermined fraction. The Black-Scholes price for a forward-start call option is given by:

\[
C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2),
\]

where:

\[
\begin{aligned}
    d_1 &= \frac{\ln \left( \frac{S_0}{K} \right) + \left( r - q + \frac{\sigma^2}{2} \right) T}{\sigma \sqrt{T}}, \\
    d_2 &= d_1 - \sigma \sqrt{T}.
\end{aligned}
\]

## Streamlit App
The Streamlit application provides:
- Interactive parameter tuning for both models.
- Visualization of implied volatility surfaces.
- Forward-start option pricing output.

## Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```
Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage
1. Choose between Heston or Bates model.
2. Adjust parameters using sliders.
3. Observe the effect on implied volatility.
4. Compute forward-start option prices.

## References
- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility.
- Bates, D. S. (1996). Jumps and Stochastic Volatility in Exchange Rate Models.

---

This project serves as a valuable tool for understanding the impact of stochastic volatility and jump processes on option pricing.


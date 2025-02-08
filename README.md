Forward-Start Option Pricing and Implied Volatility Analysis with Heston and Bates Models

## Overview

This project explores forward-start option pricing using the Heston and Bates models. It provides an interactive Streamlit dashboard to analyze implied volatility (IV) behavior for short-term and long-term options and observe the impact of jump parameters. The models are implemented in Python using NumPy, SciPy, and Matplotlib.

## Features

- **Heston Model Implementation**: Computes option prices and implied volatility using the stochastic volatility model.
- **Bates Model Implementation**: Extends the Heston model by incorporating jump diffusion effects.
- **Forward-Start Option Pricing**: Evaluates forward-start options under stochastic volatility.
- **Implied Volatility Analysis**: Compares short-term and long-term IV surfaces for different parameter settings.
- **Interactive Streamlit Dashboard**: Visualizes results dynamically based on user input.

## Mathematical Formulation

### 1. Heston Model

The Heston stochastic volatility model is given by:

$$
    dS_t = \mu S_t dt + \sqrt{V_t} S_t dW_t^S
$$

$$
    dV_t = \kappa (\theta - V_t) dt + \sigma \sqrt{V_t} dW_t^V
$$

where:

- \(S_t\) is the underlying asset price.
- \(V_t\) is the stochastic variance.
- \(\mu\) is the drift of the asset price.
- \(\kappa\) is the mean-reversion rate of variance.
- \(\theta\) is the long-term mean variance.
- \(\sigma\) is the volatility of variance (vol of vol).
- \(\rho\) is the correlation between the Brownian motions.

#### Characteristic Function of Heston Model

The characteristic function of the Heston model is given by:

$$
    \phi(u) = \exp\left( C(u) + D(u) V_0 + i u \ln S_0 \right)
$$

where:

$$
    C(u) = \frac{\kappa \theta}{\sigma^2} \left[ (\kappa - i \rho \sigma u + d) t - 2 \ln \left( \frac{1 - g e^{dt}}{1 - g} \right) \right]
$$

$$
    D(u) = \frac{(\kappa - i \rho \sigma u + d) (1 - e^{dt})}{\sigma^2 (1 - g e^{dt})}
$$

where \(d\) and \(g\) are functions of \(u, \kappa, \sigma, \theta\).

### 2. Bates Model (Heston + Jumps)

The Bates model extends the Heston model by adding jump components:

$$
    dS_t = \mu S_t dt + \sqrt{V_t} S_t dW_t^S + (e^J - 1) S_t dN_t
$$

where \(J\) represents the jump process following a Poisson process:

$$
    P(N_t = k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}
$$

with \(J \sim \mathcal{N}(\mu_J, \sigma_J^2)\).

The Bates characteristic function extends the Heston model by adding:

$$
    \phi_{Bates}(u) = \phi_{Heston}(u) \times \exp \left( \lambda t \left( e^{i u \mu_J - \frac{1}{2} u^2 \sigma_J^2} - 1 \right) \right)
$$

where \(\lambda\) is the jump intensity, \(\mu_J\) is the mean jump size, and \(\sigma_J\) is the jump volatility.

### 3. Forward-Start Option Pricing

A forward-start option is an option where the strike is determined at a future date \(T_1\), with maturity at \(T_2\). The price is given by:

$$
    C_{FSO} = E_Q \left[ e^{-r (T_2 - T_0)} \max(S_{T_2} - K_{T_1}, 0) \right]
$$

where \(C_{FSO}\) is the Black-Scholes price of the option with forward-start parameters.

For the Black-Scholes model, the forward-start option price is:

$$
    C_{FSO} = S_0 e^{r T_1} N(d_1) - K e^{-r (T_2 - T_1)} N(d_2)
$$

where:

$$
    d_1 = \frac{\ln (S_0 / K) + (r + \frac{1}{2} \sigma^2)(T_2 - T_1)}{\sigma \sqrt{T_2 - T_1}}
$$

$$
    d_2 = d_1 - \sigma \sqrt{T_2 - T_1}
$$

## Installation

To run this project locally:

```sh
# Clone the repository
git clone https://github.com/shubh123a3/Forward-Start-Option-Pricing-and-Implied-Volatility-Analysis-With-Heston-and-Bates-Model.git

# Change directory
cd Forward-Start-Option-Pricing-and-Implied-Volatility-Analysis-With-Heston-and-Bates-Model

# Create virtual environment (optional)
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

## Usage

1. Run the Streamlit app.
2. Adjust the model parameters for the Heston and Bates models.
3. Observe the differences in implied volatility and forward-start option pricing.
4. Compare short-term vs. long-term IV and the effect of jump parameters.

## Results and Observations

- **Heston vs. Bates Model**: The Bates model introduces additional volatility due to jumps, leading to different IV surface shapes.
- **Forward-Start Option Pricing**: Prices vary significantly depending on volatility dynamics and jump risks.
- **Short-Term vs. Long-Term IV**: Short-term options exhibit more volatile IV behavior, while long-term options tend to stabilize.

## References

- John C. Hull, *Options, Futures, and Other Derivatives*.
- Steven E. Shreve, *Stochastic Calculus for Finance*.
- Lech A. Grzelak, *The Heston Model and Pricing of Forward Start Options*.

## Author

**Shubh Shrishrimal**\
GitHub: [shubh123a3](https://github.com/shubh123a3)

Feel free to contribute or suggest improvements!


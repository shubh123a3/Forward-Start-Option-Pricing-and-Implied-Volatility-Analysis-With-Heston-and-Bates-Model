# Forward-Start Option Pricing and Implied Volatility Analysis with Heston and Bates Models

## Overview
This project explores forward-start option pricing using the **Heston** and **Bates** models. It provides an interactive **Streamlit** dashboard to analyze implied volatility (IV) behavior for short-term and long-term options and observe the impact of jump parameters. The models are implemented in Python using **NumPy**, **SciPy**, and **Matplotlib**.

## Features
- **Heston Model Implementation**: Computes option prices and implied volatility using the stochastic volatility model.
- **Bates Model Implementation**: Extends the Heston model by incorporating jump diffusion effects.
- **Forward-Start Option Pricing**: Evaluates forward-start options under stochastic volatility.
- **Implied Volatility Analysis**: Compares short-term and long-term IV surfaces for different parameter settings.
- **Interactive Streamlit Dashboard**: Visualizes results dynamically based on user input.

---

## Mathematical Formulation

### 1. **Heston Model**
The **Heston stochastic volatility model** is given by:
\[
\begin{aligned}
    dS_t &= \mu S_t dt + \sqrt{V_t} S_t dW_t^S, \\
    dV_t &= \kappa (\theta - V_t) dt + \sigma \sqrt{V_t} dW_t^V, \\
    dW_t^S dW_t^V &= \rho dt,
\end{aligned}
\]
where:
- \( S_t \) is the underlying asset price.
- \( V_t \) is the stochastic variance.
- \( \mu \) is the drift of the asset price.
- \( \kappa \) is the mean-reversion rate of variance.
- \( \theta \) is the long-term mean variance.
- \( \sigma \) is the volatility of variance (vol of vol).
- \( \rho \) is the correlation between the Brownian motions.

#### Characteristic Function of Heston Model
The characteristic function of the Heston model is given by:
\[
\varphi(u) = \exp(A(u) + C(u)V_0)
\]
where:
\[
A(u) = i u r \tau + \frac{\kappa \theta}{\sigma^2} \left[ (\kappa - i u \rho \sigma + D) \tau - 2 \ln \left(\frac{1 - g e^{D\tau}}{1 - g} \right) \right]
\]
\[
C(u) = \frac{\kappa - i u \rho \sigma + D}{\sigma^2} \left(\frac{1 - e^{D\tau}}{1 - g e^{D\tau}} \right)
\]
where \( D \) and \( g \) are functions of \( u \), \( \rho \), \( \sigma \), and \( \kappa \).

---

### 2. **Bates Model (Heston + Jumps)**
The **Bates model** extends the Heston model by adding jump components:
\[
    dS_t = \mu S_t dt + \sqrt{V_t} S_t dW_t^S + S_t dJ_t,
\]
where \( J_t \) represents the jump process following a **Poisson process**:
\[
    J_t \sim \sum_{i=1}^{N_t} (e^{Z_i} - 1),
\]
with \( N_t \sim \text{Poisson}(\lambda t) \) and \( Z_i \sim \mathcal{N}(\mu_J, \sigma_J^2) \).

The Bates characteristic function extends the Heston model by adding:
\[
\varphi_{\text{Bates}}(u) = \varphi_{\text{Heston}}(u) \cdot \exp \left( \lambda \tau \left( e^{i u \mu_J - 0.5 u^2 \sigma_J^2} - 1 \right) \right)
\]
where \( \lambda \) is the jump intensity, \( \mu_J \) is the mean jump size, and \( \sigma_J \) is the jump volatility.

---

### 3. **Forward-Start Option Pricing**
A **forward-start option** is an option where the strike is determined at a future date \( T_1 \), with maturity at \( T_2 \). The price is given by:
\[
    C_{\text{Forward}} = e^{-rT_1} \mathbb{E} \left[ C_{\text{BS}}(S_{T_1}, K, T_2 - T_1, r, \sigma) \right],
\]
where \( C_{\text{BS}} \) is the Black-Scholes price of the option with forward-start parameters.

For the Black-Scholes model, the forward-start option price is:
\[
    C_{\text{BS}}(K, \sigma, T_1, T_2, r) = e^{-rT_1} N(d_1) - e^{-rT_2} K N(d_2),
\]
where:
\[
    d_1 = \frac{\ln(1/K) + (r + 0.5\sigma^2) (T_2 - T_1)}{\sigma \sqrt{T_2 - T_1}},
\]
\[
    d_2 = d_1 - \sigma \sqrt{T_2 - T_1}.
\]

---

## Installation
To run this project locally:
```bash
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

---

## Usage
1. Run the **Streamlit** app.
2. Adjust the model parameters for the **Heston** and **Bates** models.
3. Observe the differences in implied volatility and forward-start option pricing.
4. Compare short-term vs. long-term IV and the effect of jump parameters.

---

## Results and Observations
- **Heston vs. Bates Model**: The **Bates model** introduces additional volatility due to jumps, leading to different IV surface shapes.
- **Forward-Start Option Pricing**: Prices vary significantly depending on volatility dynamics and jump risks.
- **Short-Term vs. Long-Term IV**: Short-term options exhibit more volatile IV behavior, while long-term options tend to stabilize.

---

## References
- John C. Hull, "Options, Futures, and Other Derivatives."
- Steven E. Shreve, "Stochastic Calculus for Finance."
- Lech A. Grzelak, "The Heston Model and Pricing of Forward Start Options."

---

## Author
**Shubh Shrishrimal**  
GitHub: [shubh123a3](https://github.com/shubh123a3)

Feel free to contribute or suggest improvements!


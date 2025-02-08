Forward-Start Option Pricing and Implied Volatility Analysis with Heston and Bates Models

Overview

This project explores forward-start option pricing using the Heston and Bates models. It provides an interactive Streamlit dashboard to analyze implied volatility (IV) behavior for short-term and long-term options and observe the impact of jump parameters. The models are implemented in Python using NumPy, SciPy, and Matplotlib.

Features

Heston Model Implementation: Computes option prices and implied volatility using the stochastic volatility model.

Bates Model Implementation: Extends the Heston model by incorporating jump diffusion effects.

Forward-Start Option Pricing: Evaluates forward-start options under stochastic volatility.

Implied Volatility Analysis: Compares short-term and long-term IV surfaces for different parameter settings.

Interactive Streamlit Dashboard: Visualizes results dynamically based on user input.

Mathematical Formulation

1. Heston Model

The Heston stochastic volatility model is given by:



where:

 is the underlying asset price.

 is the stochastic variance.

 is the drift of the asset price.

 is the mean-reversion rate of variance.

 is the long-term mean variance.

 is the volatility of variance (vol of vol).

 is the correlation between the Brownian motions.

Characteristic Function of Heston Model

The characteristic function of the Heston model is given by:



where:




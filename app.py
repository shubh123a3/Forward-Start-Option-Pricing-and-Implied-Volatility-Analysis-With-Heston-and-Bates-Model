import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optimize
from enum import Enum
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")


# Enum for option type
class OptionType(Enum):
    CALL = 1.0
    PUT = -1.0

def CallPutOptionPriceCOSMthd(cf, CP, S0, r, tau, K, N, L):
    # cf   - characteristic function as a functon, in the book denoted as \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - interest rate (constant)
    # tau  - time to maturity
    # K    - list of strikes
    # N    - Number of expansion terms
    # L    - size of truncation domain (typ.:L=8 or L=10)

    # reshape K to a column vector
    if K is not np.array:
        K = np.array(K).reshape([len(K), 1])

    i = 1j
    x0 = np.log(S0 / K)
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    # sumation frm k=0 to k=n-1
    k = np.linspace(0, N - 1, N).reshape([N, 1])
    u = k * np.pi / (b - a)

    # cofficent hk
    H_K = CallPutCoefficients(CP, a, b, k)
    mat = np.exp(i * np.outer((x0 - a), u))
    temp = cf(u) * H_K
    temp[0] = 0.5 * temp[0]
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))
    return value


def CallPutCoefficients(CP, a, b, k):
    if CP == OptionType.CALL:
        c = 0.0
        d = b
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k), 1])
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)
    elif CP == OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k = 2.0 / (b - a) * (- Chi_k + Psi_k)

    return H_k


def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c

    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)), 2.0))
    expr1 = np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(k * np.pi
                                                                       * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi *
                                         (d - a) / (b - a)) - k * np.pi / (b - a) * np.sin(k
                                                                                           * np.pi * (c - a) / (
                                                                                                       b - a)) * np.exp(
        c)
    chi = chi * (expr1 + expr2)

    value = {"chi": chi, "psi": psi}
    return value


def BS_Call_Option_Price(CP, S_0, K, sigma, tau, r):
    if K is list:
        K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0))
          * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = stats.norm.cdf(d1) * S_0 - stats.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = stats.norm.cdf(-d2) * K * np.exp(-r * tau) - stats.norm.cdf(-d1) * S_0
    return value


def ImpliedVolatility(CP, marketPrice, K, T, S_0, r):
    # to difne intail volailty er ainterplooalte define a grid for sigma
    sigmaGrid = np.linspace(0, 2, 200)
    optPriceGrid = BS_Call_Option_Price(CP, S_0, K, sigmaGrid, T, r)
    sigmaInitial = np.interp(marketPrice, optPriceGrid, sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))

    # use determine input for the local search fine tuning
    func = lambda sigma: np.power(BS_Call_Option_Price(CP, S_0, K, sigma, T, r) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-10)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol


def ChFBatesModel(r, tau, kappa, gamma, vbar, v0, rho, xiP, muJ, sigmaJ):
    i = 1j
    D1 = lambda u: np.sqrt(np.power(kappa - gamma * rho * i * u, 2) + (u * u + i * u) * gamma * gamma)
    g = lambda u: (kappa - gamma * rho * i * u - D1(u)) / (kappa - gamma * rho * i * u + D1(u))
    C = lambda u: (1.0 - np.exp(-D1(u) * tau)) / (gamma * gamma * (1.0 - g(u) * \
                                                                   np.exp(-D1(u) * tau))) * (
                          kappa - gamma * rho * i * u - D1(u))
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    AHes = lambda u: r * i * u * tau + kappa * vbar * tau / gamma / gamma * (kappa - gamma * \
                                                                             rho * i * u - D1(
                u)) - 2 * kappa * vbar / gamma / gamma * np.log((1.0 - g(u) * np.exp(-D1(u) * tau)) / (1.0 - g(u)))

    A = lambda u: AHes(u) - xiP * i * u * tau * (np.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0) + \
                  xiP * tau * (np.exp(i * u * muJ - 0.5 * sigmaJ * sigmaJ * u * u) - 1.0)

    # Characteristic function for the Heston's model
    cf = lambda u: np.exp(A(u) + C(u) * v0)
    return cf


def ChFHestonModelForwardStart(r, T1, T2, kappa, gamma, vbar, v0, rho):
    i = 1j
    tau = T2 - T1
    D1 = lambda u: np.sqrt(np.power(kappa - gamma * rho * i * u, 2) + (u * u + i * u) * gamma * gamma)
    g = lambda u: (kappa - gamma * rho * i * u - D1(u)) / (kappa - gamma * rho * i * u + D1(u))
    C = lambda u: (1.0 - np.exp(-D1(u) * tau)) / (gamma * gamma * (1.0 - g(u) * np.exp(-D1(u) * tau))) \
                  * (kappa - gamma * rho * i * u - D1(u))
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    A = lambda u: r * i * u * tau + kappa * vbar * tau / gamma / gamma * (kappa - gamma * rho * i * u - D1(u)) \
                  - 2 * kappa * vbar / gamma / gamma * np.log((1.0 - g(u) * np.exp(-D1(u) * tau)) / (1.0 - g(u)))
    c_bar = lambda t1, t2: gamma * gamma / (4.0 * kappa) * (1.0 - np.exp(-kappa * (t2 - t1)))
    delta = 4.0 * kappa * vbar / gamma / gamma
    kappa_bar = lambda t1, t2: 4.0 * kappa * v0 * np.exp(-kappa * (t2 - t1)) / (
                gamma * gamma * (1.0 - np.exp(-kappa * (t2 - t1))))
    term1 = lambda u: A(u) + C(u) * c_bar(0.0, T1) * kappa_bar(0.0, T1) / (1.0 - 2.0 * C(u) * c_bar(0.0, T1))
    term2 = lambda u: np.power(1.0 / (1.0 - 2.0 * C(u) * c_bar(0.0, T1)), 0.5 * delta)
    cf = lambda u: np.exp(term1(u)) * term2(u)
    return cf
def CallPutOptionPriceCOSMthd_FrwdStart(cf, CP, r, T1, T2, K, N, L):
    # cf   - characteristic function as a functon, in the book denoted as \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - interest rate (constant)
    # K    - list of strikes
    # N    - Number of expansion terms
    # L    - size of truncation domain (typ.:L=8 or L=10)

    tau = T2 - T1
    # reshape K to a column vector
    if K is not np.array:
        K = np.array(K).reshape([len(K), 1])

    # Adjust strike
    K = K + 1.0

    # assigning i=sqrt(-1)
    i = 1j
    x0 = np.log(1.0 / K)

    # truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)

    # sumation from k = 0 to k=N-1
    k = np.linspace(0, N - 1, N).reshape([N, 1])
    u = k * np.pi / (b - a);

    # Determine coefficients for Put Prices
    H_k = CallPutCoefficients(CP, a, b, k)
    mat = np.exp(i * np.outer((x0 - a), u))
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = np.exp(-r * T2) * K * np.real(mat.dot(temp))
    return value
# Implied volatility for the forward start call option
def ImpliedVolatility_FrwdStart(marketPrice, K, T1, T2, r):
    # To determine initial volatility we interpolate define a grid for sigma
    # and interpolate on the inverse
    sigmaGrid = np.linspace(0, 2, 200)
    optPriceGrid = BS_Call_Option_Price_FrwdStart(K, sigmaGrid, T1, T2, r)
    sigmaInitial = np.interp(marketPrice, optPriceGrid, sigmaGrid)
    print("Initial volatility = {0}".format(sigmaInitial))

    # Use determined input for the local-search (final tuning)
    func = lambda sigma: np.power(BS_Call_Option_Price_FrwdStart(K, sigma, T1, T2, r) - marketPrice, 1.0)
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    print("Final volatility = {0}".format(impliedVol))
    return impliedVol

# Forward start Black-Scholes option price
def BS_Call_Option_Price_FrwdStart(K, sigma, T1, T2, r):
    if K is list:
        K = np.array(K).reshape([len(K), 1])
    K = K + 1.0
    tau = T2 - T1
    d1 = (np.log(1.0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    value = np.exp(-r * T1) * stats.norm.cdf(d1) - stats.norm.cdf(d2) * K * np.exp(-r * T2)
    return value

def plot_implied_volatility_heston(K, TMat, title, color_palette, r, kappa, gamma, vbar, v0, rho, CP, N, L):
    """
    Streamlit function to plot implied volatility for a Heston forward-start option.

    Parameters:
    - K: Strike prices
    - TMat: List of (T1, T2) pairs for forward-start options
    - title: Title of the plot (str)
    - color_palette: Seaborn color palette (str)
    - Other model parameters required for computation
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Strike, K", fontsize=12)
    ax.set_ylabel("Implied Volatility (%)", fontsize=12)
    ax.grid()

    colors = sns.color_palette(color_palette, len(TMat))

    for i, (T1, T2) in enumerate(TMat):
        cf = ChFHestonModelForwardStart(r, T1, T2, kappa, gamma, vbar, v0, rho)
        valCOS = CallPutOptionPriceCOSMthd_FrwdStart(cf, CP, r, T1, T2, K, N, L)

        # Compute implied volatilities
        IV = np.array([ImpliedVolatility_FrwdStart(valCOS[idx], K[idx], T1, T2, r) for idx in range(len(K))])

        # Plot with unique color
        ax.plot(K, IV * 100.0, linestyle='--', marker='o', markersize=4, color=colors[i], label=f"T1={T1}, T2={T2}")

    ax.legend(fontsize=10)
    st.pyplot(fig)  # Display plot in Streamlit

st.title("Forward-Start Option Pricing and Implied Volatility Analysis With Heston and Bates Model")

model_choice = st.sidebar.selectbox("Select Model", ["Heston Model", "Bates Model"])

if model_choice == "Heston Model":
    st.header("Heston Model Analysis")

    col1, col2 = st.columns(2)
    with col1:
        kappa = st.slider("κ (Mean Reversion)", 0.1, 2.0, 0.6)
        gamma = st.slider("γ (Vol of Vol)", 0.05, 0.5, 0.2)
        vbar = st.slider("v̄ (Long-term Var)", 0.01, 0.3, 0.1)
    with col2:
        v0 = st.slider("v₀ (Initial Var)", 0.01, 0.3, 0.05)
        rho = st.slider("ρ (Correlation)", -0.95, 0.95, -0.5)
        r = st.slider("r (Risk-free Rate)", 0.0, 0.1, 0.0)

    analysis_type = st.radio("Analysis Type", ["Implied Volatility", "Option Prices"])

    if analysis_type == "Implied Volatility":
        st.subheader("Forward-Start Implied Volatility")
        K = np.linspace(-0.4, 4.0, 20)
        N = 500
        L = 10

        TMat1 = [[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0]]
        TMat2 = [[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]

        #fig1, ax1 = plt.subplots(figsize=(10, 6))
        plot_implied_volatility_heston(K, TMat1, "Short-Term IV", "coolwarm", r, kappa, gamma, vbar, v0, rho,
                                       OptionType.CALL, N, L)
        #st.pyplot(fig1)


        #fig2, ax2 = plt.subplots(figsize=(10, 6))
        plot_implied_volatility_heston(K, TMat2, "Long-Term IV", "viridis", r, kappa, gamma, vbar, v0, rho,
                                       OptionType.CALL, N, L)
        #st.pyplot(fig2)

    else:
        st.subheader("Forward-Start Option Prices")
        T1 = st.slider("T1 (Fixation Date)", 0.1, 5.0, 1.0)
        T2 = st.slider("T2 (Maturity)", T1 + 0.1, 10.0, 2.0)
        K = np.linspace(-0.4, 4.0, 50)
        N = 500
        L = 10

        cf = ChFHestonModelForwardStart(r, T1, T2, kappa, gamma, vbar, v0, rho)
        prices = CallPutOptionPriceCOSMthd_FrwdStart(cf, OptionType.CALL, r, T1, T2, K, N, L)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(K, prices, 'b-', linewidth=2)
        ax.set_xlabel("Strike (K)")
        ax.set_ylabel("Option Price")
        ax.set_title("Forward Start Option Prices")
        ax.grid(True)
        st.pyplot(fig)

elif model_choice == "Bates Model":
    st.header("Bates Model Analysis")

    col1, col2 = st.columns(2)
    with col1:
        S0 = st.slider("S₀ (Spot Price)", 50.0, 150.0, 100.0)
        r = st.slider("r (Risk-free Rate)", 0.0, 0.1, 0.0)
        tau = st.slider("τ (Maturity)", 0.1, 5.0, 1.0)
        kappa = st.slider("κ (Mean Reversion)", 0.1, 2.0, 1.2)
    with col2:
        gamma = st.slider("γ (Vol of Vol)", 0.01, 0.5, 0.05)
        vbar = st.slider("v̄ (Long-term Var)", 0.01, 0.3, 0.05)
        rho = st.slider("ρ (Correlation)", -0.95, 0.95, -0.75)

    st.subheader("Jump Parameters")
    xiP = st.slider("ξ (Jump Intensity)", 0.0, 0.5, 0.1)
    muJ = st.slider("μ_J (Jump Mean)", -0.5, 0.5, 0.0)
    sigmaJ = st.slider("σ_J (Jump Vol)", 0.01, 0.5, 0.2)

    K = np.linspace(40, 180, 20)
    N = 1000
    L = 6

    param_choice = st.selectbox("Select Parameter to Analyze", ["xiP", "muJ", "sigmaJ"])

    if param_choice == "xiP":
        param_values = [0.01, 0.1, 0.2, 0.3]
    elif param_choice == "muJ":
        param_values = [-0.5, -0.25, 0, 0.25]
    else:
        param_values = [0.01, 0.15, 0.2, 0.25]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(param_values))

    for i, value in enumerate(param_values):
        if param_choice == "xiP":
            cf = ChFBatesModel(r, tau, kappa, gamma, vbar, vbar, rho, value, muJ, sigmaJ)
        elif param_choice == "muJ":
            cf = ChFBatesModel(r, tau, kappa, gamma, vbar, vbar, rho, xiP, value, sigmaJ)
        else:
            cf = ChFBatesModel(r, tau, kappa, gamma, vbar, vbar, rho, xiP, muJ, value)

        prices = CallPutOptionPriceCOSMthd(cf, OptionType.CALL, S0, r, tau, K, N, L)
        IV = [ImpliedVolatility(OptionType.CALL, p, k, tau, S0, r) for p, k in zip(prices, K)]

        ax.plot(K, np.array(IV) * 100, label=f"{param_choice}={value}", color=colors[i])

    ax.set_title(f"Effect of {param_choice} on Implied Volatility")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility (%)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
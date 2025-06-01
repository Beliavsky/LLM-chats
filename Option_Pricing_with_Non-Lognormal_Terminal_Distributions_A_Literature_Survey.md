# Option Pricing with Non-Lognormal Terminal Distributions: A Literature Survey

## Table of Contents
1. [Introduction](#introduction)
2. [The Black-Scholes Model and Its Limitations](#the-black-scholes-model-and-its-limitations)
3. [Stochastic Volatility Models](#stochastic-volatility-models)
4. [Jump-Diffusion Models](#jump-diffusion-models)
5. [Lévy Process Models](#lévy-process-models)
6. [Local Volatility Models](#local-volatility-models)
7. [Other Non-Gaussian Approaches](#other-non-gaussian-approaches)
8. [Hybrid Models](#hybrid-models)
9. [Recent Developments](#recent-developments)
10. [Comparative Analysis](#comparative-analysis)
11. [Practical Considerations](#practical-considerations)
12. [Conclusion](#conclusion)
13. [References](#references)

## Introduction

Option pricing theory has evolved significantly since the groundbreaking work of Black, Scholes, and Merton in the early 1970s. While the Black-Scholes model revolutionized financial mathematics and derivatives pricing, its fundamental assumption that asset returns follow a lognormal distribution has been consistently contradicted by empirical evidence. Market data exhibits features such as volatility smiles and skews, fat tails in return distributions, excess kurtosis, and asymmetric returns that cannot be captured by the lognormal distribution assumption.

This literature survey examines the extensive body of research on option pricing models that incorporate non-lognormal terminal distributions. These alternative approaches have been developed to address the limitations of the Black-Scholes framework and better capture the empirical characteristics of financial markets. The survey covers major categories of models including stochastic volatility, jump-diffusion, Lévy processes, and local volatility, as well as hybrid approaches and recent developments in the field.

By relaxing the restrictive assumptions of the Black-Scholes model, particularly the lognormal distribution assumption, these alternative models aim to provide more accurate option prices, better hedging strategies, and improved risk management tools. This survey synthesizes the theoretical foundations, implementation challenges, empirical performance, and practical applications of these models, offering a comprehensive overview of the state of the art in non-lognormal option pricing.

## The Black-Scholes Model and Its Limitations

### The Black-Scholes Framework

The Black-Scholes model, developed by Fischer Black, Myron Scholes, and Robert Merton in the early 1970s, provides a theoretical framework for pricing European options. The model assumes that the price of the underlying asset follows a geometric Brownian motion:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Where:
- $S_t$ is the asset price at time $t$
- $\mu$ is the drift (expected return)
- $\sigma$ is the volatility (assumed constant)
- $W_t$ is a standard Wiener process

Under this model, the terminal distribution of the asset price at maturity $T$ is lognormal:

$$S_T = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)T + \sigma W_T\right)$$

The celebrated Black-Scholes formula for a European call option with strike price $K$ and maturity $T$ is:

$$C(S_0, K, T) = S_0 N(d_1) - Ke^{-rT} N(d_2)$$

Where:
- $N(\cdot)$ is the cumulative distribution function of the standard normal distribution
- $d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$
- $d_2 = d_1 - \sigma\sqrt{T}$
- $r$ is the risk-free interest rate

### Empirical Limitations

Despite its elegance and theoretical importance, the Black-Scholes model has several well-documented empirical limitations:

1. **Volatility Smile/Skew**: Market-implied volatilities exhibit a pattern across strike prices (the "volatility smile" or "skew") that contradicts the constant volatility assumption. This pattern became particularly pronounced after the 1987 market crash.

2. **Fat Tails**: Empirical asset return distributions show significantly fatter tails than the normal distribution, indicating a higher probability of extreme events.

3. **Excess Kurtosis**: Related to fat tails, market returns exhibit excess kurtosis (a higher peak and fatter tails) compared to the normal distribution.

4. **Asymmetric Returns**: Market returns often show negative skewness, with large negative returns occurring more frequently than large positive returns.

5. **Volatility Clustering**: Periods of high volatility tend to cluster together, contradicting the constant volatility assumption.

These empirical observations have motivated the development of alternative option pricing models that incorporate non-lognormal terminal distributions. The following sections examine the major categories of these models.

## Stochastic Volatility Models

Stochastic volatility models extend the Black-Scholes framework by allowing the volatility itself to follow a stochastic process. This addresses one of the key limitations of the Black-Scholes model, which assumes constant volatility.

### Heston Model

The Heston model, developed by Steven L. Heston in 1993, is one of the most widely used stochastic volatility models. It assumes that the asset price and its variance follow coupled stochastic differential equations:

For the asset price:
$$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S$$

For the variance:
$$dv_t = \kappa(\theta - v_t) dt + \xi\sqrt{v_t} dW_t^v$$

Where:
- $S_t$ is the asset price
- $v_t$ is the instantaneous variance
- $\mu$ is the drift
- $\kappa$ is the rate at which $v_t$ reverts to $\theta$
- $\theta$ is the long-run average variance
- $\xi$ is the volatility of the volatility
- $W_t^S$ and $W_t^v$ are Wiener processes with correlation $\rho$

The model has five parameters:
1. $v_0$: Initial variance
2. $\theta$: Long-run average variance
3. $\kappa$: Mean reversion speed
4. $\xi$: Volatility of volatility
5. $\rho$: Correlation between the two Wiener processes

The Feller condition ($2\kappa\theta > \xi^2$) ensures that the variance process remains strictly positive.

Heston's model provides a closed-form solution for European options using characteristic functions and Fourier transforms. The model can generate implied volatility smiles and skews, making it more consistent with market observations than the Black-Scholes model.

### SABR Model

The SABR (Stochastic Alpha, Beta, Rho) model was developed by Patrick Hagan, Deep Kumar, Andrew Lesniewski, and Diana Woodward in 2002. It is particularly popular in interest rate markets.

The model is defined by:
$$dF_t = \alpha_t F_t^\beta dW_t^1$$
$$d\alpha_t = \nu \alpha_t dW_t^2$$

Where:
- $F_t$ is the forward price
- $\alpha_t$ is the stochastic volatility
- $\beta$ is the elasticity parameter (0 ≤ β ≤ 1)
- $\nu$ is the volatility of volatility
- $W_t^1$ and $W_t^2$ are Wiener processes with correlation $\rho$

The SABR model is particularly valued for its analytical approximation for implied volatility, which makes it computationally efficient for calibration and pricing.

### Hull-White Model

Hull and White (1987) proposed a model where the variance follows a mean-reverting process:

$$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^1$$
$$dv_t = \kappa(\theta - v_t) dt + \xi v_t^{\gamma} dW_t^2$$

Where $\gamma$ is typically set to 0.5 or 1. This model generalizes the Heston model and provides more flexibility in the variance process.

### GARCH Option Pricing Models

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models have been adapted for option pricing, notably by Heston and Nandi (2000). These models provide a bridge between time-series modeling of volatility and option pricing.

The discrete-time GARCH(1,1) process for returns can be written as:
$$r_t = \mu + \sigma_t \epsilon_t$$
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Where $\epsilon_t$ are i.i.d. standard normal variables.

### Empirical Performance of Stochastic Volatility Models

Research has shown that stochastic volatility models generally outperform the Black-Scholes model in capturing market phenomena such as:
- Volatility smiles and skews
- Term structure of implied volatility
- Mean-reverting nature of volatility

However, calibration remains challenging, and model performance varies across different market conditions and asset classes. The correlation parameter $\rho$ is particularly important for capturing the leverage effect (negative correlation between returns and volatility) observed in equity markets.

## Jump-Diffusion Models

Jump-diffusion models extend the standard Black-Scholes framework by incorporating sudden, discontinuous price movements (jumps) alongside the continuous diffusion process. These models were developed to address empirical observations that asset returns exhibit fat tails, excess kurtosis, and asymmetric distributions.

### Merton Jump-Diffusion Model

Robert Merton's seminal paper "Option Pricing When Underlying Stock Returns are Discontinuous" (1976) introduced the first jump-diffusion model. In this model, the asset price follows:

$$dS_t = (\mu - \lambda k)S_t dt + \sigma S_t dW_t + (J - 1)S_t dN_t$$

Where:
- $S_t$ is the asset price
- $\mu$ is the drift
- $\sigma$ is the volatility of the diffusion component
- $W_t$ is a standard Wiener process
- $N_t$ is a Poisson process with intensity $\lambda$
- $J$ is the jump size, typically assumed to be lognormally distributed with mean $(1 + k)$
- $\lambda k$ represents the compensator that ensures the risk-neutral property

The jump sizes in Merton's model are lognormally distributed, with:
$$\ln(J) \sim N(\mu_J, \sigma_J^2)$$

The price of a European call option under Merton's model can be expressed as an infinite sum:

$$C(S, K, T) = \sum_{n=0}^{\infty} \frac{e^{-\lambda T}(\lambda T)^n}{n!} BS(S, K, T, r_n, \sigma_n)$$

Where:
- $BS$ is the Black-Scholes formula
- $r_n = r - \lambda k + \frac{n\mu_J}{T}$
- $\sigma_n^2 = \sigma^2 + \frac{n\sigma_J^2}{T}$

Merton's model can capture fat tails and excess kurtosis in the return distribution, making it more consistent with empirical observations than the Black-Scholes model.

### Kou Double Exponential Jump-Diffusion Model

Steven Kou (2002) proposed a jump-diffusion model where the jump sizes follow a double exponential distribution rather than a lognormal distribution. This allows for more flexible modeling of both upward and downward jumps.

The asset price dynamics are:

$$dS_t = \mu S_t dt + \sigma S_t dW_t + S_t d\left(\sum_{i=1}^{N_t} (V_i - 1)\right)$$

Where:
- $N_t$ is a Poisson process with intensity $\lambda$
- $V_i$ are i.i.d. random variables representing the jump sizes
- $\ln(V_i)$ follows an asymmetric double exponential distribution with density:
  $$f(y) = p \cdot \eta_1 e^{-\eta_1 y} \mathbf{1}_{\{y \geq 0\}} + (1-p) \cdot \eta_2 e^{\eta_2 y} \mathbf{1}_{\{y < 0\}}$$

Where:
- $p$ is the probability of an upward jump
- $\eta_1 > 1$ and $\eta_2 > 0$ are parameters controlling the decay of the tails

Kou's model is particularly valued for its analytical tractability for various option pricing problems, including barrier and lookback options. The double exponential distribution allows for more realistic modeling of market crashes and rallies compared to the lognormal jump distribution in Merton's model.

### Bates Model

David Bates (1996) combined Heston's stochastic volatility model with Merton's jump-diffusion approach, creating a hybrid model that captures both stochastic volatility and jumps:

$$dS_t = (\mu - \lambda k)S_t dt + \sqrt{v_t} S_t dW_t^S + (J - 1)S_t dN_t$$
$$dv_t = \kappa(\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_t^v$$

Where $W_t^S$ and $W_t^v$ are correlated Wiener processes with correlation $\rho$, and the jump component is similar to Merton's model.

This hybrid approach combines the benefits of both stochastic volatility and jump-diffusion, providing a more comprehensive framework for capturing market dynamics, especially during crises.

### Empirical Performance of Jump-Diffusion Models

Jump-diffusion models have been shown to outperform pure diffusion models in:
- Capturing fat tails in return distributions
- Modeling market crashes and sudden price movements
- Explaining the volatility smile, especially for short-term options
- Pricing options across different strikes and maturities

However, calibration can be challenging due to the need to separate the diffusion and jump components, and model performance may vary across different market conditions and asset classes.

## Lévy Process Models

Lévy process models represent a significant extension to the Black-Scholes framework by generalizing the Brownian motion to allow for jumps and non-Gaussian distributions. These models are particularly effective at capturing the fat tails, excess kurtosis, and asymmetric returns observed in financial markets. Unlike jump-diffusion models that typically have finite jump activity, many Lévy processes allow for infinite activity (infinite number of small jumps in any finite time interval).

### Variance Gamma (VG) Model

Developed by Dilip Madan and Eugene Seneta (1990) and further extended by Madan, Carr, and Chang (1998), the Variance Gamma process is obtained by evaluating a Brownian motion with drift at a random time given by a gamma process.

The asset price dynamics under the VG model are:
$$S_t = S_0 \exp\left((r - q + \omega)t + X_t^{VG}(\sigma, \nu, \theta)\right)$$

Where:
- $X_t^{VG}$ is the VG process
- $\sigma$ controls the volatility
- $\nu$ controls the kurtosis (tail heaviness)
- $\theta$ controls the skewness
- $\omega$ is a compensator term to ensure martingale property

The VG process can be represented as:
$$X_t^{VG}(\sigma, \nu, \theta) = \theta G_t + \sigma W_{G_t}$$

Where:
- $G_t$ is a gamma process with mean rate 1 and variance rate $\nu$
- $W_t$ is a standard Brownian motion

The VG model provides a more realistic description of asset price dynamics than the Black-Scholes model, particularly in capturing fat tails and skewness in the return distribution.

### Normal Inverse Gaussian (NIG) Model

Introduced by Ole Barndorff-Nielsen (1997), the NIG process is another popular Lévy process for option pricing. It is obtained by evaluating a Brownian motion at a random time given by an inverse Gaussian process.

The asset price dynamics under the NIG model are:
$$S_t = S_0 \exp\left((r - q + \omega)t + X_t^{NIG}(\alpha, \beta, \delta)\right)$$

Where:
- $X_t^{NIG}$ is the NIG process
- $\alpha$ controls the tail heaviness
- $\beta$ controls the skewness
- $\delta$ is a scale parameter
- $\omega$ is a compensator term

The NIG model provides an excellent fit to empirical return distributions and has semi-analytical solutions for European options.

### CGMY/KoBoL Model

Developed by Peter Carr, Hélyette Geman, Dilip Madan, and Marc Yor (2002), the CGMY model (also known as KoBoL) is a generalization of the VG process that allows for additional flexibility in modeling the behavior of small jumps.

The Lévy density of the CGMY process is:
$$\nu(x) = \begin{cases}
C \frac{e^{-G|x|}}{|x|^{1+Y}} & \text{for } x < 0 \\
C \frac{e^{-Mx}}{x^{1+Y}} & \text{for } x > 0
\end{cases}$$

Where:
- $C > 0$ is a scale parameter
- $G, M > 0$ control the rate of exponential decay on the left and right tails
- $Y < 2$ controls the fine structure of the process

The CGMY model is highly flexible and can capture a wide range of market behaviors. The parameter $Y$ allows for fine-tuning of jump activity, with different ranges of $Y$ corresponding to different types of processes:
- $Y < 0$: Finite activity (compound Poisson process)
- $0 < Y < 1$: Infinite activity but finite variation
- $1 < Y < 2$: Infinite activity and infinite variation

### Meixner Model

The Meixner process, introduced by Schoutens and Teugels (1998), is another Lévy process used in option pricing. Its probability density function is related to the Meixner-Pollaczek polynomials.

The Lévy density of the Meixner process is:
$$\nu(x) = d \frac{e^{\alpha x/2}}{x \sinh(\alpha x/2)}$$

Where $d > 0$ and $\alpha > 0$ are parameters.

The Meixner model provides a good fit to empirical data and has analytical tractability for certain calculations.

### Implementation and Calibration of Lévy Process Models

Lévy process models are typically implemented using characteristic functions and Fourier transform techniques. The Fast Fourier Transform (FFT) algorithm, as described by Carr and Madan (1999), is particularly useful for efficient option pricing under these models.

Calibration methods include:
- Maximum Likelihood Estimation
- Method of Moments
- Empirical Characteristic Function
- Calibration to option prices using least squares

### Empirical Performance of Lévy Process Models

Lévy process models have been shown to outperform both the Black-Scholes model and basic jump-diffusion models in:
- Capturing the entire volatility surface
- Modeling both short-term and long-term option prices
- Providing a more realistic description of asset price dynamics
- Handling a wide range of market conditions

However, these models can be mathematically complex and computationally intensive, and their performance may vary across different market conditions and asset classes.

## Local Volatility Models

Local volatility models extend the Black-Scholes framework by allowing the volatility to be a deterministic function of both the current asset price and time. Unlike stochastic volatility models, local volatility models do not introduce additional sources of randomness.

### Dupire's Local Volatility Model

Bruno Dupire's seminal work (1994) established that given a complete set of European option prices for all strikes and maturities, one can derive a unique local volatility function σ(S,t) that is consistent with these prices. The asset price dynamics under this model are:

$$dS_t = (r - q)S_t dt + \sigma(S_t, t)S_t dW_t$$

Where:
- $S_t$ is the asset price
- $r$ is the risk-free rate
- $q$ is the dividend yield
- $\sigma(S_t, t)$ is the local volatility function
- $W_t$ is a standard Wiener process

The key result, known as Dupire's formula, relates the local volatility function to the partial derivatives of option prices:

$$\sigma^2(K, T) = \frac{\frac{\partial C}{\partial T} + (r - q)K\frac{\partial C}{\partial K} + qC}{\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}}$$

Where:
- $C$ is the price of a European call option
- $K$ is the strike price
- $T$ is the time to maturity

Dupire's model perfectly calibrates to the market prices of European options by construction. However, it requires a complete, arbitrage-free volatility surface, and interpolation and extrapolation issues can arise in practice.

### Constant Elasticity of Variance (CEV) Model

Introduced by John Cox in 1975, the CEV model is a parametric local volatility model where the volatility is a power function of the asset price:

$$dS_t = \mu S_t dt + \sigma S_t^{\beta} dW_t$$

Where:
- $\beta$ is the elasticity parameter (typically $\beta < 1$)
- When $\beta = 1$, the model reduces to geometric Brownian motion (Black-Scholes)
- When $\beta = 0$, the model becomes the Bachelier model with constant absolute volatility

The CEV model captures the leverage effect (negative correlation between returns and volatility) when $\beta < 1$, as observed in equity markets. It has closed-form solutions for European options when $\beta = 0$ or $\beta = 1/2$.

### Displaced Diffusion Model

Introduced by Mark Rubinstein, this model assumes that the asset price follows:

$$dS_t = \mu S_t dt + \sigma (S_t + a) dW_t$$

Where $a$ is a displacement parameter. This can be viewed as a shifted lognormal model.

The displaced diffusion model can generate skewed implied volatility curves and has closed-form solutions for European options.

### Lognormal Mixture Dynamics Model

Developed by Damiano Brigo, Fabio Mercurio, and co-authors, this model represents the asset price density as a weighted sum of lognormal densities:

$$p(S_T) = \sum_{i=1}^n \lambda_i p_{BS}(S_T; \sigma_i)$$

Where:
- $p_{BS}$ is the Black-Scholes lognormal density
- $\lambda_i$ are weights (summing to 1)
- $\sigma_i$ are different volatility values

This mixture model implies a specific form of local volatility and can capture multimodal distributions.

### Implementation and Calibration of Local Volatility Models

Local volatility models are typically implemented using finite difference methods for PDE solution, Monte Carlo simulation, or binomial and trinomial trees (Derman-Kani approach).

Calibration methods include:
- Implied volatility surface fitting
- Tikhonov regularization for Dupire's formula
- Parameterization of the local volatility function
- Least squares minimization of option price differences

### Empirical Performance of Local Volatility Models

Local volatility models have been shown to:
- Perfectly fit European option prices by construction
- Struggle with the dynamics of the implied volatility surface
- Perform well for short-dated options
- Have limitations for exotic options with strong path-dependency

While local volatility models provide an excellent static fit to market prices, their dynamic properties may not be realistic, leading to potential issues in hedging and risk management.

## Other Non-Gaussian Approaches

Beyond the major categories discussed above, researchers have explored various other approaches to address the limitations of the lognormal distribution assumption in the Black-Scholes model.

### Mixture Models

#### Regime-Switching Models
Regime-switching models assume that the market can exist in different states or regimes, with different parameter values in each regime. The switching between regimes is typically governed by a Markov chain.

$$dS_t = \mu(r_t) S_t dt + \sigma(r_t) S_t dW_t$$

Where $r_t$ is a Markov chain representing the current regime.

These models can capture market regime changes (e.g., calm vs. volatile periods) and generate complex distributional shapes. However, they introduce increased complexity and calibration challenges.

#### Mixed Diffusion-Jump Models
These models combine features of continuous diffusion and jump processes in various ways beyond the standard jump-diffusion framework.

### Non-Parametric and Semi-Parametric Approaches

#### Kernel Density Estimation
Non-parametric approaches that directly estimate the risk-neutral density from option prices without assuming a specific functional form. These methods require large amounts of option data and face challenges in smoothing parameter selection and extrapolation.

#### Implied Distribution Methods
Methods that extract the entire risk-neutral distribution directly from option prices. These model-free approaches provide a direct link to market prices but are sensitive to available strikes and face interpolation and extrapolation issues.

#### Edgeworth and Gram-Charlier Expansions
These approaches approximate the risk-neutral density as a perturbation of the normal or lognormal distribution using higher-order moments.

$$f(x) \approx \phi(x) \left( 1 + \frac{\gamma_1}{3!}H_3(x) + \frac{\gamma_2}{4!}H_4(x) + ... \right)$$

Where:
- $\phi(x)$ is the standard normal density
- $\gamma_1$ and $\gamma_2$ are skewness and excess kurtosis
- $H_n(x)$ are Hermite polynomials

These methods offer analytical tractability and clear interpretation of parameters but may not be valid densities (can be negative) and have limited flexibility for extreme deviations from normality.

## Hybrid Models

Researchers have developed various hybrid approaches that combine the strengths of different model categories to better capture market dynamics.

### Stochastic Volatility with Jumps

Models like the Bates model and the SVJJ (Stochastic Volatility with Jumps in both Returns and Volatility) model combine stochastic volatility with jumps to capture both volatility clustering and extreme events. These models provide a more comprehensive framework but face complex calibration challenges due to the large number of parameters.

### Local-Stochastic Volatility Models

Local-stochastic volatility (LSV) models combine the market fit of local volatility with the realistic dynamics of stochastic volatility. These models take the form:

$$dS_t = (r - q)S_t dt + \sigma(S_t, t) \sqrt{v_t} S_t dW_t^S$$
$$dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v$$

Where $\sigma(S_t, t)$ is a deterministic function chosen to ensure calibration to market prices.

LSV models provide a better balance between static calibration and dynamic properties but face computational complexity and implementation challenges.

### Time-Changed Lévy Processes

Time-changed Lévy processes, as proposed by Carr and Wu (2004), combine the distributional flexibility of Lévy processes with the volatility clustering of stochastic time changes. These models take the form:

$$S_t = S_0 \exp\left((r - q + \omega)t + X_{T_t}\right)$$

Where:
- $X_t$ is a Lévy process
- $T_t$ is a stochastic time change, often modeled as an integrated CIR process
- $\omega$ is a compensator term

These models are highly flexible but face mathematical complexity and challenging calibration.

## Recent Developments

### Machine Learning Approaches

Recent years have seen significant growth in the application of machine learning techniques to option pricing, bypassing traditional parametric models entirely.

#### Neural Network Models
Deep learning approaches that directly map option characteristics to prices or implied volatilities have gained popularity. These include:

- **Deep Hedging**: End-to-end learning of hedging strategies
- **Neural SDE**: Neural network parameterization of stochastic differential equations
- **Volatility Surface Prediction**: Direct prediction of implied volatility surfaces

These approaches offer minimal assumptions about underlying distributions and potential to capture complex patterns but face challenges in interpretability and extrapolation.

#### Gaussian Process Regression
Probabilistic modeling of volatility surfaces using Gaussian processes provides uncertainty quantification and smooth interpolation but faces computational costs for large datasets.

### Alternative Distributional Approaches

#### Weibull Distribution
Recent research has explored using the Weibull distribution for option pricing, particularly in specialized contexts like car lease valuation.

#### Quadratic Normal Model (QNM)
A recently developed model that has shown promise in commodity options markets, particularly for oil options.

### Rough Volatility Models

An emerging class of models where volatility is driven by a fractional Brownian motion with Hurst parameter $H < 1/2$, leading to "rough" paths. These models better fit empirical volatility dynamics and improve forecasting of the volatility term structure but face mathematical and computational challenges due to their non-Markovian nature.

## Comparative Analysis

### Terminal Distributions

The terminal distributions of asset prices under different models exhibit varying degrees of departure from the lognormal distribution assumed in the Black-Scholes model:

- **Stochastic Volatility Models**: Mixture of lognormals (conditional on volatility path), with fatter tails and potential skewness
- **Jump-Diffusion Models**: Mixture of jump and diffusion components, with pronounced fat tails and potential significant skewness
- **Lévy Process Models**: Various non-Gaussian distributions (e.g., Variance Gamma, NIG), highly flexible with potentially extreme fat tails and skewness
- **Local Volatility Models**: Implied by market prices, non-parametric, with perfect calibration to European option prices by construction

### Model Complexity and Tractability

The models vary significantly in their complexity and tractability:

- **Stochastic Volatility Models**: Typically 4-5 parameters, with analytical solutions available for European options in some models (e.g., Heston)
- **Jump-Diffusion Models**: Typically 3-5 parameters, with analytical solutions available for European options (infinite series)
- **Lévy Process Models**: Typically 3-4 parameters, with semi-analytical solutions for European options via characteristic functions
- **Local Volatility Models**: Non-parametric or model-specific parameters, with analytical solutions rarely available except in special cases

### Empirical Performance

The models exhibit different strengths in capturing market phenomena:

- **Volatility Smile/Skew Reproduction**: Local volatility models excel at short-term smiles, while stochastic volatility and Lévy process models perform better across the term structure
- **Handling of Market Regimes**: Stochastic volatility and Lévy process models handle normal and stressed markets well, while local volatility models struggle with regime changes
- **Exotic Option Pricing**: Stochastic volatility models perform well for path-dependent and forward-starting options, while local volatility models excel for barrier options

## Practical Considerations

### Calibration Challenges

Calibration of non-lognormal option pricing models presents several challenges:

- **Data Requirements**: Different models require different types and amounts of market data for calibration
- **Stability and Uniqueness**: Many models face issues with parameter stability and uniqueness of calibration
- **Time Consistency**: Models vary in their ability to maintain consistent parameters over time

### Risk Management Implications

The choice of model has significant implications for risk management:

- **Hedging Performance**: Stochastic volatility models generally perform well for delta and vega hedging, while jump models may struggle with sudden price movements
- **Risk Metrics**: Jump and Lévy models typically produce higher Value-at-Risk (VaR) estimates due to their fat-tailed distributions
- **Stress Testing**: Different models produce different stress scenarios, with jump models generally providing more conservative estimates

### Implementation Considerations

Practical implementation of these models involves several considerations:

- **Computational Efficiency**: Models vary in their computational requirements, with closed-form solutions offering significant advantages
- **Numerical Stability**: Some models face numerical challenges, particularly in extreme market conditions
- **Model Risk**: The choice of model introduces model risk, which should be quantified and managed

## Conclusion

The literature on non-lognormal option pricing models reveals a rich ecosystem of approaches, each with distinct strengths and limitations. While no single model dominates across all criteria, the evolution from Black-Scholes to increasingly sophisticated alternatives has substantially improved our ability to capture market realities.

Stochastic volatility models excel in capturing volatility dynamics and term structure effects, jump-diffusion models are particularly effective for modeling crash risk and short-term smiles, Lévy processes offer flexible distributional shapes, and local volatility models provide perfect calibration to current market prices.

The trend toward hybrid models that combine multiple features suggests that the future lies in balanced approaches that capture both the static fit to market prices and realistic dynamic behavior. Meanwhile, machine learning and non-parametric methods are emerging as promising alternatives that may eventually complement or replace traditional parametric approaches.

For practitioners, model selection should be guided by the specific application, market conditions, and available data, with careful attention to calibration stability and hedging performance. The ongoing research in this field continues to push the boundaries of our understanding of option pricing and risk management in non-lognormal markets.

## References

1. Barndorff-Nielsen, O. E. (1997). "Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling." Scandinavian Journal of Statistics, 24(1), 1-13.

2. Bates, D. S. (1996). "Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options." Review of Financial Studies, 9(1), 69-107.

3. Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." Journal of Political Economy, 81(3), 637-654.

4. Brigo, D., & Mercurio, F. (2002). "Lognormal-Mixture Dynamics and Calibration to Market Volatility Smiles." International Journal of Theoretical and Applied Finance, 5(4), 427-446.

5. Bueno-Guerrero, A. (2023). "Option Pricing under a Generalized Black–Scholes Model with Stochastic String Shocks." Mathematics, 12(1), 82.

6. Carr, P., & Madan, D. (1999). "Option Valuation Using the Fast Fourier Transform." Journal of Computational Finance, 2(4), 61-73.

7. Carr, P., Geman, H., Madan, D. B., & Yor, M. (2002). "The Fine Structure of Asset Returns: An Empirical Investigation." Journal of Business, 75(2), 305-332.

8. Carr, P., & Wu, L. (2004). "Time-Changed Lévy Processes and Option Pricing." Journal of Financial Economics, 71(1), 113-141.

9. Chavas, J.P. (2024). "Option pricing revisited: The role of price volatility and expectations." Journal of Commodity Markets, 33.

10. Cont, R., & Tankov, P. (2004). "Financial Modelling with Jump Processes." Chapman & Hall/CRC.

11. Cox, J. C. (1975). "Notes on Option Pricing I: Constant Elasticity of Variance Diffusions." Unpublished note, Stanford University.

12. D'Uggento, A.M. (2025). "From the Black-Scholes model to machine learning methods." Journal of Financial Innovation.

13. Derman, E., & Kani, I. (1994). "Riding on a Smile." Risk, 7(2), 32-39.

14. Dupire, B. (1994). "Pricing with a Smile." Risk, 7(1), 18-20.

15. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide." Wiley Finance.

16. Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). "Volatility is rough." Quantitative Finance, 18(6), 933-949.

17. Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). "Managing Smile Risk." Wilmott Magazine, 84-108.

18. Heston, S. L. (1993). "A closed-form solution for options with stochastic volatility with applications to bond and currency options." Review of Financial Studies, 6(2), 327-343.

19. Heston, S. L., & Nandi, S. (2000). "A Closed-Form GARCH Option Valuation Model." Review of Financial Studies, 13(3), 585-625.

20. Hull, J., & White, A. (1987). "The Pricing of Options on Assets with Stochastic Volatilities." Journal of Finance, 42(2), 281-300.

21. Ko, S.B. (2025). "Real option valuation using Weibull distribution." Journal of Derivatives and Quantitative Studies.

22. Kou, S. G. (2002). "A Jump-Diffusion Model for Option Pricing." Management Science, 48(8), 1086-1101.

23. Kou, S. G., & Wang, H. (2004). "Option Pricing Under a Double Exponential Jump Diffusion Model." Management Science, 50(9), 1178-1192.

24. Madan, D. B., & Seneta, E. (1990). "The Variance Gamma (VG) Model for Share Market Returns." Journal of Business, 63(4), 511-524.

25. Madan, D. B., Carr, P. P., & Chang, E. C. (1998). "The Variance Gamma Process and Option Pricing." European Finance Review, 2(1), 79-105.

26. Merton, R. C. (1973). "Theory of Rational Option Pricing." Bell Journal of Economics and Management Science, 4(1), 141-183.

27. Merton, R. C. (1976). "Option Pricing When Underlying Stock Returns are Discontinuous." Journal of Financial Economics, 3(1-2), 125-144.

28. Rubinstein, M. (1983). "Displaced Diffusion Option Pricing." Journal of Finance, 38(1), 213-217.

29. Schoutens, W. (2003). "Lévy Processes in Finance: Pricing Financial Derivatives." Wiley.

30. Schoutens, W., & Teugels, J. L. (1998). "Lévy processes, polynomials and martingales." Communications in Statistics. Stochastic Models, 14(1-2), 335-349.

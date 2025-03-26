import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn

def hyperbolic_density(x, a):
    """
    Compute the density of a symmetric hyperbolic distribution, which is a
    special case of the generalized hyperbolic distribution with parameters:
      mu = 0, beta = 0, alpha = 1, delta = a.
    
    The density is given by:
        f(x) = (1/(2*a*K_1(a))) * exp(-sqrt(x^2 + a^2))
    
    When a = 0, the density reduces to the Laplace distribution:
        f(x) = (1/2) * exp(-|x|)
    
    Parameters:
      x : array_like
          Points at which to evaluate the density.
      a : nonnegative float
          Parameter corresponding to delta in the generalized hyperbolic form.
    
    Returns:
      f : array_like
          The probability density evaluated at x.
    """
    if np.isclose(a, 0.0):
        # Return Laplace density for a=0.
        return 0.5 * np.exp(-np.abs(x))
    else:
        c = 1.0 / (2 * a * kn(1, a))
        return c * np.exp(-np.sqrt(x**2 + a**2))

def excess_kurtosis(a):
    """
    Compute the excess kurtosis of the symmetric hyperbolic distribution,
    a special case of the generalized hyperbolic distribution:
    
        f(x) = (1/(2*a*K_1(a))) * exp(-sqrt(x^2 + a^2))
    
    For a = 0 the distribution reduces to the Laplace distribution which has
    excess kurtosis equal to 3.
    
    For a > 0 the excess kurtosis is calculated using:
    
        Excess Kurtosis = (K_1(a)*(K_5(a) - 3*K_3(a) + 2*K_1(a))) / (K_3(a) - K_1(a))**2 - 3
    
    where K_nu(a) are the modified Bessel functions of the second kind.
    
    Parameters:
      a : nonnegative float
          Parameter corresponding to delta in the generalized hyperbolic form.
    
    Returns:
      kurt : float
          The excess kurtosis of the distribution.
    """
    if np.isclose(a, 0.0):
        return 3.0
    else:
        K1 = kn(1, a)
        K3 = kn(3, a)
        K5 = kn(5, a)
        kurt = (K1 * (K5 - 3 * K3 + 2 * K1)) / ((K3 - K1) ** 2) - 3
        return kurt

def standard_deviation(a):
    """
    Compute the standard deviation of the symmetric hyperbolic distribution,
    a special case of the generalized hyperbolic distribution:
    
        f(x) = (1/(2*a*K_1(a))) * exp(-sqrt(x^2 + a^2))
    
    For a = 0 the distribution reduces to the Laplace distribution which has
    variance 2 (and standard deviation sqrt(2)).
    
    For a > 0 the variance is given by:
    
        Var(x) = E[x^2] = (a^2/(4*K_1(a)))*(K_3(a) - K_1(a))
    
    and the standard deviation is the square root of the variance.
    
    Parameters:
      a : nonnegative float
          Parameter corresponding to delta in the generalized hyperbolic form.
    
    Returns:
      std : float
          The standard deviation of the distribution.
    """
    if np.isclose(a, 0.0):
        return np.sqrt(2)
    else:
        K1 = kn(1, a)
        K3 = kn(3, a)
        variance = (a**2 / (4 * K1)) * (K3 - K1)
        std = np.sqrt(variance)
        return std
    
if __name__ == "__main__":
    # Parameter 'a' corresponds to the delta parameter in the
    # generalized hyperbolic distribution.
    plot_density = True
    fmt_r = "%10.4f"
    print(" ".join(["%10s"%label for label in ["a", "sd", "kurtosis"]]))
    for a in [0.0, 1.0, 2.0, 3.0, 10.0]:
        # Compute the excess kurtosis and standard deviation for the distribution with the chosen 'a'.
        print(fmt_r%a, fmt_r%standard_deviation(a), fmt_r%excess_kurtosis(a))

        if plot_density:
            # Create an array of x values for evaluating the density.
            x_values = np.linspace(-10, 10, 400)

            # Evaluate the hyperbolic density at these x values.
            density = hyperbolic_density(x_values, a)

            # Plot the hyperbolic density.
            plt.figure(figsize=(8, 4))
            plt.plot(x_values, density, label=f'a = {a}')
            plt.title("Special Case of the Generalized Hyperbolic Distribution")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            plt.grid(True)
            plt.show()

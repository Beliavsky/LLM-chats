import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def champernowne_norm_const(a):
    """
    Compute the normalization constant c for the Champernowne distribution:
    
        p(x) = c/(exp(x) + exp(-x) + a) = c/(2*cosh(x) + a)
    
    so that the density integrates to 1 over (-∞,∞).
    """
    # The integrand for the unnormalized density.
    integrand = lambda x: 1.0 / (2 * np.cosh(x) + a)
    I, err = quad(integrand, -np.inf, np.inf)
    return 1.0 / I

def champernowne_density(x, a):
    """
    Evaluate the normalized Champernowne density:
    
        p(x) = c/(exp(x) + exp(-x) + a) = c/(2*cosh(x) + a)
    
    where the normalization constant c is computed so that
        ∫_{-∞}^{∞} p(x) dx = 1.
    
    Parameters:
      x : array_like or float
          The point(s) at which to evaluate the density.
      a : float
          Parameter of the distribution (a > -2).
          
    Returns:
      p : array_like or float
          The probability density at x.
    """
    c = champernowne_norm_const(a)
    return c / (np.exp(x) + np.exp(-x) + a)

def moment(n, a):
    """
    Compute the nth moment of the Champernowne distribution.
    Because the distribution is symmetric the odd moments vanish;
    for even n the moment is given by:
    
        E[x^n] = 2 * ∫_0^∞ x^n * p(x) dx.
    
    Parameters:
      n : int
          The moment order.
      a : float
          Parameter of the Champernowne distribution.
          
    Returns:
      m : float
          The nth moment.
    """
    # For odd n, symmetry implies the moment is zero.
    if n % 2 == 1:
        return 0.0
    c = champernowne_norm_const(a)
    integrand = lambda x: x**n * c / (2 * np.cosh(x) + a)
    m, err = quad(integrand, 0, np.inf)
    return 2 * m

def standard_deviation(a):
    """
    Compute the standard deviation of the Champernowne distribution.
    
    The variance is E[x^2], so the standard deviation is the square root of that.
    
    Parameters:
      a : float
          Parameter of the Champernowne distribution.
          
    Returns:
      std : float
          The standard deviation.
    """
    var = moment(2, a)
    return np.sqrt(var)

def excess_kurtosis(a):
    """
    Compute the excess kurtosis of the Champernowne distribution.
    
    Excess kurtosis = (E[x^4] / (E[x^2])^2) - 3.
    
    Parameters:
      a : float
          Parameter of the Champernowne distribution.
          
    Returns:
      ek : float
          The excess kurtosis.
    """
    m2 = moment(2, a)
    m4 = moment(4, a)
    return m4 / (m2 * m2) - 3

if __name__ == "__main__":
    plot_density = False
    fmt_r = "%14.4f"
    print(" ".join(["%14s"%label for label in ["a", "sd", "kurtosis"]]))
    for a in [-1.999, -1.99, -1.9, -1.0, 0.0, 2.0, 10.0, 100.0, 1000.0, 1.0e6]:
        print(fmt_r%a, fmt_r%standard_deviation(a), fmt_r%excess_kurtosis(a))
        if plot_density:
            # Evaluate the density over a grid for plotting.
            x_values = np.linspace(-10, 10, 400)
            density_values = champernowne_density(x_values, a)

            plt.figure(figsize=(8, 4))
            plt.plot(x_values, density_values, label=f"a = {a}")
            plt.title("Normalized Champernowne Distribution")
            plt.xlabel("x")
            plt.ylabel("p(x)")
            plt.legend()
            plt.grid(True)
            plt.show()

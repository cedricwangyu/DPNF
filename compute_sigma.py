import numpy as np
from scipy import optimize
from scipy.stats import norm
from scipy.special import gammainc


def compute_mu_poisson(T, sigma, n, batch_size):
    """Compute mu from Poisson subsampling."""
    return np.sqrt((np.exp(sigma ** (-2)) - 1) * T) * batch_size / n


def delta_eps_mu(eps, mu):
    """Compute dual between mu-GDP and (epsilon, delta)-DP."""
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):
    """Compute epsilon from mu given delta via inverse dual."""
    return optimize.root_scalar(lambda x: delta_eps_mu(x, mu) - delta, bracket=[0, 700], method='brentq').root


def sigma_from_eps(delta, epsilon, T, total_size, batch_size):  # function computing sigma based on delta and epsilon
    return optimize.root_scalar(
        lambda x: eps_from_mu(compute_mu_poisson(T, x, total_size, batch_size), delta) - epsilon, bracket=[3, 100],
        method='brentq').root


def p_from_eps_delta(delta, epsilon, T, total_size, batch_size, tol_prob, tol_thr):
    return optimize.root_scalar(lambda x: gammainc(x / 2,
                                                   tol_thr ** 2 * batch_size ** 2 / sigma_from_eps(delta, epsilon, T,
                                                                                                   total_size,
                                                                                                   batch_size) ** 2 / 8) - tol_prob,
                                bracket=[1, 10 ** 9], method='brentq').root


def rho_from_p_eps_delta(delta, epsilon, T, total_size, batch_size, p, tol_thr):
    sigma = sigma_from_eps(delta, epsilon, T, total_size, batch_size)
    return gammainc(p / 2, tol_thr ** 2 * batch_size ** 2 / sigma ** 2 / 8)


def rho_from_p_sigma(sigma, p, T, batch_size, tol_thr):
    return gammainc(p / 2, T * tol_thr ** 2 * batch_size ** 2 / sigma ** 2 / 2)


def thr_from_p_sigma(sigma, p, T, batch_size, rho):
    return optimize.root_scalar(lambda x: gammainc(p / 2, T * x ** 2 * batch_size ** 2 / sigma ** 2 / 4) - rho,
                                bracket=[1e-16, 50],
                                method='brentq').root


def sigma_from_eps_delta(delta, epsilon, T, total_size, batch_size):
    mu = optimize.root_scalar(lambda x: delta_eps_mu(epsilon, x) - delta, bracket=[-100, 100], method='brentq').root
    print("mu: ", mu)
    return optimize.root_scalar(lambda x: compute_mu_poisson(T, x, total_size, batch_size) - mu, bracket=[0.1, 100],
                                method='brentq').root

import numpy as np
from scipy.stats import invwishart, multivariate_normal


def sample_niw_posterior(
    X_k: np.ndarray,
    mu0: np.ndarray,
    kappa0: float,
    nu0: float,
    psi0: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normal-Inverse-Wishart prior に基づいて、
    クラスタ k の平均 mu と共分散 Sigma を事後分布からサンプリングする。

    Prior:
        Sigma ~ Inv-Wishart(nu0, psi0)
        mu | Sigma ~ N(mu0, Sigma / kappa0)
    """

    n_k, dim = X_k.shape

    if n_k == 0:
        sigma = invwishart.rvs(df=nu0, scale=psi0, random_state=rng)
        mu = rng.multivariate_normal(mu0, sigma / kappa0)
        return mu, sigma

    x_bar = X_k.mean(axis=0)
    centered = X_k - x_bar
    scatter = centered.T @ centered

    kappa_n = kappa0 + n_k
    nu_n = nu0 + n_k

    mu_n = (kappa0 * mu0 + n_k * x_bar) / kappa_n

    diff = (x_bar - mu0).reshape(-1, 1)
    psi_n = psi0 + scatter + (kappa0 * n_k / kappa_n) * (diff @ diff.T)

    # 数値安定化
    psi_n = psi_n + 1e-6 * np.eye(dim)

    sigma = invwishart.rvs(df=nu_n, scale=psi_n, random_state=rng)
    mu = rng.multivariate_normal(mu_n, sigma / kappa_n)

    return mu, sigma


def log_gaussian_density(
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    多次元ガウス分布の対数密度を計算する。
    """

    dim = X.shape[1]
    sigma = sigma + 1e-6 * np.eye(dim)

    return multivariate_normal.logpdf(
        X,
        mean=mu,
        cov=sigma,
        allow_singular=False,
    )
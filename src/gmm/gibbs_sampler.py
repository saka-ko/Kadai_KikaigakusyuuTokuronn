from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .distributions import sample_niw_posterior, log_gaussian_density


@dataclass
class GibbsGMMResult:
    z: np.ndarray
    cluster_prob: np.ndarray
    pi_samples: np.ndarray
    mu_samples: np.ndarray
    final_pi: np.ndarray
    final_mu: np.ndarray
    final_sigma: np.ndarray


class BayesianGMMGibbsSampler:
    """
    Bayesian Gaussian Mixture Model を Gibbs サンプリングで推論するクラス。

    推定する変数:
        z      : 各データ点のクラスタ割当
        pi     : 混合比率
        mu     : 各クラスタの平均
        Sigma  : 各クラスタの分散共分散行列
    """

    def __init__(
        self,
        n_components: int = 3,
        alpha: float = 1.0,
        kappa0: float = 0.01,
        nu0: float | None = None,
        psi_scale: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.kappa0 = kappa0
        self.nu0 = nu0
        self.psi_scale = psi_scale
        self.rng = np.random.default_rng(random_state)

    def _initialize_parameters(
        self,
        X: np.ndarray,
        initial_z: np.ndarray | None = None,
    ) -> None:
        n_samples, dim = X.shape
        self.n_samples = n_samples
        self.dim = dim

        self.mu0 = np.zeros(dim)
        self.nu0_value = self.nu0 if self.nu0 is not None else dim + 2
        self.psi0 = self.psi_scale * np.eye(dim)

        if initial_z is None:
            self.z = self.rng.integers(0, self.n_components, size=n_samples)
        else:
            self.z = initial_z.copy()

        self.pi = np.ones(self.n_components) / self.n_components
        self.mu = np.zeros((self.n_components, dim))
        self.sigma = np.zeros((self.n_components, dim, dim))

        for k in range(self.n_components):
            X_k = X[self.z == k]
            self.mu[k], self.sigma[k] = sample_niw_posterior(
                X_k=X_k,
                mu0=self.mu0,
                kappa0=self.kappa0,
                nu0=self.nu0_value,
                psi0=self.psi0,
                rng=self.rng,
            )

    def _sample_pi(self) -> None:
        counts = np.bincount(self.z, minlength=self.n_components)
        alpha_post = self.alpha + counts
        self.pi = self.rng.dirichlet(alpha_post)

    def _sample_mu_sigma(self, X: np.ndarray) -> None:
        for k in range(self.n_components):
            X_k = X[self.z == k]
            self.mu[k], self.sigma[k] = sample_niw_posterior(
                X_k=X_k,
                mu0=self.mu0,
                kappa0=self.kappa0,
                nu0=self.nu0_value,
                psi0=self.psi0,
                rng=self.rng,
            )

    def _sample_z(self, X: np.ndarray) -> None:
        log_prob = np.zeros((self.n_samples, self.n_components))

        for k in range(self.n_components):
            log_prob[:, k] = np.log(self.pi[k] + 1e-12) + log_gaussian_density(
                X,
                self.mu[k],
                self.sigma[k],
            )

        log_prob = log_prob - logsumexp(log_prob, axis=1, keepdims=True)
        prob = np.exp(log_prob)

        cumulative_prob = np.cumsum(prob, axis=1)
        random_values = self.rng.random(self.n_samples)

        self.z = (cumulative_prob < random_values[:, None]).sum(axis=1)

    def fit(
        self,
        X: np.ndarray,
        n_iter: int = 500,
        burn_in: int = 200,
        thin: int = 5,
        initial_z: np.ndarray | None = None,
        verbose: bool = True,
    ) -> GibbsGMMResult:
        self._initialize_parameters(X, initial_z=initial_z)

        pi_samples = []
        mu_samples = []

        z_count = np.zeros((self.n_samples, self.n_components), dtype=np.int64)
        kept_samples = 0

        for it in range(1, n_iter + 1):
            self._sample_pi()
            self._sample_mu_sigma(X)
            self._sample_z(X)

            if it > burn_in and ((it - burn_in) % thin == 0):
                pi_samples.append(self.pi.copy())
                mu_samples.append(self.mu.copy())

                for k in range(self.n_components):
                    z_count[:, k] += self.z == k

                kept_samples += 1

            if verbose and (it == 1 or it % 50 == 0 or it == n_iter):
                counts = np.bincount(self.z, minlength=self.n_components)
                print(
                    f"iter={it:4d} | "
                    f"counts={counts.tolist()} | "
                    f"pi={np.round(self.pi, 3).tolist()}"
                )

        if kept_samples > 0:
            cluster_prob = z_count / kept_samples
            z_hat = cluster_prob.argmax(axis=1)
        else:
            cluster_prob = np.zeros((self.n_samples, self.n_components))
            z_hat = self.z.copy()

        return GibbsGMMResult(
            z=z_hat,
            cluster_prob=cluster_prob,
            pi_samples=np.array(pi_samples),
            mu_samples=np.array(mu_samples),
            final_pi=self.pi.copy(),
            final_mu=self.mu.copy(),
            final_sigma=self.sigma.copy(),
        )
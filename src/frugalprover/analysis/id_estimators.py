"""
TwoNN intrinsic-dimension estimator (Facco, D'Errico, Rodriguez, Laio, 2017).

Only needs distances to the 1st and 2nd nearest neighbor of each point, and is
robust to non-uniform density / curved manifolds -- which is why the pipeline
proposal cites it (see references/DIMENSIONALITY_METHODS.md) instead of PCA.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def twonn_dimension(X: np.ndarray, discard_fraction: float = 0.1) -> float:
    """Estimate the intrinsic dimension of the point cloud X (n_samples, n_features).

    discard_fraction: drop the top fraction of points by mu = r2/r1 before the
    linear fit, per the original paper (largest mu is noisiest / most affected
    by a handful of outlier neighbor distances).
    """
    n = X.shape[0]
    if n < 10:
        raise ValueError(f"Need at least ~10 points for a stable TwoNN estimate, got {n}")

    nbrs = NearestNeighbors(n_neighbors=3).fit(X)  # self + 1st + 2nd neighbor
    dist, _ = nbrs.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]

    # guard against duplicate points (r1 == 0), which break the mu ratio
    valid = r1 > 0
    mu = r2[valid] / r1[valid]

    mu_sorted = np.sort(mu)
    m = len(mu_sorted)
    keep = int(np.floor(m * (1 - discard_fraction)))
    mu_sorted = mu_sorted[:keep]

    F_emp = np.arange(1, len(mu_sorted) + 1) / m  # empirical CDF at kept points
    x = np.log(mu_sorted)
    y = -np.log(1 - F_emp)

    # linear regression through the origin: y = d * x
    d = float(np.sum(x * y) / np.sum(x * x))
    return d


def mle_dimension(X: np.ndarray, k: int = 10) -> float:
    """Levina-Bickel (2004) maximum-likelihood intrinsic-dimension estimator.

    A DIFFERENT nonlinear estimator from TwoNN (uses k nearest neighbors, not
    just the first two, and a different likelihood derivation) -- included so
    an ID-vs-difficulty trend can be checked against a second, independently
    derived nonlinear method, not just TwoNN. If TwoNN and MLE agree, that's
    much stronger evidence than either alone; if they disagree, that itself
    is worth reporting rather than picking whichever tells a nicer story.
    """
    n = X.shape[0]
    k = min(k, n - 1)
    if k < 2:
        raise ValueError(f"Need at least 3 points for a stable MLE estimate, got {n}")

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)  # column 0 is self (distance 0)
    dist = dist[:, 1:]  # (n, k): distances to the k nearest neighbors, ascending

    valid = dist[:, -1] > 0  # guard against duplicate/near-duplicate points
    dist = dist[valid]
    r_k = dist[:, -1:]
    r_j = dist[:, :-1]
    log_ratios = np.log(r_k / r_j)  # log(r_k / r_j) for j = 1..k-1
    local_dim = 1.0 / np.mean(log_ratios, axis=1)  # per-point MLE estimate
    return float(np.mean(local_dim))


def _sanity_check():
    """Points uniformly sampled on a d-dim linear subspace embedded in D-dim space."""
    rng = np.random.default_rng(0)
    n = 2000
    for true_d, D in [(2, 50), (5, 200), (10, 1000)]:
        Z = rng.uniform(-1, 1, size=(n, true_d))
        A = rng.normal(size=(true_d, D))
        X = Z @ A  # linear embedding preserves ID exactly
        est_twonn = twonn_dimension(X)
        est_mle = mle_dimension(X)
        print(f"true_d={true_d:>2} (embedded in D={D:>4}) -> TwoNN = {est_twonn:.2f}  MLE = {est_mle:.2f}")


if __name__ == "__main__":
    _sanity_check()

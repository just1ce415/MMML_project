import numpy as np
from tqdm import tqdm


def p_joint_probabilities(X, sigmas):
    """
    Arguments:
        X: matrix NxN with negative pairwise distances
    Returns:
        P: matrix NxN with join probabilities
    """
    P_cond = p_conditional_probabilities(X, sigmas)
    P = (P_cond + P_cond.T) / (2 * X.shape[0])
    return P


def p_conditional_probabilities(X, sigmas):
    """
    Arguments:
        X: matrix NxN with negative pairwise distances
        sigmas: vector N
    """
    sigmas = sigmas.reshape(-1, 1)

    X_norm = X / (2 * np.square(sigmas))
    # for numerical stability
    # smaller numbers won't overfill float
    X_stable = X_norm - np.max(X_norm, axis=1).reshape(-1, 1)
    X_exp = np.e**X_stable

    np.fill_diagonal(X_exp, 0)

    # for numerical stability
    # to take log later
    X_exp = X_exp + 1e-8

    return X_exp / X_exp.sum(axis=1).reshape(-1, 1)


def binary_search_perplexity(X, perplexity):
    """
    Find sigmas using binary search technique.
    Arguments:
        X: matrix NxN with negative pairwise distances
    """
    sigmas = []
    print("Finding sigmas")
    for i in tqdm(range(X.shape[0])):
        # create eval_fn
        eval_fn = lambda sigma: calc_perplexity(X, i, sigma)
        # send to binary search
        sigma = binary_search(eval_fn, perplexity)
        sigmas.append(sigma)
    sigmas = np.array(sigmas)
    return sigmas


def binary_search(eval_fn, target, low=1e-20, high=1e3, tol=1e-10, max_iters=10000):
    """
    Performs binary search for arbitrary function and target.
    """
    for i in range(max_iters):
        guess = (low + high) / 2
        val = eval_fn(guess)
        # if i % 1000 == 0:
        #     print(val, low, high)
        if target < val:
            high = guess
        else:
            low = guess
        if np.abs(val - target) < tol:
            # print(i)
            break
    return guess


def calc_perplexity(X, i, sigma):
    """
    Arguments:
        X: matrix NxN with negative pairwise distances
        sigmas: vector N
    """
    perp = 2 ** calc_shannon_entropy(X, i, sigma)
    return perp


def calc_shannon_entropy(X, i, sigma):
    """
    Arguments:
        X: matrix NxN with negative pairwise distances
        sigmas: vector N
    """
    P_cond = p_conditional_probabilities(X[i : i + 1, :], np.array([sigma]))
    ent = -(P_cond * np.log2(P_cond)).sum(axis=1)
    return ent


def pairwise_sq_distances(X):
    """
    Compute pairwise squared euclidean distances between samples
    of X.

    Arguments:
        X: matrix NxD with initial samples
    Returns:
        D: matrix NxN
    """
    sq_dist = np.square(X).sum(axis=1).reshape(-1, 1)
    D = sq_dist - 2 * (X @ X.T) + sq_dist.T
    return D


def q_joint_probabilities(Y):
    """
    Arguments:
        Y: matrix Nx2 with samples in reduced dimension
        space
    Returns:
        Q: matrix NxN
    """
    D = pairwise_sq_distances(Y)
    D_inv = np.reciprocal(1 + D)

    np.fill_diagonal(D_inv, 0)

    Q = D_inv / D_inv.sum()

    return Q, D_inv


def kl_divergence_grad(P, Q, Y, D_inv):
    pq_expand = np.expand_dims(P - Q, 2)

    y_i = np.expand_dims(Y, 1)
    y_j = np.expand_dims(Y, 0)
    y_diff = y_i - y_j

    d_expand = np.expand_dims(D_inv, 2)

    grad = 4 * (pq_expand * y_diff * d_expand).sum(axis=1)

    return grad


class TSNE:
    def __init__(self, num_iters, perplexity, learning_rate):
        self.num_iters = num_iters
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.momentum = 0.2

        self.sigmas = None

    def fit(self, X):
        rng = np.random.default_rng()
        N = X.shape[0]

        neg_sq_distances = -pairwise_sq_distances(X)

        if self.sigmas is None:
            self.sigmas = binary_search_perplexity(X, self.perplexity)

        P = p_joint_probabilities(neg_sq_distances, self.sigmas)

        Y = rng.normal(0, 1e-4, (N, 2))
        Y = self._gradient_descent(P, Y)

        return Y

    def _gradient_descent(self, P, Y):
        Y_1 = Y
        Y_2 = Y

        for t in tqdm(range(self.num_iters)):
            Q, D_inv = q_joint_probabilities(Y)

            grad = kl_divergence_grad(P, Q, Y, D_inv)

            Y = Y_1 + self._learning_rate(t) * grad + self._momentum(t) * (Y_1 - Y_2)

            Y_2 = Y_1
            Y_1 = Y

        return Y

    def _momentum(self, t):
        return self.momentum

    def _learning_rate(self, t):
        return self.learning_rate

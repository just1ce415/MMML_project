import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#################################################
# Probabilities
#################################################


def p_joint_probabilities(X, sigmas):
    """
    Arguments:
        X: matrix NxN with negative pairwise distances
    Returns:
        P: matrix NxN with joint probabilities
    """
    P_cond = p_conditional_probabilities(X, sigmas)
    P = (P_cond + P_cond.T) / (2 * X.shape[0])
    return P


def p_conditional_probabilities(X, sigmas):
    """
    Arguments:
        X: matrix KxN with negative pairwise squared distances
        sigmas: vector K with sigmas for corresponding samples
    """
    sigmas = sigmas.reshape(-1, 1)

    X_norm = X / (2 * np.square(sigmas))
    # for numerical stability
    # smaller numbers won't overfill float
    X_stable = X_norm - np.max(X_norm, axis=1).reshape(-1, 1)
    X_exp = np.exp(X_stable)

    np.fill_diagonal(X_exp, 0)

    # for numerical stability
    # to take log later
    X_exp = X_exp + 1e-8

    return X_exp / X_exp.sum(axis=1).reshape(-1, 1)


def q_joint_probabilities(Y):
    """
    Arguments:
        Y: matrix NxT with samples in reduced T-dimensional
        space
    Returns:
        Q: matrix NxN
        D_recip: matrix NxN with reciprocals to
        adjusted distances
    """
    D = pairwise_sq_distances(Y)
    D_recip = np.reciprocal(1 + D)

    np.fill_diagonal(D_recip, 0)

    Q = D_recip / D_recip.sum()

    return Q, D_recip


#################################################
# Sigmas
#################################################


def binary_search_perplexity(X, perplexity):
    """
    Find sigmas using binary search technique.

    Arguments:
        X: matrix NxN with negative squared pairwise distances
    """
    sigmas = []
    print("Finding sigmas")
    for i in tqdm(range(X.shape[0])):
        # create eval_fn
        X_i = X[i : i + 1, :]
        eval_fn = lambda sigma: calc_perplexity(X_i, sigma)
        # send to binary search
        sigma = binary_search(eval_fn, perplexity)
        sigmas.append(sigma)
    sigmas = np.array(sigmas)
    return sigmas


def calc_perplexity(X_i, sigma):
    """
    Arguments:
        X_i: row matrix 1xN with negative pairwise distances
        for point x_i to others
        sigma: corresponding value
    """
    perp = 2 ** calc_shannon_entropy(X_i, sigma)
    return perp


def calc_shannon_entropy(X_i, sigma):
    """
    Arguments:
        X_i: row matrix 1xN with negative pairwise distances
        for point x_i to others
        sigma: corresponding value
    """
    P_cond = p_conditional_probabilities(X_i, np.array([sigma]))
    ent = -(P_cond * np.log2(P_cond)).sum(axis=1)
    return ent


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


#################################################
# Utils
#################################################


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


def kl_divergence(P, Q):
    # for numerical stability
    # to divide
    np.fill_diagonal(Q, 1e-8)

    terms = P * np.log(P / Q)
    np.fill_diagonal(terms, 0)

    return terms.sum()


def kl_divergence_grad(P, Q, Y, D_recip, eef=1):
    pq_expand = np.expand_dims(eef*P - Q, 2)

    y_i = np.expand_dims(Y, 1)
    y_j = np.expand_dims(Y, 0)
    y_diff = y_i - y_j

    d_expand = np.expand_dims(D_recip, 2)

    grad = 4 * (pq_expand * y_diff * d_expand).sum(axis=1)
    return grad


def l2_penalty_grad(Y, beta):
    """
    Arguments:
        Y: matrix NxK with embedding samples
        beta: magnitude of the penalty term
    """
    return 2 * beta * Y


#################################################
# t-SNE
#################################################


class TSNE:
    """
    Arguments:
        n_components: number of output dimensions
        num_iters: number of iterations of optimization
        perplexity: soft number of neighbors for each point to form a cluster
        learning_rate: learning rate
        momentum: momentum term
        compression_period: number of iterations for early compression
        compression_term: L2 penalty factor for early compression
        initialization: type of initialization: 'random' or 'pca'
        ee: enable early exaggeration
        ee_iterations: if ee=True, number of iteration for early exaggeration
        verbose: enable plotting every 100 iterations
        color: if verbose=True, list ints of size N (number of samples) which represent encoded labels
        seed: seed for random generator
    """
    def __init__(
        self,
        n_components=2,
        num_iters=1000,
        perplexity=30,
        learning_rate=200,
        momentum=0.001,
        compression_period=20,
        compression_term=2,
        initialization='random',
        ee=False,
        ee_iterations=250,
        verbose=False,
        color=None,
        seed=0,
    ):
        self.num_iters = num_iters
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.compression_period = compression_period
        self.compression_term = compression_term
        self.initialization = initialization
        self.ee = ee
        self.ee_iterations = ee_iterations

        self.verbose = verbose
        self.color = color

        self.seed = seed

        self.sigmas = None
        self.Y = None

        self.metrics = {"kl_divergence": [], "spearman_corr": []}


    def fit(self, X):
        rng = np.random.default_rng(seed=self.seed)
        N = X.shape[0]

        if self.ee:
            self.learning_rate = N / 12

        neg_sq_distances = -pairwise_sq_distances(X)

        if self.sigmas is None:
            self.sigmas = binary_search_perplexity(neg_sq_distances, self.perplexity)

        P = p_joint_probabilities(neg_sq_distances, self.sigmas)

        if self.Y is None:
            if self.initialization == 'random':
                Y = rng.normal(0, 1e-4, (N, self.n_components))
            elif self.initialization == 'pca':
                pca = PCA(n_components=self.n_components)
                Y = pca.fit(X=X)

        Y = self._gradient_descent(P, Y)

        return Y

    def _gradient_descent(self, P, Y):
        Y_1 = Y
        Y_2 = Y

        for t in tqdm(range(self.num_iters)):
            Q, D_inv = q_joint_probabilities(Y)

            if self.ee and t < self.ee_iterations:
                grad = kl_divergence_grad(P, Q, Y, D_inv, eef=12)
            else:
                grad = kl_divergence_grad(P, Q, Y, D_inv)
            if t < self.compression_period:
                grad += l2_penalty_grad(Y, self.compression_term)

            Y = Y_1 - self._learning_rate(t) * grad + self._momentum(t) * (Y_1 - Y_2)

            Y_2 = Y_1
            Y_1 = Y

            if self.verbose and (t + 1) % 100 == 0:
                print(f"Max difference between Y_1 and Y_2: {np.max(abs(Y_1 - Y_2))}")
                plt.scatter(Y[:, 0], Y[:, 1], c=self.color)
                plt.show()

        self.metrics["kl_divergence"].append(kl_divergence(P, Q))

        return Y

    def _momentum(self, t):
        return self.momentum

    def _learning_rate(self, t):
        return self.learning_rate

    def plot_metrics(self):
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs[0][0] = plt.plot(self.metrics["kl_divergence"])
        axs[0][0].set_title("KL Divergence")
        axs[0][0].set_xlabel("Iteration")
        axs[0][0].set_ylabel("Value")
        fig.show()


#################################################
# Example
#################################################


def run_TSNE():
    tsne = TSNE(num_iters=10)

    X_test = np.array([[1, 2, 3], [1, 1, 1], [3, 3, 3]], dtype="float64")
    Y_test = np.power(X_test, 1e-4)

    Y = tsne.fit(X_test)

    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()


if __name__ == "__main__":
    run_TSNE()

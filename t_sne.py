import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.special import logsumexp


#################################################
# Random Walk
#################################################


def build_nn_graph(X, num_nn, perplexity):
    """
    Build nearest neighbors graph.

    Parameters:
        X (np.array): Array of samples
        num_nn (int): Number of nearest neighbors
        perplexity: Desired perplexity for all points
    Returns:
        graph (dict): Built graph
    """
    graph = {}
    print("Building a neighborhood graph...")
    for i in tqdm(range(X.shape[0])):
        sq_distances = np.square(np.linalg.norm(X[i] - X, axis=1))

        # to remove current point from nn
        sq_distances[i] = np.inf

        nn_idx = np.argpartition(sq_distances, num_nn)[:num_nn]

        neg_nn_distances = -sq_distances[nn_idx]
        # for numerical stability
        # e**k / e**m = e**(k-d) / e**(m-d)
        neg_nn_distances = neg_nn_distances - np.max(neg_nn_distances)
        nn_probs = np.exp(neg_nn_distances)
        nn_probs = nn_probs / nn_probs.sum()
        nn_probs = np.cumsum(nn_probs)

        graph[i] = [nn_idx, nn_probs]
    return graph


def find_terminal_node(graph, node, landmarks_set):
    while True:
        r = np.random.rand()
        nn_idx, nn_probs = graph[node]
        k = np.searchsorted(nn_probs, r)
        node = nn_idx[k]
        if node in landmarks_set:
            return node


def random_walk(graph, landmarks_idx, num_iters, labels=None):
    """
    Performs random walk on the graph from one landmark point to another.

    Parameters:
        graph (dict): Mapping from point to 2 lists. First list is
        indices of nearest heighbors. Second is probabilities to
        transit to them.
        landmarks_idx (np.array): Array with indices of landmark points.
        num_iters: Number of random walks starting from each point.
    """
    K = landmarks_idx.shape[0]

    landmarks_set = set(landmarks_idx)
    node_array_map = {landmarks_idx[i]: i for i in range(K)}
    p_cond = np.zeros((K, K))

    correct_paths = 0
    short_circuits = 0

    print("Running RW...")
    for t in tqdm(range(num_iters)):
        for i in range(K):
            start = landmarks_idx[i]
            finish = find_terminal_node(graph, start, landmarks_set)
            j = node_array_map[finish]
            p_cond[i][j] += 1

            if labels is not None:
                if labels[start] == labels[finish]:
                    correct_paths += 1
                if start == finish:
                    short_circuits += 1
    if labels is not None:
        print(f"Fraction of correct paths: {correct_paths / (num_iters * K)}")
        print(f"Fraction of short circuits: {short_circuits / (num_iters * K)}")

    np.fill_diagonal(p_cond, 0)

    # for numerical stability
    # to take log later
    p_cond = p_cond + 1e-8

    return p_cond / p_cond.sum(axis=1).reshape(-1, 1)


#################################################
# Probabilities
#################################################


def random_walk_joint_probabilities(graph, landmarks_idx, num_iters, labels):
    p_cond = random_walk(graph, landmarks_idx, num_iters, labels)

    p = (p_cond + p_cond.T) / (2 * p_cond.shape[0])

    return p


def p_joint_probabilities(X, sigmas):
    """
    Parameters:
        X: matrix NxN with negative pairwise distances
    Returns:
        P: matrix NxN with joint probabilities
    """
    P_cond = p_conditional_probabilities(X, sigmas)
    P = (P_cond + P_cond.T) / (2 * X.shape[0])
    return P


def p_conditional_probabilities(X, sigmas):
    """
    Parameters:
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
    Parameters:
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


def find_exact_sigmas(X, perplexity):
    """
    Find sigmas using binary search technique.

    Parameters:
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
    Parameters:
        X_i: row matrix 1xN with negative pairwise distances
        for point x_i to others
        sigma: corresponding value
    """
    perp = 2 ** calc_shannon_entropy(X_i, sigma)
    return perp


def calc_shannon_entropy(X_i, sigma):
    """
    Parameters:
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

    Parameters:
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
    pq_expand = np.expand_dims(eef * P - Q, 2)

    y_i = np.expand_dims(Y, 1)
    y_j = np.expand_dims(Y, 0)
    y_diff = y_i - y_j

    d_expand = np.expand_dims(D_recip, 2)

    grad = 4 * (pq_expand * y_diff * d_expand).sum(axis=1)
    return grad


def l2_penalty_grad(Y, beta):
    """
    Parameters:
        Y: matrix NxK with embedding samples
        beta: magnitude of the penalty term
    """
    return 2 * beta * Y


#################################################
# t-SNE
#################################################


class TSNE:
    """
    Parameters:
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
        random_walk: enable random walk
        data_ratio: if random_walk=True, float ratio of data that will be used for landmarks
        num_nn: if random_walk=True, number of nearest neighbors in neighborhood graph
        random_walk_num_iters: if random_walk=True, number of iterations of random walk
        verbose: enable plotting every 100 iterations
        labels: if verbose=True, list ints of size N (number of samples) which represent encoded labels
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
        initialization="random",
        ee=False,
        ee_iterations=250,
        random_walk=False,
        data_ratio=0.1,
        num_nn=20,
        random_walk_num_iters=1000,
        verbose=False,
        labels=None,
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

        self.random_walk = random_walk
        self.data_ratio = data_ratio
        self.num_nn = num_nn
        self.random_walk_num_iters = random_walk_num_iters

        self.verbose = verbose
        self.labels = labels

        # in case if want to continue optimization
        self.landmarks_idx = None
        self.graph = None
        self.sigmas = None
        self.X = None
        self.P = None
        self.Y = None

        self.metrics = {"kl_divergence": [], "spearman_corr": []}

    def get_highdimensional_similarities(self, X, seed=0):
        """
        Find joint probabilities matrix for given X.
        """
        rng = np.random.default_rng(seed)
        N = X.shape[0]

        if self.ee:
            self.learning_rate = N / 12

        if self.random_walk:
            self.landmarks_idx = rng.choice(
                np.arange(N), size=int(N * self.data_ratio), replace=False
            )
            self.landmarks_idx = np.sort(self.landmarks_idx)

            self.graph = build_nn_graph(X, self.num_nn, self.perplexity)

            P = random_walk_joint_probabilities(
                self.graph, self.landmarks_idx, self.random_walk_num_iters, self.labels
            )

            # further works with selected data points
            X = X[self.landmarks_idx]
            self.labels = self.labels[self.landmarks_idx]
            N = X.shape[0]
        else:
            neg_sq_distances = -pairwise_sq_distances(X)

            self.sigmas = find_exact_sigmas(neg_sq_distances, self.perplexity)

            P = p_joint_probabilities(neg_sq_distances, self.sigmas)

        self.P = P
        self.X = X

    # seperate fit() function with custom seed allows
    # a few runs for once calculated P
    def fit(self, seed=0):
        """
        Find mapping for earlier calculated probabilities matrix.
        """
        N = self.X.shape[0]
        rng = np.random.default_rng(seed)

        if self.Y is None:
            if self.initialization == "random":
                self.Y = rng.normal(0, 1e-4, (N, self.n_components))
            elif self.initialization == "pca":
                pca = PCA(n_components=self.n_components)
                self.Y = pca.fit(X=self.X)

        self.Y = self._gradient_descent(self.P, self.Y)

        return self.Y

    def _gradient_descent(self, P, Y):
        Y_1 = Y
        Y_2 = Y

        print("Running gradient descent...")
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
                plt.scatter(Y[:, 0], Y[:, 1], c=self.labels)
                plt.show()

            self.metrics["kl_divergence"].append(kl_divergence(P, Q))

        return Y

    def _momentum(self, t):
        return self.momentum

    def _learning_rate(self, t):
        return self.learning_rate

    def plot_metrics(self):
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.plot(self.metrics["kl_divergence"])
        axs.set_title("KL Divergence")
        axs.set_xlabel("Iteration")
        axs.set_ylabel("Value")
        fig.show()


#################################################
# Example
#################################################


def run_TSNE():
    tsne = TSNE(num_iters=10)

    X_test = np.array([[1, 2, 3], [1, 1, 1], [3, 3, 3]], dtype="float64")
    Y_test = np.power(X_test, 1e-4)

    tsne.get_highdimensional_similarities(X_test)
    Y = tsne.fit()

    plt.scatter(Y[:, 0], Y[:, 1])
    plt.show()

    print(tsne.metrics["kl_divergence"])
    tsne.plot_metrics()
    plt.show()


if __name__ == "__main__":
    run_TSNE()

import networkx as nx
import numpy as np
from numpy.linalg import inv



def generate_graph_laplacian(n=None, p=None, setting=None, name=None, verbose=False):
    """
    Generates an undirected graph laplacian from ER/SB model or loads nx graph
    """
    assert p is None or name is None, "Remove p or name"
    if p:
        if setting:
            raise Exception(f"Only p or setting (ER or SB model).")
        while True:
            G = nx.erdos_renyi_graph(n, p)
            if nx.is_connected(G):
                break
        if verbose:
            print(f"Generating ER graph with {G.number_of_nodes()} nodes")
    elif setting:
        sizes = [int(n * .75), n - int(n * .75)]
        # Homophilic setting
        probs_homoph = [
            [0.55, 0.07],
            [0.07, 0.50],
        ]
        # More heterogenous setting
        probs_heter = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        probs = probs_heter if setting == "hete" else probs_homoph
        G = nx.stochastic_block_model(sizes, probs, directed=False)
        G = G.to_undirected()
    elif name == "REAL_karate_club_graph":
        G = nx.karate_club_graph()
        if verbose:
            print(f"Loading karate_club graph with {G.number_of_nodes()} nodes")
    elif name == "REAL_florentine_families_graph":
        G = nx.florentine_families_graph()
        if verbose:
            print(f"Loading florentine_families_graph graph with {G.number_of_nodes()} nodes")
    elif name == "REAL_davis_southern_women_graph":
        G = nx.davis_southern_women_graph()
        if verbose:
            print(f"Loading davis_southern_women_graph graph with {G.number_of_nodes()} nodes")
    elif name == "REAL_les_miserables_graph":
        G = nx.les_miserables_graph()
        if verbose:
            print(f"Loading les_miserables_graph graph with {G.number_of_nodes()} nodes")
    else:
        raise Exception(f"{name} not implemented. set=REAL_les_miserables_graph,REAL_davis_southern_women_graph,REAL_florentine_families_graph,REAL_karate_club_graph")

    A = nx.to_numpy_array(G)
    D = np.diag(A.sum(axis=1))


    return D - A


def generate_diverse_arms(n_arms, n=None, p=None, setting=None, name=None, verbose=False):
    arms_set = []
    for _ in range(n_arms):
        arms_set.append(generate_graph_laplacian(n=n, p=p, setting=setting, name=name, verbose=verbose))
    return arms_set


def generate_arms(L, k, n_arms=None, delta_range=(0.5, 1.5), only_existing=False, rng=None):
    """
    Generate a list of Laplacians, each obtained by changing the weights of k edges
    with respect to the base Laplacian L by ADDING positive amounts.

    Args
    ----
    L : (n,n) ndarray
        Base Laplacian (assumed connected and valid).
    k : int
        Number of edges to modify per arm.
    n_arms : int or None
        Number of Laplacians to return. If None, returns exactly n (same as user's request).
    delta_range : (low, high)
        Each modified edge gets a weight increment delta ~ Uniform(low, high), low>0.
    only_existing : bool
        If True, sample edges only among those present in the base graph (where -L[i,j] > 0).
        If False, sample among all unordered pairs i<j (may add new edges).
    rng : np.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    arms : list of (n,n) ndarray
        Each element is a Laplacian L_r = L + sum_{t=1}^k delta_t * (e_i - e_j)(e_i - e_j)^T
    """
    rng = np.random.default_rng() if rng is None else rng
    n = L.shape[0]
    m = n if n_arms is None else int(n_arms)

    # candidate edge set
    if only_existing:
        E = [(i, j) for i in range(n) for j in range(i+1, n) if L[i, j] < 0]
        if len(E) < k:
            raise ValueError("Not enough existing edges to choose k distinct updates.")
    else:
        E = [(i, j) for i in range(n) for j in range(i+1, n)]

    lo, hi = float(delta_range[0]), float(delta_range[1])
    if lo <= 0 or hi <= 0 or hi < lo:
        raise ValueError("delta_range must be positive and satisfy 0 < low <= high.")

    arms = []
    for _ in range(m):
        # pick k distinct edges
        idxs = rng.choice(len(E), size=k, replace=False)
        Lr = L.copy()
        for idx in idxs:
            i, j = E[idx]
            delta = rng.uniform(lo, hi)
            # rank-one Laplacian update for edge (i,j)
            # B_ij has +1 on (i,i) and (j,j), and -1 on (i,j) and (j,i)
            Lr[i, i] += delta
            Lr[j, j] += delta
            Lr[i, j] -= delta
            Lr[j, i] -= delta
        arms.append(inv(np.eye(n) + Lr))

    return arms
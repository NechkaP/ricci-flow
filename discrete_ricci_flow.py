import numpy as np
from numpy.typing import DTypeLike
from ot import emd2
from scipy.sparse.csgraph import floyd_warshall
from sklearn.preprocessing import normalize
from typing import Callable


def MakeDiscreteRicciFlowIterator(
        N: int,
        alpha: float = 0.5,
        exp_power: float = 0.0,
        eps: float = 1.0,
        ElementType: DTypeLike = np.float64,
        inplace: bool = True
) -> Callable:
    MatrixType = np.ndarray[(N, N), ElementType]
    alpha = ElementType(alpha)

    # storages for intermediate calculation results
    graph_shortest_distances: MatrixType = np.zeros((N, N), dtype=ElementType)
    neighbor_probability_distribution: MatrixType = np.zeros((N, N), dtype=ElementType)
    olivier_ricci_curvatures: MatrixType = np.zeros((N, N), dtype=ElementType)
    non_neighbor_mask: np.ndarray[(N, N), np.bool8] = np.zeros((N, N), dtype=np.bool8)
    need_to_calculate_mask = True
    n_edges = 0

    def CalcShortestDistances(adj_matrix: MatrixType) -> MatrixType:
        nonlocal need_to_calculate_mask, n_edges
        np.copyto(
            dst=graph_shortest_distances,
            src=adj_matrix
        )
        if need_to_calculate_mask:
            np.equal(adj_matrix, ElementType(0), out=non_neighbor_mask)
            need_to_calculate_mask = False
            n_edges = N * N - np.sum(non_neighbor_mask)
            # print("n edges:", n_edges)

        graph_shortest_distances[non_neighbor_mask] = ElementType(np.Inf)
        floyd_warshall(
            graph_shortest_distances,
            directed=False,
            overwrite=True
        )
        return graph_shortest_distances

    def CalcEdgeOlivierRicciCurvatures(shortest_distances: MatrixType) -> MatrixType:
        probability_distribution = neighbor_probability_distribution
        np.copyto(
            dst=probability_distribution,
            src=shortest_distances
        )
        probability_distribution[non_neighbor_mask] = ElementType(0)
        # renorm
        probability_distribution *= n_edges / np.sum(probability_distribution)
        np.power(
            shortest_distances,
            exp_power,
            out=probability_distribution
        )
        probability_distribution *= -1
        np.exp(
            probability_distribution,
            out=probability_distribution
        )
        probability_distribution[non_neighbor_mask] = ElementType(0)
        normalize(probability_distribution, norm="l1", axis=1, copy=False)
        probability_distribution *= ElementType(1) - alpha
        np.fill_diagonal(probability_distribution, alpha)
        normalize(probability_distribution, norm="l1", axis=1, copy=False)

        for u in range(N):
            for v in range(u + 1, N):
                if not non_neighbor_mask[u, v]:  # u and v are neighbors
                    wasserstein_distance_uv = emd2(
                        probability_distribution[u],
                        probability_distribution[v],
                        shortest_distances
                    )
                    olivier_ricci_curvatures[u, v] = olivier_ricci_curvatures[v, u] = \
                        ElementType(1) - wasserstein_distance_uv / shortest_distances[u, v]
                    # print("Wasserstein distance between {} and {}: {}"
                    #   .format(u, v, wasserstein_distance_uv)
                    # )
                    # if u == 0 and v < 6:
                    #     print("Olivier-Ricci curvature between {} and {}: {}"
                    #     .format(u, v, olivier_ricci_curvatures[u, v])
                    #     )

        return olivier_ricci_curvatures

    def DoDiscreteRicciFlowIteration(adj_matrix: MatrixType) -> MatrixType:
        shortest_distances: MatrixType = CalcShortestDistances(adj_matrix)
        olivier_ricci_curvatures = CalcEdgeOlivierRicciCurvatures(shortest_distances)
        olivier_ricci_curvatures *= eps
        one_minus_edge_curvatures = np.subtract(
            ElementType(1),
            olivier_ricci_curvatures,
            out=olivier_ricci_curvatures
        )
        one_minus_edge_curvatures[non_neighbor_mask] = ElementType(0)
        new_adj_matrix = np.multiply(
            shortest_distances,
            one_minus_edge_curvatures,
            out=adj_matrix if inplace else None
        )
        # renorm
        new_adj_matrix *= n_edges / np.sum(new_adj_matrix)
        return new_adj_matrix

    DoDiscreteRicciFlowIteration.non_neighbor_mask = non_neighbor_mask
    DoDiscreteRicciFlowIteration.olivier_ricci_curvatures = olivier_ricci_curvatures

    return DoDiscreteRicciFlowIteration
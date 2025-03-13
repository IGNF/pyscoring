import pyscoring.metrics.edges
from shapely import Polygon


def test_edges():

    edges_computer = pyscoring.metrics.edges.EdgesMetric()
    assert edges_computer.compute() == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    pred = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    gt = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]

    edges_computer.update(pred, gt)
    assert edges_computer.compute() == (0, 0, 0, 0, 0, 0, 0, 0, 1, 1)

    edges_computer.update(pred, gt)
    assert edges_computer.compute() == (0, 0, 0, 0, 0, 0, 0, 0, 2, 2)

    edges_computer.reset()
    assert edges_computer.compute() == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    pred = [Polygon([(0.25, 0), (1.25, 0), (1.25, 1), (0.25, 1)])]
    gt = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]

    edges_computer.update(pred, gt)
    assert edges_computer.compute() == (
        0.25,
        0.25,
        0.25,
        0.25,
        11.25,
        11.25,
        90.0,
        90.0,
        1,
        1,
    )

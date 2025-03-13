import pyscoring.metrics.intrinsic
from shapely import Polygon


def test_intrinsic():

    intrinsic_computer = pyscoring.metrics.intrinsic.IntrinsicMetrics()
    assert intrinsic_computer.compute() == (0, 0, 0, 0, 0, 0)

    pred = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]

    intrinsic_computer.update(pred)

    assert intrinsic_computer.lengths == [1, 1, 1, 1]
    assert intrinsic_computer.areas == [1]
    assert intrinsic_computer.holes == 0
    assert intrinsic_computer.compute() == (1, 1, 1, 4, 4, 0)

    pred2 = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)], [[(3, 3), (4, 3), (4, 4), (3, 4)]]
        ),
    ]

    intrinsic_computer.compute()

    intrinsic_computer.update(pred2)

    assert intrinsic_computer.lengths == [1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10]
    assert intrinsic_computer.areas == [1, 1, 99]
    assert intrinsic_computer.holes == 1
    assert intrinsic_computer.compute() == (4, 101 / 3, 3, 12, 4, 1 / 3)

    intrinsic_computer.reset()
    assert intrinsic_computer.compute() == (0, 0, 0, 0, 0, 0)

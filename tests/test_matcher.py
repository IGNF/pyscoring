import pyscoring.metrics.matcher
from shapely import Polygon


def test_IOMA():
    matcher_ioma = pyscoring.metrics.matcher.MatchEngineIoMA(0.5, True, True)

    assert matcher_ioma.threshold == 0.5
    assert matcher_ioma.strict
    assert matcher_ioma.extend_matches

    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
    p3 = Polygon([(0, 0), (1, 0), (1, 3), (0, 3)])

    pno = Polygon([(100, 100), (10, 100), (10, 300), (100, 300)])
    pno2 = Polygon([(0.9, 0), (1.9, 0), (1.9, 1), (0.9, 1)])
    assert matcher_ioma.condition(p1, p2)
    assert matcher_ioma.condition(p1, p3)
    assert not matcher_ioma.condition(p1, pno)
    assert not matcher_ioma.condition(p1, pno2)


def test_IOU():
    matcher_ioma = pyscoring.metrics.matcher.MatchEngineIoU(0.5, False)

    assert matcher_ioma.threshold == 0.5
    assert not matcher_ioma.strict

    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
    p3 = Polygon([(0, 0), (1, 0), (1, 3), (0, 3)])

    pno = Polygon([(100, 100), (10, 100), (10, 300), (100, 300)])
    pno2 = Polygon([(0.9, 0), (1.9, 0), (1.9, 1), (0.9, 1)])
    assert matcher_ioma.condition(p1, p2)
    assert not matcher_ioma.condition(p1, p3)
    assert not matcher_ioma.condition(p1, pno)
    assert not matcher_ioma.condition(p1, pno2)

from wzc.geometry import foot_point, inside, polygon_area

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
# L-shape: the notch at top right (6..10, 6..10) is outside.
L_SHAPE = [(0, 0), (10, 0), (10, 6), (6, 6), (6, 10), (0, 10)]


def test_foot_point_is_bottom_centre():
    assert foot_point((10, 20, 30, 100)) == (20.0, 100.0)


def test_inside_outside_and_edge():
    assert inside((5, 5), SQUARE)
    assert not inside((15, 5), SQUARE)
    assert inside((10, 5), SQUARE)  # on the edge counts as inside


def test_concave_polygon():
    assert inside((2, 8), L_SHAPE)
    assert not inside((8, 8), L_SHAPE)


def test_polygon_area():
    assert polygon_area(SQUARE) == 100.0
    assert polygon_area([(0, 0), (5, 5), (10, 10)]) == 0.0

from src.geometry.line import HorizontalLine

def test_horizontal_line_from_height():
    h = 2160
    line = HorizontalLine.from_height(h, 0.75)
    assert int(line.y) == 1620

    line2 = HorizontalLine.from_height(h, 0.90)
    assert int(line2.y) == 1944
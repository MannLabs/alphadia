from alphadia.extraction.plotting import lighten_color

def test_lighten_color():
    
    color = "#000000"
    lightened_color = lighten_color(color, 0.5)
    assert lightened_color == (0.5, 0.5, 0.5)

    color = (0, 0, 0)
    lightened_color = lighten_color(color, 0.5)
    assert lightened_color == (0.5, 0.5, 0.5)

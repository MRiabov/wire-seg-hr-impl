def test_imports():
    import wireseghr
    import wireseghr.model as m
    import wireseghr.data as d

    assert hasattr(m, "SegFormerEncoder")
    assert hasattr(d, "WireSegDataset")

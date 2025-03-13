def test_install():
    try:
        import pyscoring

        done = True
    except Exception as e:
        print(e)
        done = False

    assert done

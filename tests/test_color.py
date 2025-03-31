import numpy as np

from visionsim.utils.color import linearrgb_to_srgb, srgb_to_linearrgb


def test_linearrgb_to_srgb_and_back():
    img = np.random.random((100, 100))
    round_trip_img = srgb_to_linearrgb(linearrgb_to_srgb(img))
    assert np.allclose(img, round_trip_img)


def test_srgb_to_linearrgb_and_back():
    img = np.random.random((100, 100))
    round_trip_img = linearrgb_to_srgb(srgb_to_linearrgb(img))
    assert np.allclose(img, round_trip_img)

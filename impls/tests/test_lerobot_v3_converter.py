import io

import numpy as np
from PIL import Image

from scripts.convert_lerobot_v3_to_lance import encode_jpeg, preprocess_frame, to_hwc_uint8


def test_lerobot_float_chw_image_becomes_rgb_uint8():
    frame = np.zeros((3, 6, 8), dtype=np.float32)
    frame[0] = 1.0
    converted = to_hwc_uint8(frame)
    assert converted.shape == (6, 8, 3)
    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted[..., 0], 255)
    np.testing.assert_array_equal(converted[..., 1:], 0)


def test_lerobot_frame_preprocessing_and_jpeg_shape():
    frame = np.arange(10 * 16 * 3, dtype=np.uint8).reshape(10, 16, 3)
    processed = preprocess_frame(frame, 8, 'long_edge')
    assert processed.shape == (5, 8, 3)
    encoded = encode_jpeg(frame, 8, 'long_edge', 95)
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.mode == 'RGB'
        assert image.size == (8, 5)

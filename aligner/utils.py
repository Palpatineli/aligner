from typing import Tuple, Optional
from PIL import Image
import numpy as np

def _bytescale(data: np.ndarray, cmin: Optional[int] = None, cmax: Optional[int] = None,
               high: int = 255, low: int = 0) -> np.ndarray:
    if data.dtype == np.uint8:
        return data
    assert high <= 255
    assert low >= 0
    assert high > low
    cmin = data.min() if cmin is None else cmin
    cmax = data.max() if cmax is None else cmax
    cscale = cmax - cmin
    scale = float(high - low) / (cscale if cscale > 0 else 1)
    return (((data - cmin) * scale + low).clip(low, high) + 0.5).astype(np.uint8)

def imresize(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image using Pillow. Because scipy/Pillow dependency often goes to dependency
    hell during upgrade. Way less testing than the actual version. Only works grayscale.
    Args:
        arr: the image
        size: target image size (y, x)
    Returns:
        the resized image array
    """
    byte_data = _bytescale(np.asarray(arr))
    y_size_s, x_size_s = byte_data.shape
    image_s = Image.frombytes('L', (x_size_s, y_size_s), byte_data.tostring())
    y_size_t, x_size_t = size
    image_t = image_s.resize((x_size_t, x_size_s), resample=2)  # bilinear
    return np.array(image_t)

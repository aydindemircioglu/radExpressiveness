import numpy as np
import skimage.draw
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import label
from scipy.spatial.distance import cdist
from skimage.draw import line
from scipy.ndimage import binary_fill_holes



def create_smiley(size=32, frown=False):
    smiley = np.zeros((size, size), dtype=np.uint8)
    center = size // 2

    rr, cc = skimage.draw.disk((center, center), center, shape=(size, size))
    smiley[rr, cc] = 1

    eye_y = center - size//5
    left_eye_x = center - size//6
    right_eye_x = center + size//6
    eye_radius = size//12

    rr, cc = skimage.draw.disk((eye_y, left_eye_x), eye_radius, shape=(size, size))
    smiley[rr, cc] = 0
    rr, cc = skimage.draw.disk((eye_y, right_eye_x), eye_radius, shape=(size, size))
    smiley[rr, cc] = 0

    # Mouth
    mouth_y = center + size//12 if not frown else center + size//4
    mouth_radius = size//4
    angles = np.linspace(0, np.pi, 100) if not frown else np.linspace(np.pi, 2*np.pi, 100)
    mouth_x = center + (mouth_radius * np.cos(angles)).astype(int)
    mouth_y_coords = mouth_y + (mouth_radius * np.sin(angles)).astype(int)

    for x, y in zip(mouth_x, mouth_y_coords):
        if 0 <= y < size and 0 <= x < size:
            rr, cc = skimage.draw.disk((y, x), eye_radius//2, shape=(size, size))
            smiley[rr, cc] = 0

    return smiley



def create_pm(size=32, frown=False):
    icon = np.zeros((size, size), dtype=np.uint8)
    center = size // 2

    rr, cc = skimage.draw.disk((center, center), center, shape=(size, size))
    icon[rr, cc] = 1
    thickness = max(1, size // 16)
    half_len = size // 4

    if not frown:
        rr, cc = skimage.draw.rectangle(
            start=(center - thickness // 2, center - half_len),
            end=(center + thickness // 2, center + half_len),
            shape=icon.shape
        )
        icon[rr, cc] = 0

        rr, cc = skimage.draw.rectangle(
            start=(center - half_len, center - thickness // 2),
            end=(center + half_len, center + thickness // 2),
            shape=icon.shape
        )
        icon[rr, cc] = 0
    else:
        rr, cc = skimage.draw.rectangle(
            start=(center - thickness, center - half_len),
            end=(center + thickness, center + half_len),
            shape=icon.shape
        )
        icon[rr, cc] = 0

    return icon


def create_spots(size=32, two_spots=False):
    icon = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    R = size // 3

    if not two_spots:
        rr, cc = skimage.draw.disk((center, center), R, shape=(size, size))
        icon[rr, cc] = 1
    else:
        r = int(R / np.sqrt(2))
        offset = size // 4
        rr1, cc1 = skimage.draw.disk((center - offset, center), r, shape=(size, size))
        rr2, cc2 = skimage.draw.disk((center + offset, center), r, shape=(size, size))

        icon[rr1, cc1] = 1
        icon[rr2, cc2] = 1

    return icon



def create_clover(size=32, four_spots=False):
    icon = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    R = size // 3

    if not four_spots:
        rr, cc = skimage.draw.disk((center, center), R/3, shape=(size, size))
        icon[rr, cc] = 1
    else:
        r = R / 2
        offsets = [
            (center - r, center), # Top
            (center + r, center), # Bottom
            (center, center - r), # Left
            (center, center + r)  # Right
        ]
        for r_offset, c_offset in offsets:
            rr, cc = skimage.draw.disk((r_offset, c_offset), R/4, shape=(size, size))
            icon[rr, cc] = 1

    return icon



def create_simple(size=16, frown=None):
    icon = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    rr, cc = skimage.draw.disk((center, center), size//2, shape=(size, size))
    icon[rr, cc] = 1
    return icon



def create_pattern(ptype, size=32, frown=False):
    # here are multiples, the pattern doesn matter, overall, it will yield similar
    # results, as described in the paper
    if ptype == "smiley":
        return create_smiley (size, frown)
    elif ptype == "pm":
        return create_pm (size, frown)
    elif ptype == "spots":
        return create_spots (size, frown)
    elif ptype == "clover":
        return create_clover (size, frown)
    elif ptype == "simple":
        return create_simple (size, frown)
    return None

#

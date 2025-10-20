import numpy as np

FRONTAL_R = [110, 112, 114, 116, 118,84,86,88,90,92,58,60,62,64,66,38,40,42]
FRONTAL_L = [111, 113, 115, 117, 85, 87, 89, 91, 59,61,63,65, 39,41]
FRONTAL_A = np.hstack((FRONTAL_R, FRONTAL_L))

CENTRAL_R = [22, 8, 0, 1, 9, 23, 20, 6, 4, 2, 10, 24, 18, 16, 14, 12, 26, 36, 34, 32, 30, 28, 56, 54, 52, 50]
CENTRAL_L = [21, 7, 5, 3, 11, 25, 19, 17, 15, 13, 27, 37, 35, 33, 31, 29, 57, 55, 53, 51]
CENTRAL_A = np.hstack((CENTRAL_R, CENTRAL_L))

TEMPORAL_R = [82,108, 134, 152, 80, 106, 132, 150, 78, 104, 130, 148, 76, 102, 128, 146, 74, 100, 126, 144]
TEMPORAL_L = [83,109, 135, 81, 107, 133, 151, 79, 105, 131, 147, 77, 149, 103, 129, 145, 75, 101, 127, 143]
TEMPORAL_A = np.hstack((TEMPORAL_R, TEMPORAL_L))

PARIETAL_R = [48,46,44,43,72,70, 68, 67]
PARIETAL_L = [49,47,45,73,71,69]
PARIETAL_A = np.hstack((PARIETAL_R, PARIETAL_L))

OCCIPITAL_R = [98,96,94,93,124, 122, 120, 119, 142, 140, 138, 136]
OCCIPITAL_L = [99,97,95,125, 123, 121, 141, 139, 137]
OCCIPITAL_A = np.hstack((OCCIPITAL_R, OCCIPITAL_L))


def channel2lobes(channel):

    if channel in PARIETAL_A:
        area = 'parietal'
    elif channel in TEMPORAL_A:
        area = 'temporal'
    elif channel in OCCIPITAL_A:
        area = 'occipital'
    elif channel in CENTRAL_A:
        area = 'central'
    else:
        area = 'frontal'
    
    return area

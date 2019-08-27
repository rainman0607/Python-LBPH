from skimage import feature
import numpy as np


class LocalBinaryPatterns:
    """
    :arg nump = NumPoints / neighbors
    :arg r = Radius
    """

    def __init__(self, nump, r):
        self.nump = nump
        self.r = r

    def compute(self, frame, eps=1e-7):
        lbp = feature.local_binary_pattern(frame, self.nump, self.r, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.nump + 3),
                                 range=(0, self.nump + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist

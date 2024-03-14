import numpy as np

from lib.evaluation.multimatch import docomparison


def multimatch(s1, s2, im_size):
    s1x = s1['X']
    s1y = s1['Y']
    s1t = s1['T']
    l1 = len(s1x)
    if l1 < 3:
        scanpath1 = np.ones((3, 3), dtype=np.float32)
        scanpath1[:l1, 0] = s1x
        scanpath1[:l1, 1] = s1y
        scanpath1[:l1, 2] = s1t[:l1]
    else:
        scanpath1 = np.ones((l1, 3), dtype=np.float32)
        scanpath1[:, 0] = s1x
        scanpath1[:, 1] = s1y
        scanpath1[:, 2] = s1t[:l1]
    s2x = s2['X']
    s2y = s2['Y']
    s2t = s2['T']
    l2 = len(s2x)
    if l2 < 3:
        scanpath2 = np.ones((3, 3), dtype=np.float32)
        scanpath2[:l2, 0] = s2x
        scanpath2[:l2, 1] = s2y
        scanpath2[:l2, 2] = s2t[:l2]
    else:
        scanpath2 = np.ones((l2, 3), dtype=np.float32)
        scanpath2[:, 0] = s2x
        scanpath2[:, 1] = s2y
        scanpath2[:, 2] = s2t[:l2]
    mm = docomparison(scanpath1, scanpath2, sz=im_size)
    return mm[0]
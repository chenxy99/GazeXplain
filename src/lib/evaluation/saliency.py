import numpy as np
import scipy.ndimage as filters

def cal_cc_score(pred_, gt_):
    eps = 1e-15
    pred = pred_ + eps
    gt = gt_ + eps
    pred = pred / np.sum(pred)
    gt = gt / np.sum(gt)

    if np.std(gt.reshape(-1)) <= 1e-8 or np.std(pred.reshape(-1)) <= 1e-8:
        cc_score = 1
    else:
        cc_score = np.corrcoef(pred.reshape(-1), gt.reshape(-1))[0][1]

    return cc_score

def cal_sim_score(pred_, gt_):
    eps = 1e-15
    pred = pred_ + eps
    gt = gt_ + eps
    pred = pred / np.sum(pred)
    gt = gt / np.sum(gt)

    sim_score = np.sum(np.minimum(pred, gt))

    return sim_score

def cal_kld_score(pred_, gt_):
    eps = 1e-15
    pred = pred_ + eps
    gt = gt_ + eps
    pred = pred / np.sum(pred)
    gt = gt / np.sum(gt)
    kl_score = gt * np.log(eps + gt / (pred + eps))
    kl_score = np.sum(kl_score)

    return kl_score
def cal_nss_score(pred_, gt_fix):
    pred = pred_ - np.mean(pred_)
    if np.max(pred) > 0:
        pred /= np.std(pred)

    nss_score = np.mean(pred[gt_fix['rows'], gt_fix['cols']])
    return nss_score

def cal_auc_score(pred, gt_fix, stepSize = .01, Nrand = 100000):
    salMap = pred
    S = salMap.reshape(-1)
    fixations = gt_fix
    Sth = np.asarray([salMap[y][x] for y, x in zip(fixations['rows'], fixations['cols'])])

    Nfixations = len(fixations['rows'])
    Npixels = len(S)

    # sal map values at random locations
    randfix = S[np.random.randint(Npixels, size=Nrand)]

    allthreshes = np.arange(0, np.max(np.concatenate((Sth, randfix), axis=0)), stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes) + 2)
    fp = np.zeros(len(allthreshes) + 2)
    tp[-1] = 1.0
    fp[-1] = 1.0
    tp[1:-1] = [float(np.sum(Sth >= thresh)) / Nfixations for thresh in allthreshes]
    fp[1:-1] = [float(np.sum(randfix >= thresh)) / Nrand for thresh in allthreshes]

    auc = np.trapz(tp, fp)
    return auc

def cal_sauc_score(pred, gt_fix, shufmap, stepSize=.01):
    """
    Computer SAUC score.
    """

    salMap = pred
    salMap = salMap - np.min(salMap)
    if np.max(salMap) > 0:
        salMap /= np.max(salMap)

    fixations = gt_fix
    rows = fixations['rows']
    cols = fixations['cols']
    Sth = np.asarray([ salMap[y][x] for y,x in zip(rows, cols) ])
    Nfixations = len(rows)

    shufMap = shufmap
    others = np.copy(shufMap)
    for y,x in zip(rows, cols):
        others[y][x] = 0

    ind = np.nonzero(others) # find fixation locations on other images
    nFix = shufMap[ind]
    randfix = salMap[ind]
    Nothers = sum(nFix)

    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1]=1.0
    fp[-1]=1.0
    tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

    auc = np.trapz(tp,fp)
    return auc

def filter_heatmap(att):
    att = filters.gaussian_filter(att, (22.4, 22.4))
    att /= att.max()
    return att
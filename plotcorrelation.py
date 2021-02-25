import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, roc_auc_score

def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

def corrfunc(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

path = "/nfs/dust/cms/user/flabe/MLCorner/TstarNN/output/"
foldername = "_layers3_nodes50_50_50_dropout0.5__TEST"

kappa = ["0.0","0.1","0.2"]

fpr = {}
tpr = {}
auc = {}
corr = {}
weights = {}
res = {}

# old line to compare
xold = np.load("/nfs/dust/cms/user/flabe/TstarTstar/data/numpy/OLDST.npy")
yold = np.load("/nfs/dust/cms/user/flabe/TstarTstar/data/numpy/OLDDNNoutput.npy")
wold = np.load("/nfs/dust/cms/user/flabe/TstarTstar/data/numpy/OLDweights.npy")

# print correlation
corr["old"] = corrfunc(xold, yold, wold)
print("correlation ", corr["old"])

counts, bins = np.histogram(xold, bins=12, weights=yold, range=(500,3000))
counts2, bins = np.histogram(xold, bins=12, range=(500,3000))
print(counts2)
weights["old"] = np.divide(counts, counts2)
weights["old"] = np.nan_to_num(weights["old"])
res["old"] = np.polyfit(bins[:-1], weights["old"], 1)

# all new lines
for k in kappa:
    x = np.load(path + "lambda"+ k + foldername + "/data/ST.npy")
    y = np.load(path + "lambda"+ k + foldername + "/data/DNNoutput.npy")
    w = np.load(path + "lambda"+ k + foldername + "/data/weights.npy")
    y = np.ravel(y)

    # print correlation
    corr[k] = corrfunc(x, y, w)
    print("correlation ", corr[k])

    counts, bins = np.histogram(x, bins=12, weights=y, range=(500,3000))
    counts2, bins = np.histogram(x, bins=12, range=(500,3000))
    print(counts2)
    weights[k] = np.divide(counts, counts2)
    weights[k] = np.nan_to_num(weights[k])
    res[k] = np.polyfit(bins[:-1], weights[k], 1)

fig = plt.figure()
plt.step(bins[:-1],weights["old"], label="planing" + " (corr: %.2f)" % (corr["old"]) )
for k in kappa:
    plt.step(bins[:-1],weights[k], label="DisCo " + k + " (corr: %.2f)" % (corr[k]) )
plt.legend(loc="lower right")
plt.xlabel("ST [GeV]")
plt.ylabel("classifier output")
fig.savefig(path+"/corr.png")

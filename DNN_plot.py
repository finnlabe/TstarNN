import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.metrics import confusion_matrix,accuracy_score,average_precision_score,roc_auc_score

def outputWeights(model, datadir):
    weights = model.layers[0].get_weights()[0]
    f = open(datadir+"/weights_layer1.txt")
    f.write(str(weights))
    print("Event weights: "+str(weights))
    return weights

def plot_accuracy(history, outputdir):
    fig_acc = plt.figure(figsize=(12,8))
    plt.plot(history.history['acc_all'],color="blue")
    plt.plot(history.history['val_acc_all'],color="red")
    plt.title('',fontsize=20)
    plt.ylabel('weighted accuracy',fontsize=30, labelpad=15)
    plt.xlabel('epoch',fontsize=30, labelpad=15)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(['training', 'validation'], loc='lower right',fontsize=30)
    plt.tight_layout()
    fig_acc.savefig(outputdir+'/plots/accuracy.png')


def plot_accuracy_partial(history, outputdir):
    fig_acc = plt.figure(figsize=(12,8))
    plt.plot(history.history['acc_sig'],color="blue")
    plt.plot(history.history['val_acc_sig'],color="red")
    plt.title('',fontsize=20)
    plt.ylabel('weighted accuracy',fontsize=20)
    plt.xlabel('epoch',fontsize=20)
    plt.legend(['train', 'validation'], loc='lower right',fontsize=20)
    fig_acc.savefig(outputdir+'/plots/accuracy_sig.png')

    fig_acc = plt.figure(figsize=(12,8))
    plt.plot(history.history['acc_bkg'],color="blue")
    plt.plot(history.history['val_acc_bkg'],color="red")
    plt.title('',fontsize=20)
    plt.ylabel('weighted accuracy',fontsize=20)
    plt.xlabel('epoch',fontsize=20)
    plt.legend(['train', 'validation'], loc='lower right',fontsize=20)
    fig_acc.savefig(outputdir+'/plots/accuracy_bkg.png')


def plot_loss(history, outputdir):
    fig_loss = plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'],color="blue")
    plt.plot(history.history['val_loss'],color="red")
    plt.title('',fontsize=20)
    plt.ylabel('loss',fontsize=30,labelpad=15)
    plt.xlabel('epoch',fontsize=30,labelpad=15)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(['training', 'validation'], loc='upper right',fontsize=30)
    plt.tight_layout()
    fig_loss.savefig(outputdir+'/plots/loss.png')


def plot_loss_partial(history, outputdir):
    fig_acc = plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'],color="black")
    plt.plot(history.history['nominalLoss'],color="red")
    plt.plot(history.history['DisCoLoss'],color="green")
    plt.title('',fontsize=20)
    plt.ylabel('loss',fontsize=30, labelpad=15)
    plt.xlabel('epoch',fontsize=30, labelpad=15)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(['total','BCE', 'DisCo'], loc='lower right',fontsize=30)
    plt.tight_layout()
    fig_acc.savefig(outputdir+'/plots/loss_partial.png')


def plot_output(predictions, labels, outputdir, name="test"):
    # Plotting outputs
    predictionsSig = []
    predictionsBkg = []

    for i in range(0,len(labels)):
        if(labels[i]==1):
            predictionsSig.append(predictions[i])
        if(labels[i]==0):
            predictionsBkg.append(predictions[i])

    predictionsSigPlot = np.asarray(predictionsSig)
    predictionsBkgPlot = np.asarray(predictionsBkg)

    fout = plt.figure(figsize=(12,8))
    plt.hist(predictionsBkgPlot, label='Background', histtype="stepfilled", color="blue", density=True, alpha=0.5)
    plt.hist(predictionsSigPlot, label='Signal', histtype="stepfilled", color="red", density=True, alpha=0.5)
    plt.title("")
    plt.xlabel("DNN output",fontsize=20)
    plt.ylabel("Number of events",fontsize=20)
    #plt.set_yscale('log')
    plt.legend(loc='upper left')
    fout.savefig(outputdir+'/plots/output_'+name+'.png')


def plot_ROC(predictions, labels, names, outputdir):
    fig_roc=plt.figure()
    for i in range(0, len(predictions)):
        prediction = predictions[i]
        label = labels[i]
        name = names[i]
        fpr, tpr, _ = roc_curve(label, prediction)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=name+' ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.legend(loc='lower right')
    print('AUC: %f' % roc_auc)
    fig_roc.savefig(outputdir+'/plots/ROC.png')

def plot_2D(output, HT, weights, outputdir):
    output = np.ravel(output)
    assert(len(output) == len(HT))
    H, xedges, yedges = np.histogram2d(output, HT, range=[[0,1],[0,3000]], weights=weights)
    H = H.T
    H_norm = normalize(H, norm="l1")
    fig1=plt.figure()
    mesh1 = plt.pcolormesh(xedges, yedges, H, norm=colors.LogNorm(vmin=H[H > 0].min(), vmax=H.max()))
    fig1.colorbar(mesh1)
    fig1.savefig(outputdir+'/plots/2D.png')
    fig=plt.figure()
    mesh = plt.pcolormesh(xedges, yedges, H_norm, norm=colors.LogNorm(vmin=H_norm[H_norm > 0].min(), vmax=H_norm.max()))
    fig.colorbar(mesh)
    fig.savefig(outputdir+'/plots/2Dnormed.png')

def plot_ST(STsig, weightsig, STbkg, weightbkg, outputdir):
    fig=plt.figure()
    plt.hist(STsig, range(0, 3000, 100), weights=weightsig, label="Signal", alpha = 0.5)
    plt.hist(STbkg, range(0, 3000, 100), weights=weightbkg, label="Background", alpha = 0.5)
    plt.legend()
    fig.savefig(outputdir+'/plots/ST.png')

def doHistoryPlots(history, outputdir):
    plot_accuracy(history, outputdir)
    plot_accuracy_partial(history, outputdir)
    plot_loss(history, outputdir)
    plot_loss_partial(history, outputdir)

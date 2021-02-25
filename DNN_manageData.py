import numpy as np
from root_numpy import root2array, tree2array, list_structures, rec2array

# getting input data for one file (BG-like)
def getDataOneFile(filename, branches_to_analyze, doSTreweighting=False):

    arr = rec2array(root2array(filename, treename='AnalysisTree',branches=branches_to_analyze))
    arr_weights = rec2array(root2array(filename, treename='AnalysisTree',branches=["evt_weight"]))
    arr_ST = rec2array(root2array(filename, treename='AnalysisTree',branches=["ST"]))
    if(doSTreweighting):
        arr_STweights = rec2array(root2array(filename, treename='AnalysisTree',branches=["ST_weight"]))
        #arr_weights = np.multiply(arr_weights, arr_STweights)
        arr_weights = arr_STweights
    return arr, arr_weights, arr_ST



# getting input data for multiple files (signal-like)
def getDataMultipleFiles(filename_base, mass_train, branches_to_analyze, doSTreweighting=False):

    arr_arr = []
    arr_weights_arr = []
    arr_ST_arr = []
    for mp in mass_train:
        filename = filename_base + str(mp) + ".root"
        arr_part = rec2array(root2array(filename, treename='AnalysisTree',branches=branches_to_analyze))
        arr_weights_part = rec2array(root2array(filename, treename='AnalysisTree',branches=["evt_weight"]))
        arr_ST_part = rec2array(root2array(filename, treename='AnalysisTree',branches=["ST"]))
        if(doSTreweighting):
            arr_STweights_part = rec2array(root2array(filename, treename='AnalysisTree',branches=["ST_weight"]))
            arr_weights_part = np.multiply(arr_weights_part, arr_STweights_part)
        arr_arr.append(arr_part)
        arr_weights_arr.append(arr_weights_part)
        arr_ST_arr.append(arr_ST_part)
    # TODO this part could be shortened by directly concatenating instead of first building an array
    arr = arr_arr[0]
    arr_weights = arr_weights_arr[0]
    arr_ST = arr_ST_arr[0]
    for i in range(1, len(mass_train)):
        arr = np.concatenate((arr,arr_arr[i]),axis=0)
        arr_weights = np.concatenate((arr_weights,arr_weights_arr[i]),axis=0)
        arr_ST = np.concatenate((arr_ST,arr_ST_arr[i]),axis=0)

    return arr, arr_weights, arr_ST


# ST cut
def ST_cut(val):
    return (val > 500)
def doSTcut(arr, arr_weights, arr_ST):
    ST_cut_vec = np.vectorize(ST_cut)
    msk = np.ravel(ST_cut_vec(arr_ST))
    arr = arr[msk]
    arr_weights = arr_weights[msk]
    arr_ST = arr_ST[msk]

    return arr, arr_weights, arr_ST

# function to shuffle all three arrays the same ways
def shuffle_three_arrays(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def shuffle_four_arrays(a, b, c, d):
    assert len(a) == len(b) == len(c) == len(d)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]

### normalization function ###
# if outputValues == True, mean and std is calculated and saved for later use
# else mean and std are taken from the save file
# need to run with True once for false to work!
def normalize(vector, length, outputdir, outputValues = False):
    if(outputValues):
        #open(outputdir+"/means.txt", "x")
        #open(outputdir+"/stds.txt", "x")
        file_mean = open(outputdir+"/means.txt", "w")
        file_std = open(outputdir+"/stds.txt", "w")
    else:
        file_mean = open(outputdir+"/means.txt", "r")
        file_std = open(outputdir+"/stds.txt", "r")

    for i in range(0,length):
        if(outputValues):
            mean = np.mean(vector[:,i])
            std = np.std(vector[:,i])
            if(std!=0):
                vector[:,i] = (vector[:,i] - mean)/std
            file_mean.write(str(mean)+"\n")
            file_std.write(str(std)+"\n")
        else:
            mean = float(file_mean.readline().strip())
            std = float(file_std.readline().strip())
            if(std!=0):
                vector[:,i] = (vector[:,i] - mean)/std

    return vector

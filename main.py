import numpy as np
from root_numpy import root2array, tree2array, list_structures, rec2array

import tensorflow as tf

import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as k

import pickle

from DNN_misc import manageOutputdir, acc_all, acc_bkg, acc_sig, nominalLoss_wrapper, DisCoLoss_wrapper, cross_entropy_DisCo, mean_squared_error_DisCo, layerblock
from DNN_manageData import getDataOneFile, getDataMultipleFiles, doSTcut, normalize, shuffle_four_arrays
from DNN_plot import doHistoryPlots, plot_output, plot_ROC, plot_2D, plot_ST

import argparse
parser = argparse.ArgumentParser(description='DNN for TstarTstar search.')
parser.add_argument('--DisCo', help="parameter determining decorrelation strength", default=0.0)
parser.add_argument('--layers', help="number of hidden layers", default=3)
parser.add_argument('--nodes', help="number of nodes per hidden layer", default=50)
parser.add_argument('--dropout', help="parameter determining dropout amount", default=0)
parser.add_argument('--STreweighting', action='store_const', help="do ST reweighting?", default=0, const=1)
args = parser.parse_args()
decorrelation_strength = float(args.DisCo)
number_hidden_nodes=[]
for layer in range(0, int(args.layers)):
    number_hidden_nodes.append(int(args.nodes))
dropout = float(args.dropout)
doSTreweighting = bool(args.STreweighting)

print("Training a network with decorrelation strength: "+str(decorrelation_strength))
if(doSTreweighting): print("Network is using ST-reweighted inputs.")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

##################################

# JUST THESE FOLLOWING LINES TO BE MODIFIED

##################################

inputpath = '/nfs/dust/cms/user/flabe/TstarTstar/data/Analysis/hadded/'
mass_train = [700, 800, 900, 1000, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
mass_notseen = 1200

number_inputs = 33
epochs=200
batch_size=2048
learning_rate = 0.0007
kernel_init = "random_uniform" # default is glorot_uniform

testrun = False
balanceSigBkg = False

test_split = 0.1
validation_split = 0.1
do_ST_cut = True
batch_norm = False

##########################################################################

# calculating some derived information
branches_to_analyze = ["DNN_Input_"+str(x) for x in range(number_inputs)]
number_layers=len(number_hidden_nodes)
filenameBkg = inputpath + 'uhh2.AnalysisModuleRunner.MC.TTbar.root'
filenameSig_base = inputpath + 'uhh2.AnalysisModuleRunner.MC.TstarTstar_M-'
filenameSig_notseen = inputpath + 'uhh2.AnalysisModuleRunner.MC.TstarTstar_M-'+str(mass_notseen)+'.root'
output_path = "/nfs/dust/cms/user/flabe/MLCorner/TstarNN/DisCoApproach/output/" # TODO later implement parametric

# def outputdir
outputdir = output_path+"/STweight"+str(doSTreweighting)+"_lambda"+str(decorrelation_strength)+"_layers"+str(number_layers)+"_nodes"
for node in number_hidden_nodes:
    outputdir += str(node)+"_"
outputdir += "dropout"+str(dropout)
if(testrun): outputdir+="__TEST"

# manage directory
outputdir = manageOutputdir(outputdir);


####################################
#                                  #
#  Data inclusion and preparation  #
#                                  #
####################################
print("Starting to include data")

# get input data
arrBkg, arrBkg_weights, arrBkg_ST = getDataOneFile(filenameBkg, branches_to_analyze, doSTreweighting)
arrSig, arrSig_weights, arrSig_ST = getDataMultipleFiles(filenameSig_base, mass_train, branches_to_analyze, False) # for the moment, signal is not reweighted. but can be changed here if needed

# do HT cut
if(do_ST_cut):
    arrBkg, arrBkg_weights, arrBkg_ST = doSTcut(arrBkg, arrBkg_weights, arrBkg_ST)
    arrSig, arrSig_weights, arrSig_ST = doSTcut(arrSig, arrSig_weights, arrSig_ST)

# throw away some data for test purposes
if(testrun):
    lenBkg = round(len(arrBkg)/10)
    lenSig = round(len(arrSig)/10)
    arrBkg = arrBkg[:lenBkg]
    arrBkg_weights = arrBkg_weights[:lenBkg]
    arrBkg_ST = arrBkg_ST[:lenBkg]
    arrSig = arrSig[:lenSig]
    arrSig_weights = arrSig_weights[:lenSig]
    arrSig_ST = arrSig_ST[:lenSig]

# check weighting
sig_weight = np.sum(arrSig_weights)
bkg_weight = np.sum(arrBkg_weights)

# status before reweighting
print("------ Input ------")
print("Signal event count: "+str(len(arrSig)))
print("Total signal weight: "+str(sig_weight))
print("Background event count: "+str(len(arrBkg)))
print("Total background weight: "+str(bkg_weight))

if(balanceSigBkg):
    if(len(arrSig) < len(arrBkg)):
        arrBkg = arrBkg[:len(arrSig)]
        arrBkg_weights = arrBkg_weights[:len(arrSig)]
        arrBkg_ST = arrBkg_ST[:len(arrSig)]
    else:
        arrSig = arrSig[:len(arrBkg)]
        arrSig_weights = arrSig_weights[:len(arrBkg)]
        arrSig_ST = arrSig_ST[:len(arrBkg)]

    sig_weight = np.sum(arrSig_weights)
    bkg_weight = np.sum(arrBkg_weights)
    weightFactor = sig_weight/bkg_weight
    arrBkg_weights = arrBkg_weights*weightFactor
    bkg_weight = np.sum(arrBkg_weights)

    print("------ For training ------")
    print("Signal event count: "+str(len(arrSig)))
    print("Total signal weight: "+str(sig_weight))
    print("Background event count: "+str(len(arrBkg)))
    print("Total background weight: "+str(bkg_weight))


plot_ST(arrSig_ST, arrSig_weights, arrBkg_ST, arrBkg_weights, outputdir)

print("Preparing data...")

# Creating Labels and joining samples
Values_array = np.concatenate((arrBkg,arrSig),axis=0)
label_bkg = np.zeros(arrBkg.shape[0])
label_sig = np.ones(arrSig.shape[0])
Labels_array = np.concatenate((label_bkg,label_sig),axis=0)
Weights_array = np.concatenate((arrBkg_weights, arrSig_weights), axis=0)
ST_array = np.concatenate((arrBkg_ST, arrSig_ST), axis=0)

# shuffling
Values_array, Labels_array, Weights_array, ST_array = shuffle_four_arrays(Values_array, Labels_array, Weights_array, ST_array)

# transforming arrays in float
Labels_array = Labels_array.astype(np.float)
Values_array = Values_array.astype(np.float)
Weights_array = Weights_array.astype(np.float)
ST_array = ST_array.astype(np.float)

# Normalizing value array
Values_array = normalize(Values_array,len(branches_to_analyze),outputdir+"/data/",True) # note the true, outputting values here!

# Splitting in Training and Test
random_vals = np.random.rand(len(Values_array))
msk_test = random_vals < test_split
msk_val = np.logical_and((random_vals > test_split), (random_vals < test_split + validation_split))
msk_train = random_vals > test_split + validation_split

Labels_array_test = Labels_array[msk_test]
Values_array_test = Values_array[msk_test]
Weights_array_test = Weights_array[msk_test]
ST_array_test = ST_array[msk_test]

Labels_array_val = Labels_array[msk_val]
Values_array_val = Values_array[msk_val]
Weights_array_val = Weights_array[msk_val]
ST_array_val = ST_array[msk_val]

Labels_array_train = Labels_array[msk_train]
Values_array_train = Values_array[msk_train]
Weights_array_train = Weights_array[msk_train]
ST_array_train = ST_array[msk_train]

# Flattening weights array
Weights_array_train = np.ravel(Weights_array_train)
Weights_array_val = np.ravel(Weights_array_val)
Weights_array_test = np.ravel(Weights_array_test)
ST_array_train = np.ravel(ST_array_train)
ST_array_val = np.ravel(ST_array_val)
ST_array_test = np.ravel(ST_array_test)

print(Labels_array_train)

# stack true, ST arrays and weight
Labels_array_train_stacked = np.column_stack((Labels_array_train, ST_array_train, Weights_array_train))
Labels_array_val_stacked = np.column_stack((Labels_array_val, ST_array_val, Weights_array_val))

assert(len(Values_array) == (len(Values_array_train) + len(Values_array_val) + len(Values_array_test)))
assert(len(Weights_array_val) == len(Values_array_val) == len(ST_array_val) == len(Labels_array_val))
assert(len(Weights_array_train) == len(Values_array_train) == len(ST_array_train) == len(Labels_array_train))
assert(len(Weights_array_test) == len(Values_array_test) == len(ST_array_test) == len(Labels_array_test))

#################################
#                               #
#  DNN definition and training  #
#                               #
#################################
print("Starting to set up DNN")

# checkpoint for taking "best" model
#print("Defining checkpoint strategy")
#checkpoint_path = "tmp/checkpoint-{epoch:02d}.hdf5"
#save_best_only_flag = (chosenModel == 0)
#checkpoint = tf.keras.callbacks.ModelCheckpoint(
#        filepath=checkpoint_path,
#        save_weights_only=True,
#        monitor="val_weighted_acc",
#        mode="max",
#        save_best_only=save_best_only_flag
#    )

# Inputs
print("Defining inputs")
input_layer = Input(shape=(len(branches_to_analyze),), name="input_vals")
y_true = Input(shape=(3,), name="y_true_ST_weights")

# Adding model layers
print("Adding first layer")
layers = input_layer
print("Adding more layers")
for i in range(0,number_layers):
    layers = layerblock(layers, number_hidden_nodes[i], kernel_init, dropout, batch_norm)

print("Adding output layer")
y_pred = Dense(1, activation='sigmoid', name='y_pred')(layers)

# defining model
print("constructing model")
model = Model(inputs=[input_layer], outputs=[y_pred])

print(model.summary())

# Compiling the model
print("compiling model")
model.compile(loss=mean_squared_error_DisCo(decorrelation_strength), optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate), metrics=[acc_all, acc_sig, acc_bkg, nominalLoss_wrapper(), DisCoLoss_wrapper(decorrelation_strength)], run_eagerly=True)

# Fit the model
print("Fitting model")
history = model.fit(Values_array_train, Labels_array_train_stacked, validation_data=(Values_array_val, Labels_array_val_stacked), shuffle = True, epochs=epochs, batch_size=batch_size)

# get a test model (without add inputs)
test_model = Model(inputs=input_layer, outputs=y_pred, name="test_only")

print(model.summary())

# getting the weights and the gradients
outputTensor = model.output # (Or model.layers[index].output)
listOfVariableTensors = model.trainable_weights

# save model file
arch = model.to_json()
# save the architecture string to a file somehow, the below will work
with open(outputdir+'/network/architecture.json', 'w') as arch_file:
    arch_file.write(arch)

model.save(outputdir+"/network/model_arch.h5")
model.save_weights(outputdir+"/network/model_weights.h5")

# write python dict to a file
mydict="{"
for i in range(1,len(branches_to_analyze)+1):
    mydict+="\""+str(branches_to_analyze[i-1])+"\": "+str(i)+", "
mydict+="}"
output_pkl = open(outputdir+'/network/variables.pkl', 'wb')
pickle.dump(branches_to_analyze, output_pkl)
output_pkl.close()

################################
#                              #
#  plotting and other outputs  #
#                              #
################################

doHistoryPlots(history, outputdir)

# get predictions on different datasets
predictions_train = model.predict(Values_array_train)
predictions_val = model.predict(Values_array_val)
predictions_test = model.predict(Values_array_test)

# saving predictions for correlation calculation
np.save(outputdir + "/data/DNNoutput", predictions_train)
np.save(outputdir + "/data/ST", ST_array_train)
np.save(outputdir + "/data/weights", Weights_array_train)

plot_output(predictions_test, Labels_array_test, outputdir)
plot_2D(predictions_train, ST_array_train, Weights_array_train, outputdir)
plot_ROC([predictions_train, predictions_val, predictions_test], [Labels_array_train, Labels_array_val, Labels_array_test], ["training", "validation", "test"], outputdir)

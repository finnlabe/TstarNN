# import assumes TF2
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# input path is read from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('inputpath', help='path to input model')
parser.add_argument('-o', "--outputpath", help='output path', default=".")
args = parser.parse_args()
inputpath = args.inputpath
outputpath = args.outputpath

## MODEL LOADING BLOCK ##

# model needs to be loaded
from DNN_misc import acc_bkg, acc_sig, nominalLoss_wrapper, DisCoLoss_wrapper, cross_entropy_DisCo

# defining any custom functions and preparing to passing them
param = 0.03
custom_objs = {}
custom_objs.update({"loss": cross_entropy_DisCo(param)})
custom_objs.update({"acc_sig": acc_sig})
custom_objs.update({"acc_bkg": acc_bkg})
custom_objs.update({"nominalLoss": nominalLoss_wrapper()})
custom_objs.update({"DisCoLoss": DisCoLoss_wrapper(param)})

# load model file
model = tf.keras.models.load_model(inputpath, custom_objects=custom_objs)

## END MODEL LOADING BLOCK ##

#path of the directory where you want to save your model
frozen_graph_filename = "frozen_graph"

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen graph def
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
    print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# writing graph (the .pb file)
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=outputpath,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)

# writing as readable text (for reference, not needed for inclusion)
#tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                  logdir=outputpath,
#                  name=f"{frozen_graph_filename}.pbtxt",
#                  as_text=True)


# writing the .config.pbtxt file
def format_name(name):
   if ':' not in name: return name
   else:
      return name.split(':')[0]

output_names = [format_name(i.name) for i in frozen_func.outputs]
with open(outputpath+"/"+frozen_graph_filename+".config.pbtxt", 'w') as config:
   for feed in model.inputs:
      #the first element is always the batch size (None in the graph, needs to be defined here)
      shape = [int(i) for i in feed.shape[1:]]
      shape = ['    dim { size : %s }' % i for i in shape]
      shape = '\n'.join(shape)
      config.write('''feed {
  id { node_name: "inputs" }
  shape {
%s
  }
}
''' % (shape))

   config.write('\n')
   for fetch in output_names:
      config.write('fetch {\n  id { node_name: "%s" }\n}\n' % fetch)

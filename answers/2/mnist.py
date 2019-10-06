"""
Run with
python mnist.py 2> /dev/null
(there is a lot of future compatibility warnings)

TODO: print learning curves
get training time
"""

import sys
import struct
import functools
import time

import numpy as np
import matplotlib . pyplot as pyplot

import tensorflow as tf

# parameters
hidden_layer_sizes = [ 300, 150 ]
activation_function = "relu"
optimizer = "rmsprop"
batch_size = 128


model_set = "many models"

nb_epochs = 2
test_model_params_list = [
  { "hidden_layer_sizes" : [ 300, 150 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 300, 150 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },
]


nb_epochs = 10
many_model_params_list = [
  { "hidden_layer_sizes" : [ 300, 150 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 200, 150, 100 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 784, 784 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 500, 100 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 100, 100 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 100, 70, 70 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 280, 111 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 150, 150 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 450, 300, 150 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 350, 200, 150, 100 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 120, 70 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 70, 46 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },
]


nb_epochs = 35
model_params_list_longer_training = [
  { "hidden_layer_sizes" : [ 300, 150 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 120, 70 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 70, 46 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },

  { "hidden_layer_sizes" : [ 784, 784 ],
    "activation_function" : "relu",
    "optimizer" : "rmsprop",
    "nb_epochs" : nb_epochs,
    "batch_size" : batch_size, },
  ]

if (model_set == "test"):
  nb_epochs = 2
  model_params_list = test_model_params_list
elif (model_set == "many models"):
  nb_epochs = 10
  model_params_list = many_model_params_list
elif (model_set == "longer training"):
  nb_epochs = 35
  model_params_list = model_params_list_longer_training
else:
  raise Exception ("Unknow choice of model set:" + str(model_set))

# constants
mnist_image_shape = (28, 28)
mnist_input_size = functools . reduce (lambda a, b : a * b, mnist_image_shape)
mnist_output_size = 10



# functions
def compute_number_of_parameters (hidden_layer_sizes):
  number_of_parameters = 0
  prev_size = mnist_input_size
  for cur_size in hidden_layer_sizes:
    number_of_parameters += (prev_size + 1) * cur_size
    prev_size = cur_size
  number_of_parameters += (prev_size + 1) * mnist_output_size
  return number_of_parameters


def build_the_model (hidden_layer_sizes, activation_function):
  
  model = tf . keras. Sequential ()

  model . add (tf . keras . layers . Flatten (input_shape = mnist_image_shape))

  for layer_size in hidden_layer_sizes:
    model . add (tf . keras . layers . Dense (layer_size, activation = activation_function))
  model . add (tf . keras . layers . Dense (mnist_output_size, activation = None))

  logits = model

  model = tf . keras . Sequential ([ logits, tf . keras . layers . Activation ("softmax") ])

  #model = {
  # "logits model" : logits,
  # "output" : output,
  #}
  return model


def read_the_mnist_data (data_set_name):
  # http://yann.lecun.com/exdb/mnist/ (idx format)
  # https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format
  # (in particular: https://stackoverflow.com/a/53181925/2148753)
  images_file_name = data_set_name + "-images-idx3-ubyte"
  labels_file_name = data_set_name + "-labels-idx1-ubyte"
  with open (images_file_name, "rb") as images_file:
    magic, nb_images = struct . unpack (">II", images_file . read (8))
    if (magic != 2051):
      raise Exception ("wrong file")
    nb_rows, nb_cols = struct . unpack (">II", images_file . read (8))
    image_shape = (nb_rows, nb_cols)
    images = np . fromfile (images_file, dtype = np . dtype (np . uint8) . newbyteorder (">")) . astype (np . float32) / 255.
    images = images . reshape ((nb_images, ) + image_shape)
  with open (labels_file_name, "rb") as labels_file:
    magic, nb_labels = struct . unpack (">II", labels_file . read (8))
    if (magic != 2049):
      raise Exception ("wrong file")
    if (nb_labels != nb_images):
      raise Exception ("nbr of labels is not equal to number of images")
    labels = np . fromfile (labels_file, dtype = np . dtype (np . uint8) . newbyteorder (">"))
    labels = labels . reshape ((nb_images, ))
  return nb_images, images, labels


def do_the_whole_training (hidden_layer_sizes, activation_function, optimizer, nb_epochs, batch_size, timing_callback):
  model = build_the_model (hidden_layer_sizes, activation_function)
  model . compile (optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = [ "accuracy", ])
  train_history = model . fit (train_images, train_labels, epochs = nb_epochs, batch_size = batch_size, callbacks = [ timing_callback, ])
  train_history . history ["times"] = timing_callback . times
  score = model . evaluate (test_images, test_labels)
  return (compute_number_of_parameters (hidden_layer_sizes), score, train_history)

"""
https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
(Marcin Możejko)
"""
class TimeHistoryCallback (tf . keras . callbacks . Callback):

  def on_train_begin (self, logs = {}):
    self . times = []

  def on_epoch_begin (self, epoch, logs = {}):
    self . epoch_start_time = time . time ()

  def on_epoch_end (self, epoch, logs = {}):
    self . times . append (time . time () - self . epoch_start_time)


# main

nb_train_images, train_images, train_labels = read_the_mnist_data ("train")
nb_test_images, test_images, test_labels = read_the_mnist_data ("t10k")

print ("Nbr train images: " + str (nb_train_images))
print ("Nbr test images: " + str (nb_test_images))


nb_models = len (model_params_list)
score_list = []
histories = []
nb_parameters_list = []

for model_params in model_params_list:
  timing_callback = TimeHistoryCallback ()
  (nb_parameters, score, train_history) = do_the_whole_training (model_params ["hidden_layer_sizes"],
                                                                 model_params ["activation_function"],
                                                                 model_params ["optimizer"],
                                                                 model_params ["nb_epochs"],
                                                                 model_params ["batch_size"],
                                                                 timing_callback)
  score_list . append (score)
  histories . append (train_history)
  nb_parameters_list . append (nb_parameters)



#print (score_list)

xticks = np . arange (nb_models)
pyplot . bar (xticks, nb_parameters_list)
pyplot . xticks (xticks, [ "M" + str (i) for i in range (nb_models) ])
pyplot . title ("Nb parameters")
pyplot . show ()


plot_x = np . arange (nb_epochs) + 1.
bar_width = 1 / (nb_models + 1)
for (i, history) in enumerate (histories):
  pyplot . bar (plot_x + (i + 0.5) * bar_width, history . history ["times"], bar_width)
pyplot . legend ([ "Model " + str (m) for m in range (len (model_params_list)) ])
#pyplot . xticks (plot_x)
pyplot . xlabel ("Epoch")
pyplot . title ("Timing plot per epoch")
pyplot . show ()

total_times = []
for history in histories:
  total_times . append (sum (history . history ["times"]))
xticks = np . arange (nb_models)
pyplot . bar (xticks, total_times)
xlabels = [ "M" + str (x) for x in xticks ]
pyplot . xticks (xticks, xlabels)
pyplot . title ("Total training time")
pyplot . show ()


for history in histories:
  pyplot . plot (plot_x, history . history ["acc"])
pyplot . legend ([ "Model " + str (m) for m in range (len (model_params_list)) ])
#pyplot . xticks (plot_x)
pyplot . xlabel ("Epoch")
pyplot . title ("Training accuracy plot")
pyplot . show ()

for history in histories:
  pyplot . plot (plot_x, history . history ["loss"])
pyplot . legend ([ "Model " + str (m) for m in range (len (model_params_list)) ])
#pyplot . xticks (plot_x)
pyplot . xlabel ("Epoch")
pyplot . title ("Trainig loss plot")
pyplot . show ()


xticks = np . arange (nb_models)
pyplot . bar (xticks, [ score [1] for score in score_list ])
xlabels = [ "M" + str (x) for x in xticks ]
pyplot . xticks (xticks, xlabels)
pyplot . title ("Test accuracies")
pyplot . show ()

xticks = np . arange (nb_models)
pyplot . bar (xticks, [ score [0] for score in score_list ])
xlabels = [ "M" + str (x) for x in xticks ]
pyplot . xticks (xticks, xlabels)
pyplot . title ("Test losses")
pyplot . show ()



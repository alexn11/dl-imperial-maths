"""
TODO:
- compute number of parameters
- implement a global pooling layer (as in GoogLeNet / ResNet)
- several models
DONE
- cumulated computing time (by epochs) instead of just one number
- batch normalization: https://keras.io/layers/normalization/

"""
import time
import struct
import functools

import numpy as np
import matplotlib . pyplot as pyplot

import tensorflow as tf

# constants
mnist_image_shape = (28, 28, 1)
mnist_input_size = functools . reduce (lambda a, b : a * b, mnist_image_shape)
mnist_output_size = 10

mode = "simple"

# input params
nb_epochs = 40

simple_convolution_layers_params = [
 {
  "nb filters" : 32,
  "filter size" : (3, 3),
  "strides" : (1, 1),
  "batch normalization" : False,
  "activation" : "relu",
  "max pooling size" : (2, 2),
  "dropout rate" : 0.25,
 },
 {
  "nb filters" : 128,
  "filter size" : (3, 3),
  "strides" : (1, 1),
  "batch normalization" : True,
  "activation" : "relu",
  "max pooling size" : (2, 2),
  "dropout rate" : 0.25,
 },
]
simple_dense_layers_params = [
 {
   "size" : 70,
   "activation" : "relu",
 },
 {
   "size" : 58,
   "activation" : "relu",
 },
]

simple_training_params = {
  "nb epochs" : nb_epochs,
  "batch size" : 64,
}

# kaggle example

nb_epochs = 40
kaggle_convolution_layers_params = [
 {
  "nb filters" : 24,
  "filter size" : (5, 5),
  "strides" : (1, 1),
  "batch normalization" : True,
  "activation" : "relu",
  "max pooling size" : (2, 2),
  "dropout rate" : 0.25,
 },
 {
  "nb filters" : 48,
  "filter size" : (5, 5),
  "strides" : (1, 1),
  "batch normalization" : True,
  "activation" : "relu",
  "max pooling size" : (2, 2),
  "dropout rate" : 0.25,
 },
]
kaggle_dense_layers_params = [
 {
   "size" : 256,
   "activation" : "relu",
 },
]

kaggle_training_params = {
  "nb epochs" : nb_epochs,
  "batch size" : 64,
}

# test mode

test_convolution_layers_params = [
 {
  "nb filters" : 10,
  "filter size" : (3, 3),
  "strides" : (1, 1),
  "activation" : "relu",
  "batch normalization" : False,
  "max pooling size" : (2, 2),
  "dropout rate" : 0.25,
 },
 {
  "nb filters" : 10,
  "filter size" : (3, 3),
  "strides" : (1, 1),
  "activation" : "relu",
  "batch normalization" : True,
  "max pooling size" : (2, 2),
  "dropout rate" : 0.25,
 },
]
test_dense_layers_params = [
 {
   "size" : 70,
   "activation" : "relu",
 },
 {
   "size" : 58,
   "activation" : "relu",
 },
]
test_training_params = {
  "nb epochs" : 2,
  "batch size" : 64,
}


# functions


def compute_number_of_parameters (model_params):
  # note: biases
  nb_params = 0
  data_shape = mnist_image_shape
  for conv_params in model_params ["conv layers"]:
    nb_conv_params = conv_params ["nb filters"] * (conv_params ["filter size"] [0] * conv_params ["filter size"] [0] * data_shape [2] + 1) # + 1 bias
    nb_params += nb_conv_params
    if (conv_params ["batch normalization"]):
      nb_params += conv_params ["nb filters"] * 4
    data_shape = [ (data_shape [i] - conv_params ["filter size"] [i] + 1) // conv_params ["max pooling size"] [i] for i in range (2) ] + [ conv_params ["nb filters"], ]
  prev_size = data_shape [0] * data_shape [1] * data_shape [2]
  for dense_params in model_params ["dense layers"]:
    nb_dense_params = dense_params ["size"] * (prev_size + 1)
    nb_params += nb_dense_params
    prev_size = dense_params ["size"]
  nb_dense_params = 10 * (prev_size + 1)
  nb_params += nb_dense_params
  return nb_params
  

def build_conv_block (model, conv_block_params, input_shape = None):

  if (input_shape is None):
    model . add (tf . keras . layers . Conv2D (conv_block_params ["nb filters"], conv_block_params ["filter size"], activation = conv_block_params ["activation"], data_format = "channels_last"))
  else:
    model . add (tf . keras . layers . Conv2D (conv_block_params ["nb filters"], conv_block_params ["filter size"], activation = conv_block_params ["activation"], input_shape = input_shape, data_format = "channels_last"))

  try:
    if (conv_block_params ["batch normalization"]):
      model . add (tf . keras . layers . BatchNormalization (axis = 3)) # channels last
  except (KeyError):
   # no batch normalization
   pass

  try:
    model . add (tf . keras . layers . MaxPooling2D (pool_size = conv_block_params ["max pooling size"]))
  except (KeyError):
    # no max pooling
    pass

  try:
    model . add (tf . keras . layers . Dropout (conv_block_params ["dropout rate"]))
  except (KeyError):
    # no dropout
    pass



def build_dense_layer (model, dense_layer_params):
  model . add (tf . keras . layers . Dense (dense_layer_params ["size"], activation = dense_layer_params ["activation"]))


def build_model (model_params):
  model = tf . keras . Sequential ()
  conv_layers_params = model_params ["conv layers"]
  build_conv_block (model, conv_layers_params [0], input_shape = model_params ["input shape"])
  for conv_block_params in conv_layers_params [ 1 : ]:
    build_conv_block (model, conv_block_params)
  model . add (tf . keras . layers . Flatten ())
  for dense_layer_params in model_params ["dense layers"]:
    build_dense_layer (model, dense_layer_params)
  model . add (tf . keras . layers . Dense (mnist_output_size, activation = "softmax"))
  return model



def read_the_mnist_data (data_set_name):
  # http://yann.lecun.com/exdb/mnist/ (idx format)
  # https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format
  # (in particular: https://stackoverflow.com/a/53181925/2148753)
  data_set_dir = "../mnist/"
  images_file_name = data_set_dir + data_set_name + "-images-idx3-ubyte"
  labels_file_name = data_set_dir + data_set_name + "-labels-idx1-ubyte"
  with open (images_file_name, "rb") as images_file:
    magic, nb_images = struct . unpack (">II", images_file . read (8))
    if (magic != 2051):
      raise Exception ("wrong file")
    nb_rows, nb_cols = struct . unpack (">II", images_file . read (8))
    image_shape = (nb_rows, nb_cols, 1) # channels last
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


"""
https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
(Marcin Możejko)
"""
class TimeAndEvaluationCallback (tf . keras . callbacks . Callback):

  def on_train_begin (self, logs = {}):
    self . training_times = []
    self . test_loss = []
    self . test_acc = []

  def on_epoch_begin (self, epoch, logs = {}):
    self . epoch_start_time = time . time ()

  def on_epoch_end (self, epoch, logs = {}):
    self . training_times . append (time . time () - self . epoch_start_time)
    loss, acc = self . model . evaluate (test_images, test_labels, batch_size = self . params ["batch_size"])
    self . test_loss . append (loss)
    self . test_acc . append (acc)


# derived parameters  

if (mode == "test"):
  convolution_layers_params = test_convolution_layers_params
  dense_layers_params = test_dense_layers_params
  training_params = test_training_params
elif (mode == "simple"):
  convolution_layers_params = simple_convolution_layers_params
  dense_layers_params = simple_dense_layers_params
  training_params = simple_training_params
elif (mode == "kaggle"):
  convolution_layers_params = kaggle_convolution_layers_params
  dense_layers_params = kaggle_dense_layers_params
  training_params = kaggle_training_params

model_params = {
  "input shape" : mnist_image_shape,
  "conv layers" : convolution_layers_params,
  "dense layers" : dense_layers_params,
}

nb_convolution_layers = len (model_params ["conv layers"])
if (nb_convolution_layers == 0):
  raise Exception ("no conv?")
nb_hidden_dense_layers = len (model_params ["dense layers"])


# main

nb_train_images, train_images, train_labels = read_the_mnist_data ("train")
nb_test_images, test_images, test_labels = read_the_mnist_data ("t10k")
print ("Nbr train images: " + str (nb_train_images))
print ("Nbr test images: " + str (nb_test_images))

#print (train_images [0] . shape)



model = build_model (model_params)

optimizer = tf . keras . optimizers . SGD (lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model . compile (loss = "sparse_categorical_crossentropy", optimizer = optimizer, metrics = [ "accuracy", ])

nb_params = compute_number_of_parameters (model_params)
print ("computed nb params=" + str (nb_params))


timing_and_evaluation_callback = TimeAndEvaluationCallback ()
train_history = model . fit (train_images, train_labels, epochs = training_params ["nb epochs"], batch_size = training_params ["batch size"], callbacks = [ timing_and_evaluation_callback, ])

total_training_time = sum (timing_and_evaluation_callback . training_times)
cumulated_training_times = [ sum (timing_and_evaluation_callback . training_times [ : e + 1]) for e in range (training_params ["nb epochs"]) ]
test_loss_history = timing_and_evaluation_callback . test_loss
test_accuracy_history = timing_and_evaluation_callback . test_acc


xlist = np . arange (training_params ["nb epochs"])

pyplot . plot (xlist, test_accuracy_history)
pyplot . plot (xlist, train_history . history ["acc"])
pyplot . legend (["Train acc", "Test acc"])
pyplot . title ("Accuracy")
pyplot . xlabel ("Epoch")
pyplot . show ()

pyplot . plot (xlist, test_loss_history)
pyplot . plot (xlist, train_history . history ["loss"])
pyplot . legend (["Train loss", "Test loss"])
pyplot . title ("Loss")
pyplot . xlabel ("Epoch")
pyplot . show ()


pyplot . plot (xlist, cumulated_training_times)
pyplot . title ("Training time")
pyplot . xlabel ("Epoch")
pyplot . show ()

print (total_training_time)



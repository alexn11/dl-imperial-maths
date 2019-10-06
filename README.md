# dl-imperial-maths
Code and assignment repository for the Imperial College Mathematics department Deep Learning course

## Course description

Deep Learning is a fast-evolving field in artificial intelligence that has been driving breakthrough advances in many application areas in recent years. It has become one of the most in-demand skillsets in machine learning and AI, far exceeding the supply of people with an expertise in this field. This course is aimed at PhD students within the Mathematics department at Imperial College who have no prior knowledge or experience of the field. It will cover the foundations of Deep Learning, including the various types of neural networks used for supervised and unsupervised learning. Practical tutorials in TensorFlow/PyTorch are an integral part of the course, and will enable students to build and train their own deep neural networks for a range of applications. The course also aims to describe the current state-of-the-art in various areas of Deep Learning, theoretical underpinnings and outstanding problems.

Topics covered in this course will include: 

* Convolutional and recurrent neural networks
* Reinforcement Learning
* Generative Adversarial Networks (GANs)
* Variational autoencoders (VAEs)
* Theoretical foundations of Deep Learning

There is a course website where registrations can be made and further logistical details can be found [here](https://www.deeplearningmathematics.com).

## Coursework

This fork contains my own answers to the coursework in the directory answers. Each answer is contained in its own directory numeroted according to the assignment number.

Note: since the end of course and the submission of my answers, TensorFlow version 2 has been released. This new version requires to adapt the implementation of ealier program using TensorFlow. I have used Tensorflow 2 in my answer from Assignment 2 on.

### Assignment 1

This assignment is on the implementation of a simple linear model using the basic functionalities of the TensorFlow library.

This answer use a previous version of TensorFlow.

### Assignment 2

This assignment is about solving the MNIST classification problem by using a model only with dense layers.

I use Keras sequential model to create the model.

The modifiable model and training parameters of the models are as follows:
- the number of dense layers and their respective sizes,
- the activation function used for all the layers (except the last one),
- the optimizer,
- the number of training epochs and the batch size.

I have tested several models with varying number of layers and sizes. The following results are plotted for each model:
- number of parameters,
- training loss and accuracy,
- final test loss and accuracy,
- training time for each epoch,
- the total training time.

Two sets of models have been trained.

The larger one contains 12 models which have been trained for 10 epochs with resulting test accuracies from 0.973 to 0.982. Model "M2" from this set is the one with most parameters and longest training time.

The smaller set of models contains 4 different models trained for 35 epochs. The resulting test accuracies lie from 0.975 to slightly above 0.982. There seems to be a strong postive correlation between the number of parameters of the model and the resulting accuracy except for large models. Indeed for the smalles models this correlation appears clearly. Meanwhile the models "M0" and "M3" are the models with the most parameters, with M3 having much more than M0. Despite this, the resulting accuracies seem to be roughly equal.

Other aspects that could have been explored is the dependence of the duration of the training on the type of optimizer (all the models have used the "rmsprop" optimizer). It would also be interesting to compare the model performances for different activation functions.

### Assignment 3

This assignment is about solving the MNIST classification problem by using a model based on a convolutional neural network.

Using Keras sequential model I have implemented a couple of models.

The modifiable model and training parameters are:
- the number of convolutional layers and for each layer:
  - the number of filters and their respective sizes,
  - the strides,
  - the use of batch renormalization (boolean),
  - the activation function,
  - the size of the max pooling layer,
  - the dropout rate;
- the number of dense layers (following the convolutional layers) and for each layer:
  - the size,
  - the activation function;
- the number of training epochs,
- the batch size.

I have a implemented and tested some arbitrary model (2 convolutional layers and 2 hidden dense layers) and a good example found on Kaggle [https://www.kaggle.com/c/digit-recognizer/overview]. The program has been improved from the assignment 2 and now plots the accuracy and loss evaluated on the test sets after each training epochs.

The final accuracy is above 0.99 for the first model (arbitrary model) after 40 epochs and a comparable result for the Kaggle model. I think the main reason for the succes of the first model is its size. This gives more than enough room for learning the proper fit. It is probably possible to improve it by removing a lot of redundant variables. The Kaggle model achieves its performance with much less parameters.




















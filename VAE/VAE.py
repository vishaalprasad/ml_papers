from collections import OrderedDict

import numpy as np
import tensorflow as tf

#Create wrapper functions to more easily create TF vars
def weights(shape):
  return tf.get_variable("W", shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())

def bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name="b")


class VAE:
  """
  Variational Autoencoders, from Kingma and Welling (2013).

  Credit to Jan Hendrik Metzen (https://jmetzen.github.io/2015-11-27/vae.html)
  for explanation and code to guide this implementation.

  Credit to Carl Doersch (2016) for the explanation on the theory.
  """

  def __init__(self, arch, transfer_func=tf.nn.relu, lr=0.001,
               batch_size=50):

    self.network_architecture = arch
    self.transfer_function = transfer_func
    self.batch_size = batch_size


  def _create_network(self):


  def _initialize_weights(self, n_hid_encoder1, n_hid_encoder_2,
                          n_hid_decoder1, n_hid_decoder2, n_input, n_z):
    all_weights = dict()
    all_weights['weights_encoder'] = {
            'h1': weights([n_input, n_hid_encoder1]),
            'h2': weights([n_hid_encoder1, n_hid_encoder2]),
            'out_mean': weights([n_hid_encoder2, n_z]),
            'out_log_sigma': weights([n_hid_encoder2, n_z])}
    all_weights['biases_encoder'] = {
            'b1': = bias([n_hid_encoder1]),
            'b2': bias([n_hid_encoder2]),
            'out_mean': bias([n_z]),
            'out_log_sigma': bias([n_z])}
    all_weights['weights_decoder'] = {
            'h1': weights([n_z, n_hid_encoder1]),
            'h2': weights([n_hid_encoder1, n_hid_encoder2]),
            'out_mean': weights([n_hid_encoder2, n_input]),
            'out_log_sigma': weights([n_hid_encoder2, n_input])}
    all_weights['biases_decoder'] = {
            'b1': = bias([n_hid_decoder1]),
            'b2': bias([n_hid_decoder2]),
            'out_mean': bias([n_input]),
            'out_log_sigma': bias([n_input])}
    return all_weights


  def encoder(self):
  	pass

  def decoder (self):
  	pass
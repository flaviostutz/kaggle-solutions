# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


  image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)





#### TENSOR FLOW GRAPH PREPARATION


batch_size = 128
hidden_nodes_1 = 1024
hidden_nodes_2 = 512
hidden_nodes_3 = 1024
learning_rate = 0.0001
beta = 0.005

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Placeholder to control dropout probability.
  keep_prob = tf.placeholder(tf.float32)

  # Variables.
  weights_1 = tf.Variable(tf.random_normal([image_size * image_size, hidden_nodes_1]))
  biases_1 = tf.Variable(tf.zeros([hidden_nodes_1]))
  weights_2 = tf.Variable(tf.random_normal([hidden_nodes_1, hidden_nodes_2]))
  biases_2 = tf.Variable(tf.zeros([hidden_nodes_2]))
  weights_3 = tf.Variable(tf.random_normal([hidden_nodes_2, hidden_nodes_3]))
  biases_3 = tf.Variable(tf.zeros([hidden_nodes_3]))
  weights_out = tf.Variable(tf.random_normal([hidden_nodes_3, num_labels]))
  biases_out = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  def forward_prop(input):
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(input, weights_1) + biases_1), keep_prob)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(   h1, weights_2) + biases_2), keep_prob)
    h3 = tf.nn.dropout(tf.nn.relu(tf.matmul(   h2, weights_3) + biases_3), keep_prob)
    return tf.matmul(h3, weights_out) + biases_out

  logits = forward_prop(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Add the regularization term to the loss.
  loss += beta * (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_out))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#  optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(forward_prop(tf_valid_dataset))
  test_prediction = tf.nn.softmax(forward_prop(tf_test_dataset))



#### TENSOR FLOW SESSION RUN


num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 1.0}
    feed_dict_w_drop = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.5}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict_w_drop)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(feed_dict=feed_dict), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict=feed_dict), test_labels))

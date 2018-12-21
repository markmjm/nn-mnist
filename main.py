import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# OUTLINE the model.
# ### Each number image id 28 X 28 picel = 784 ... Therefore input is 784 X 1
# ### we will have 10 outputs (0 - 9) using One_hot notation
input_size =784
output_size = 10
hidden_layer_size = 100 # each hidden layer will have x nodes

tf.reset_default_graph() # clear anything fro previous run that is still in memory

inputs = tf.placeholder(tf.float32, [None, input_size])  # input is [1,784].  We use None to say TF can figure out the first index
targets = tf.placeholder(tf.float32, [None, output_size])  # targets is [1,10].  We use None to say TF can figure out the first index

weights_1 = tf.get_variable('weights_1', [input_size, hidden_layer_size])
biases_1 = tf.get_variable('biases_1', [hidden_layer_size])
outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable('weights_2', [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable('biases_2', [hidden_layer_size])
outputs_2 = tf.nn.relu(tf.matmul(outputs_1, weights_2) + biases_2)

# weights_3 = tf.get_variable('weights_3', [hidden_layer_size, hidden_layer_size])
# biases_3 = tf.get_variable('biases_3', [hidden_layer_size])
# outputs_3 = tf.nn.tanh(tf.matmul(outputs_2, weights_3) + biases_3)
#
# weights_4 = tf.get_variable('weights_4', [hidden_layer_size, hidden_layer_size])
# biases_4 = tf.get_variable('biases_4', [hidden_layer_size])
# outputs_4 = tf.nn.tanh(tf.matmul(outputs_3, weights_4) + biases_4)
#
# weights_5 = tf.get_variable('weights_5', [hidden_layer_size, hidden_layer_size])
# biases_5 = tf.get_variable('biases_5', [hidden_layer_size])
# outputs_5 = tf.nn.tanh(tf.matmul(outputs_4, weights_5) + biases_5)

weights_3 = tf.get_variable('weights_3', [hidden_layer_size, output_size])
biases_3 = tf.get_variable('biases_3', [output_size])
outputs = tf.matmul(outputs_2, weights_3) + biases_3

loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)

mean_loss = tf.reduce_mean(loss)

optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_loss)

# Accuracy of prediction
out_equals_target = tf.equal(tf.argmax(outputs,1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

# Batching ans Stopping
sess = tf.InteractiveSession()
initializer = tf.global_variables_initializer()
sess.run(initializer)
batch_size = 100
batches_number = mnist.train.num_examples // batch_size
max_epochs = 100
prev_validation_loss = 9999999.

# Test
for epoch_counter in range(max_epochs):
    curr_epoch_loss = 0
    for batch_counter in range(batches_number):
        input_batch, target_batch = mnist.train.next_batch(batch_size)
        _, batch_loss = sess.run([optimize, mean_loss],
                                 feed_dict={inputs: input_batch, targets:target_batch})
        curr_epoch_loss += batch_loss
    curr_epoch_loss /= batches_number
    input_batch, target_batch = mnist.validation.next_batch(mnist.validation.num_examples)
    validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
                             feed_dict={inputs: input_batch, targets: target_batch})
    print('Epoch ' + str(epoch_counter+1) +
          '. Training loss: ' + '{0:.3f}'.format(curr_epoch_loss) +
          '. Validation loss: ' + '{0:.3f}'.format(validation_loss) +
          '. Validation accuracy: ' + '{0:.3f}'.format(validation_accuracy * 100) + '%'
          )
    if validation_loss > prev_validation_loss:
        break
    prev_validation_loss = validation_loss
print('End of Training')

#
#
# TEST MODEL
input_batch, target_batch = mnist.test.next_batch(mnist.test.num_examples)
test_accuracy = sess.run([accuracy], feed_dict={inputs: input_batch, targets: target_batch})
test_accuracy_percent = test_accuracy[0] * 100.0
print('Test accuracy: ' + '{0:.3f}'.format(test_accuracy_percent) + '%')

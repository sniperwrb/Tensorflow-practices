from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
# from tensorflow.python.tools import inspect_checkpoint
"""
Check variables:
tf.all_variables()
ckpt = tf.train.get_checkpoint_state("saves")
step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
inspect_checkpoint.print_tensors_in_checkpoint_file("saves/model.ckpt-%i"%step,None,None,True)
"""

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
num_labels = 10
def train_next_batch(n):
    c=np.random.randint(0,60000,n)
    d=train_images[c,:,:]/255.0
    d=np.reshape(d,(n,784))
    e=np.zeros((n,num_labels))
    e[range(n),train_labels[c]]=1
    return (d,e)

# Training Parameters
learning_rate = 0.01
gan_rate_bad = 0.1
gan_rate_good = 0.5
num_steps = 50000
batch_size = 100

display_step = 1000
save_step = 10000
#examples_to_show = 10

# Network Parameters
num_hidden = 128 
num_hidden_2 = num_labels # timbre features
num_hidden_1 = num_hidden - num_hidden_2 # num features not related to timbre
num_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
tf.reset_default_graph()
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_labels])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1]),name='encoder_h1'),
    'encoder_h2': tf.Variable(tf.random_normal([num_input, num_hidden_2]),name='encoder_h2'),
    'decoder_h': tf.Variable(tf.random_normal([num_hidden, num_input]),name='decoder_h'),
    'predicter_h1': tf.Variable(tf.random_normal([num_hidden_1, num_labels]),name='predicter_h1'),
    'predicter_h2': tf.Variable(tf.random_normal([num_hidden_2, num_labels]),name='predicter_h2'),
    
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1]),name='encoder_b1'),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2]),name='encoder_b2'),
    'decoder_b': tf.Variable(tf.random_normal([num_input]),name='decoder_b'),
    'predicter_b1': tf.Variable(tf.random_normal([num_labels]),name='predicter_b1'),
    'predicter_b2': tf.Variable(tf.random_normal([num_labels]),name='predicter_b2'),
}

# Building the encoder
def encoder1(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    return layer1

def encoder2(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer2

# Building the decoder
def decoder(x,y):
    # Decoder Hidden layer with sigmoid activation #2
    layer = tf.nn.sigmoid(tf.add(tf.matmul(tf.concat([x,y],1), weights['decoder_h']),
                                   biases['decoder_b']))
    return layer

# build the predictor
def predicter1(x):
    # the bad predicter
    layer = tf.nn.softmax(tf.add(tf.matmul(x, weights['predicter_h1']),
                                 biases['predicter_b1']))
    return layer

def predicter2(x):
    # the good predicter
    layer = tf.nn.softmax(tf.add(tf.matmul(x, weights['predicter_h2']),
                                 biases['predicter_b2']))
    return layer

# Construct model
encoder_op1 = encoder1(X)
encoder_op2 = encoder2(X)
decoder_op = decoder(encoder_op1, encoder_op2)
decoder_eval = decoder(encoder_op1, 0*encoder_op2)
pred_bad = predicter1(encoder_op1)
pred_good = predicter2(encoder_op2)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

C=tf.ones_like(Y)*(1.0/num_labels)

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + \
        gan_rate_bad * tf.reduce_mean(tf.pow(C - pred_bad, 2)) + \
        gan_rate_good * tf.reduce_mean(tf.pow(Y - pred_good, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    
    try:
        ckpt = tf.train.get_checkpoint_state("saves")
        step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        saver.restore(sess, "saves/model.ckpt-%i"%step)
    except:
        step=0
    else:
        print("Model restored at step %i" % step)
    """
    ckpt = tf.train.get_checkpoint_state("saves")
    step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
    saver.restore(sess, "saves/model.ckpt-%i"%step)
    """
        
    # Training
    for i in range(step+1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, batch_y = train_next_batch(batch_size)#mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
        if i % save_step == 0:
            save_path = saver.save(sess, "saves/model.ckpt", global_step=i)
            print("Model saved in path: %s" % save_path)
            saver.restore(sess,"saves/model.ckpt-%i"%i)
    
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    res_true = np.zeros(n*n)
    res_bad = np.zeros(n*n)
    res_good = np.zeros(n*n)
    for i in range(n):
        # MNIST test set
        batch_x, batch_y = train_next_batch(n)#mnist.test.next_batch(n)

        # Encode and decode the digit image
        g, h_bad, h_good = sess.run([decoder_eval, pred_bad, pred_good], \
                                    feed_dict={X: batch_x, Y: batch_y})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])
        
        res_true[i*n:(i+1)*n]=np.argmax(batch_y,axis=1)
        res_bad[i*n:(i+1)*n]=np.argmax(h_bad,axis=1)
        res_good[i*n:(i+1)*n]=np.argmax(h_good,axis=1)

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
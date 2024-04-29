#!/usr/bin/env python3
# Python 2 compatibility
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#import tensorflow as tf
import numpy as np
import os
import sys
from datetime import datetime
import utlty as util
# import matplotlib.pyplot as plt
# import matplotlib.backends.backend_pdf


#tf.disable_v2_behavior()

##################################################
#            READ PARAMETERS FOR DATA SET             #
##################################################
if len(sys.argv) != 6:
    print("usage: script.py nt nv seed n_epochs batch_size")
    print("nt:         number of training   samples")
    print("nv:         number of validation samples")
    print("seed:       seed for selecting training samples etc. (to make runs reproducible)")
    print("n_epochs:   number of training epochs")
    print("batch_size: size of batches for training (must not be bigger than nt)")
    quit()

nt = int(sys.argv[1])  # number of training samples
nv = int(sys.argv[2])  # number of validation samples
seed = int(sys.argv[3])  # seed for rng
n_epochs = int(sys.argv[4])  # number of training epochs
batch_size = int(sys.argv[5])  # number of samples in 1 batch
nsave = n_epochs//200  # saves approximately every 10% of the training
assert nsave != 0

# load dataset
print("Loading Dataset...\n")
data = util.DataContainer(nt, nv, seed, True)
print("total number of data:", data.num_data)
print("training examples:   ", data.num_train)
print("validation examples: ", data.num_valid)
print("test examples:       ", data.num_test)
print()

# fire up tensorflow
print("Loading TensorFlow...\n")


##################################################
#   PARAMETERS FOR NEURAL NETWORK AND TRAINING   #
##################################################
retrain_model = True  # whether or not to retrain the model from an existing file
model_parameter_save = "save/192000-222222666666999999111111111133133133151151151_loss0.10915892_2023-12-14-12-35-33-288"
# number of input variables
n_inputs = data.num_features
# number of neurons in each hidden layer (and how many hidden layers)
n_hidden = [22, 22, 22, 66, 66, 66, 99, 99, 99, 111, 111, 111, 133, 133, 133, 151, 151, 151]
# n_hidden = [59,59]
# n_hidden = [56,56]
assert len(n_hidden) >= 1  # check that we have at least 1 hidden layer
# number of output variables
n_outputs = data.num_outputs

# learning rate parameter
# learning_rate = 5e-4
global_step = tf.Variable(0, trainable=False)
zero_global_step_op = tf.assign(global_step, 0)
starter_learning_rate = 2.0e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.85, staircase=False)
# Passing global_step to minimize() will increment it at each step.

# lambda multiplier for L2 regularization (0 -> no regularization)
l2_lambda = 0.0  # default value I use is 1e-4!

# for saving logs
now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
root_logdir = "logs"
# encode NN architecture in string
NNstr = "NN-seed"+str(seed)+"-nt"+str(data.num_train)+"-nv" + \
    str(data.num_valid)+"-"+str(n_inputs)+"-"
for i in range(len(n_hidden)):
    NNstr = NNstr + str(n_hidden[i]) + "-"
NNstr = NNstr + str(n_outputs)
logdir = "{}/run_".format(root_logdir)+NNstr+"_{}/".format(now)

# create neural network architecture
# function to create a layer of neurons,
# getting X as input
# with n_out outputs


def neuron_layer(X, n_out, activation_fn=lambda x: x, scope=None, factor=2.0):
    with tf.variable_scope(scope):
        # define layer
        n_in = X.shape[1].value
        W = tf.Variable(tf.truncated_normal(
            [n_in, n_out], stddev=tf.sqrt(factor/(n_in+n_out))), name="W")
        b = tf.Variable(tf.truncated_normal([n_out], stddev=0), name="b")
        y = activation_fn(tf.add(tf.matmul(X, W), b))

        # L2 loss term for regularization
        l2_W = tf.nn.l2_loss(W, name="l2_W")

        # add to collections
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2_W)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
        tf.add_to_collection(tf.GraphKeys.BIASES, b)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, y)

        # create histogram summaries for monitoring the weights and biases
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases",  b)

        return y

# function to create a Multilayer Perceptron getting X as feature input and Z as atom idendity encoding (as integer)
# n_hidden is a list containing the number of neurons in the hidden layers (at least 1 hidden layer)
# and n_outputs is how many output variables the MLP should have.


def MLP(X, n_hidden, n_outputs, activation_fn=lambda x: x, scope=None, factor=2.0):
    assert len(n_hidden) >= 1  # check that there is at least 1 hidden layer
    with tf.name_scope(scope):
        hidden = []
        hidden.append(neuron_layer(
            X, n_hidden[0], activation_fn=activation_fn, scope="hidden0", factor=factor))
        hidden.append(neuron_layer(hidden[0], n_hidden[1],
                                   activation_fn=activation_fn, scope="hidden1", factor=factor))
        hidden.append(neuron_layer(hidden[1], n_hidden[2], scope="hidden2", factor=factor))
        hidden[2] = actf(hidden[2]+hidden[0])

        hidden.append(neuron_layer(hidden[2], n_hidden[3],
                                   activation_fn=activation_fn, scope="hidden3", factor=factor))
        hidden.append(neuron_layer(hidden[3], n_hidden[4],
                                   activation_fn=activation_fn, scope="hidden4", factor=factor))
        hidden.append(neuron_layer(hidden[4], n_hidden[5], scope="hidden5", factor=factor))
        hidden[5] = actf(hidden[5]+hidden[3])

        hidden.append(neuron_layer(hidden[5], n_hidden[6],
                                   activation_fn=activation_fn, scope="hidden6", factor=factor))
        hidden.append(neuron_layer(hidden[6], n_hidden[7],
                                   activation_fn=activation_fn, scope="hidden7", factor=factor))
        hidden.append(neuron_layer(hidden[7], n_hidden[8], scope="hidden8", factor=factor))
        hidden[8] = actf(hidden[8]+hidden[6])

        hidden.append(neuron_layer(hidden[8], n_hidden[9],
                                   activation_fn=activation_fn, scope="hidden9", factor=factor))
        hidden.append(neuron_layer(hidden[9], n_hidden[10],
                                   activation_fn=activation_fn, scope="hidden10", factor=factor))
        hidden.append(neuron_layer(hidden[10], n_hidden[11], scope="hidden11", factor=factor))
        hidden[11] = actf(hidden[11]+hidden[9])

        hidden.append(neuron_layer(hidden[11], n_hidden[12],
                                   activation_fn=activation_fn, scope="hidden12", factor=factor))
        hidden.append(neuron_layer(hidden[12], n_hidden[13],
                                   activation_fn=activation_fn, scope="hidden13", factor=factor))
        hidden.append(neuron_layer(hidden[13], n_hidden[14], scope="hidden14", factor=factor))
        hidden[14] = actf(hidden[14]+hidden[12])

        hidden.append(neuron_layer(hidden[14], n_hidden[15],
                                   activation_fn=activation_fn, scope="hidden15", factor=factor))
        hidden.append(neuron_layer(hidden[15], n_hidden[16],
                                   activation_fn=activation_fn, scope="hidden16", factor=factor))
        hidden.append(neuron_layer(hidden[16], n_hidden[17], scope="hidden17", factor=factor))
        hidden[17] = actf(hidden[17]+hidden[15])

        # hidden.append(neuron_layer(hidden[17], n_hidden[18],
        #                            activation_fn=activation_fn, scope="hidden18", factor=factor))
        # hidden.append(neuron_layer(hidden[18], n_hidden[19],
        #                            activation_fn=activation_fn, scope="hidden19", factor=factor))
        # hidden.append(neuron_layer(hidden[19], n_hidden[20], scope="hidden20", factor=factor))
        # hidden[20] = actf(hidden[20]+hidden[18])

        # hidden.append(neuron_layer(
        #     X, n_hidden[0], activation_fn=activation_fn, scope="hidden0", factor=factor))
        # hidden.append(neuron_layer(hidden[0], n_hidden[1],
        #                            activation_fn=activation_fn, scope="hidden1", factor=factor))
        # hidden.append(neuron_layer(hidden[1], n_hidden[2], scope="hidden2", factor=factor))
        # hidden[2] = actf(hidden[2]+hidden[0])
      #
      # #hidden.append(neuron_layer(hidden[2], n_hidden[3], activation_fn=activation_fn, scope="hidden3", factor=factor))
      #   hidden.append(neuron_layer(hidden[2], n_hidden[3],
      #                              activation_fn=activation_fn, scope="hidden3", factor=factor))
      #   hidden.append(neuron_layer(hidden[3], n_hidden[4], scope="hidden4", factor=factor))
      #   hidden[4] = actf(hidden[4]+hidden[2])
      #
      #  #hidden.append(neuron_layer(hidden[5], n_hidden[6], activation_fn=activation_fn, scope="hidden6", factor=factor))
      #   hidden.append(neuron_layer(hidden[4], n_hidden[5],
      #                              activation_fn=activation_fn, scope="hidden5", factor=factor))
      #   hidden.append(neuron_layer(hidden[5], n_hidden[6], scope="hidden6", factor=factor))
      #   hidden[6] = actf(hidden[6]+hidden[4])

        return tf.nn.softplus(neuron_layer(hidden[len(n_hidden)-1], n_outputs, scope="output", factor=2.0))

# activation functions


def sigm(x):
    return 8.0*tf.math.sigmoid(x)-4.0

# activation functions


def tanh(x):  # scaled tanh
    #    return 1.592537419722831*tf.tanh(x)
    return 16.0*tf.tanh(x)


def asinh(x):  # scaled asinh
    return 1.256734802399369*tf.log(x+tf.sqrt(x*x+1.0))


def actf(x):
    return tf.nn.softplus(x)-tf.log(2.0)

# def actf(x):
#     return tf.nn.relu(x)


# create neural network (and placeholders for feeding)
print("Creating neural network...\n")
X = tf.placeholder(tf.float32, shape=[None, n_inputs],  name="X")
y = tf.placeholder(tf.float32, shape=[None, n_outputs], name="y")

# NOTE: factor = 1 is only for the self-normalizing input functions!
yhat = MLP(X, n_hidden, n_outputs, activation_fn=actf,
           scope="neuralnetwork", factor=1.0)  # ordinary NN

# define loss function: here RMSE + regularization loss
with tf.name_scope("loss"):
    l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.reduce_mean(tf.squared_difference(
        tf.log(y+1.0), tf.log(y+tf.abs(tf.subtract(y, yhat))+1.0)))
    # loss = tf.reduce_mean(tf.squared_difference(y, yhat))

# define score function (performance measure, here: MAE)
# with tf.name_scope("loss2"):
#    loss = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(y,yhat),y)))

# define score function (performance measure, here: MAE)
with tf.name_scope("score"):
    score = tf.reduce_mean(tf.abs(tf.subtract(y, yhat)))

with tf.name_scope("rmsd"):
    rmsd = tf.reduce_mean(tf.squared_difference(y, yhat))

with tf.name_scope("evalu"):
    evalu = yhat

# define training method
with tf.name_scope("train"):
    training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# for logging stats
# mean absolute error
score_for_train = tf.constant(0.0)
score_for_valid = tf.constant(0.0)
score_for_best = tf.constant(0.0)
tf.summary.scalar("score-train", score_for_train)
tf.summary.scalar("score-valid", score_for_valid)
tf.summary.scalar("score-best",  score_for_best)

# Loss function
loss_for_train = tf.constant(0.0)
loss_for_valid = tf.constant(0.0)
loss_for_best = tf.constant(0.0)
tf.summary.scalar("loss-train", loss_for_train)
tf.summary.scalar("loss-valid", loss_for_valid)
tf.summary.scalar("loss-best", loss_for_best)

# root mean squared error
rmsd_for_train = tf.constant(0.0)
rmsd_for_valid = tf.constant(0.0)
rmsd_for_best = tf.constant(0.0)
tf.summary.scalar("rmsd-train", rmsd_for_train)
tf.summary.scalar("rmsd-valid", rmsd_for_valid)
tf.summary.scalar("rmsd-best", rmsd_for_best)

# merged summary op
summary_op = tf.summary.merge_all()
# create file writer for writing out summaries
file_writer = tf.summary.FileWriter(logdir=logdir,
                                    graph=tf.get_default_graph(),
                                    flush_secs=120)

# define saver nodes (max_to_keep=None lets the saver keep everything)
saver_best = tf.train.Saver(name="saver_best", max_to_keep=50)  # saves only the x best model
saver_step = tf.train.Saver(name="saver_step", max_to_keep=200)  # saves checkpoint every few steps

# counter that keeps going up for the best models
number_best = 0

# train the model
score_best = np.finfo(dtype=float).max  # initialize best score to huge value
loss_best = np.finfo(dtype=float).max  # initialize best loss to huge value
rmsd_best = np.finfo(dtype=float).max  # initialize best loss to huge value

all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# get the complete training and validation set
print("Starting evaluation...\n")
# ntot=data.num_test
# stscs=np.zeros([ntot,2])
with tf.Session() as sess:
    # initialize variables
    if retrain_model:
        saver_step.restore(sess, model_parameter_save)
        sess.run(zero_global_step_op)

    else:
        tf.global_variables_initializer().run()

#    nindx=100
#    with open("output_NN_directly.txt", "w") as txt_file:

#      pdf = matplotlib.backends.backend_pdf.PdfPages("output_NN_directly.pdf")
#      Egridout = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 3.6, 4.3, 5.2, 8.2, 11.2, 14.2, 17.6, 19.8])
#      vgridout = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0, 43.0, 47.0])
#      jgrid = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 115.0, 130.0, 145.0, 160.0, 175.0, 190.0, 205.0, 220.0, 230.0, 240.0])
#      num_outputs = len(Egridout) + len(vgridout) + len(jgrid)

#      for nindx in range(84):
#        print('data set: ' + str(nindx))
#        X_valid,y_valid = data.get_onetest_data(nindx)
#        xx=np.array([X_valid])
#        yy=np.array([y_valid])
#        yvalid = sess.run([evalu], feed_dict={X: xx})
#        fyy=np.array([yvalid])


#        fyy = fyy*data.get_stdv() + data.get_mval()
        # print (nindx,yy,fyy)
        # print(str(fyy.shape))
        # print(str(yy.shape))
        # #print(len(fyy))
        # print(yy)
        # print(fyy)
#        for i in range(61):
#          txt_file.write(str(i)+ ' ' + str(yy[0,i]) + ' ' +str(fyy[0,0,0,i]))
#          txt_file.write('\n')

        # Plotting:
#        Ep_pred = fyy[0,0,0,:len(Egridout)]
#        vp_pred = fyy[0,0,0,len(Egridout):len(vgridout)+len(Egridout)]
#        jp_pred = fyy[0,0,0,len(vgridout)+len(Egridout):num_outputs]
#
#        Ep_input = yy[0,:len(Egridout)]
#        vp_input = yy[0,len(Egridout):len(vgridout)+len(Egridout)]
#        jp_input = yy[0,len(vgridout)+len(Egridout):num_outputs]
#
#        plt.figure()
#        plt.plot(Egridout, Ep_input,'.b', label='QCT')
#        plt.plot(Egridout, Ep_pred,'.r', label = 'NN')
#        plt.legend()
#        pdf.savefig()
#        plt.close()
#
#        plt.figure()
#        plt.plot(vgridout, vp_input,'.b', label='QCT')
#        plt.plot(vgridout, vp_pred,'.r', label = 'NN')
#        plt.legend()
#        pdf.savefig()
#        plt.close()
#
#        plt.figure()
#        plt.plot(jgrid, jp_input,'.b', label='QCT')
#        plt.plot(jgrid, jp_pred,'.r', label = 'NN')
#        plt.legend()
#        pdf.savefig()
#        plt.close()
#
#      pdf.close()
#      plt.close('all')


#      stscs[nindx,0]=yy
#      stscs[nindx,1]=fyy
#
#    cmean=np.mean(stscs[:,0])
#    sstot=np.sum((stscs[:,0]-cmean)**2)
#    ssres=np.sum((stscs[:,0]-stscs[:,1])**2)
#    r2=1.0-ssres/sstot
#    rmse=np.sqrt(ssres/float(ntot))
#    print(cmean, ssres,r2,rmse)

    varlist = sess.run([all_variables])
    vardictionary = {}
    i = 0
    print(len(varlist[0]))
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        # print(var.name)
        vardictionary[var.name] = varlist[0][i]
        print(var.name, varlist[0][i].shape)
        i = i+1

    print(vardictionary.keys())
    h0W = vardictionary['neuralnetwork/hidden0/W:0']
    h0b = vardictionary['neuralnetwork/hidden0/b:0']
    h1W = vardictionary['neuralnetwork/hidden1/W:0']
    h1b = vardictionary['neuralnetwork/hidden1/b:0']
    h2W = vardictionary['neuralnetwork/hidden2/W:0']
    h2b = vardictionary['neuralnetwork/hidden2/b:0']
    h3W = vardictionary['neuralnetwork/hidden3/W:0']
    h3b = vardictionary['neuralnetwork/hidden3/b:0']
    h4W = vardictionary['neuralnetwork/hidden4/W:0']
    h4b = vardictionary['neuralnetwork/hidden4/b:0']
    h5W = vardictionary['neuralnetwork/hidden5/W:0']
    h5b = vardictionary['neuralnetwork/hidden5/b:0']
    h6W = vardictionary['neuralnetwork/hidden6/W:0']
    h6b = vardictionary['neuralnetwork/hidden6/b:0']
    h7W = vardictionary['neuralnetwork/hidden7/W:0']
    h7b = vardictionary['neuralnetwork/hidden7/b:0']
    h8W = vardictionary['neuralnetwork/hidden8/W:0']
    h8b = vardictionary['neuralnetwork/hidden8/b:0']
    h9W = vardictionary['neuralnetwork/hidden9/W:0']
    h9b = vardictionary['neuralnetwork/hidden9/b:0']
    h10W = vardictionary['neuralnetwork/hidden10/W:0']
    h10b = vardictionary['neuralnetwork/hidden10/b:0']
    h11W = vardictionary['neuralnetwork/hidden11/W:0']
    h11b = vardictionary['neuralnetwork/hidden11/b:0']
    h12W = vardictionary['neuralnetwork/hidden12/W:0']
    h12b = vardictionary['neuralnetwork/hidden12/b:0']
    h13W = vardictionary['neuralnetwork/hidden13/W:0']
    h13b = vardictionary['neuralnetwork/hidden13/b:0']
    h14W = vardictionary['neuralnetwork/hidden14/W:0']
    h14b = vardictionary['neuralnetwork/hidden14/b:0']
    h15W = vardictionary['neuralnetwork/hidden15/W:0']
    h15b = vardictionary['neuralnetwork/hidden15/b:0']
    h16W = vardictionary['neuralnetwork/hidden16/W:0']
    h16b = vardictionary['neuralnetwork/hidden16/b:0']
    h17W = vardictionary['neuralnetwork/hidden17/W:0']
    h17b = vardictionary['neuralnetwork/hidden17/b:0']
    # h18W = vardictionary['neuralnetwork/hidden18/W:0']
    # h18b = vardictionary['neuralnetwork/hidden18/b:0']
    # h19W = vardictionary['neuralnetwork/hidden19/W:0']
    # h19b = vardictionary['neuralnetwork/hidden19/b:0']
    # h20W = vardictionary['neuralnetwork/hidden20/W:0']
    # h20b = vardictionary['neuralnetwork/hidden20/b:0']
    outW = vardictionary['neuralnetwork/output/W:0']
    outb = vardictionary['neuralnetwork/output/b:0']
    np.savetxt('Coeff_h0W.dat', h0W, delimiter=',')
    np.savetxt('Coeff_h0b.dat', h0b, delimiter=',')
    np.savetxt('Coeff_h1W.dat', h1W, delimiter=',')
    np.savetxt('Coeff_h1b.dat', h1b, delimiter=',')
    np.savetxt('Coeff_h2W.dat', h2W, delimiter=',')
    np.savetxt('Coeff_h2b.dat', h2b, delimiter=',')
    np.savetxt('Coeff_h3W.dat', h3W, delimiter=',')
    np.savetxt('Coeff_h3b.dat', h3b, delimiter=',')
    np.savetxt('Coeff_h4W.dat', h4W, delimiter=',')
    np.savetxt('Coeff_h4b.dat', h4b, delimiter=',')
    np.savetxt('Coeff_h5W.dat', h5W, delimiter=',')
    np.savetxt('Coeff_h5b.dat', h5b, delimiter=',')
    np.savetxt('Coeff_h6W.dat', h6W, delimiter=',')
    np.savetxt('Coeff_h6b.dat', h6b, delimiter=',')
    np.savetxt('Coeff_h7W.dat', h7W, delimiter=',')
    np.savetxt('Coeff_h7b.dat', h7b, delimiter=',')
    np.savetxt('Coeff_h8W.dat', h8W, delimiter=',')
    np.savetxt('Coeff_h8b.dat', h8b, delimiter=',')
    np.savetxt('Coeff_h9W.dat', h9W, delimiter=',')
    np.savetxt('Coeff_h9b.dat', h9b, delimiter=',')
    np.savetxt('Coeff_h10W.dat', h10W, delimiter=',')
    np.savetxt('Coeff_h10b.dat', h10b, delimiter=',')
    np.savetxt('Coeff_h11W.dat', h11W, delimiter=',')
    np.savetxt('Coeff_h11b.dat', h11b, delimiter=',')
    np.savetxt('Coeff_h12W.dat', h12W, delimiter=',')
    np.savetxt('Coeff_h12b.dat', h12b, delimiter=',')
    np.savetxt('Coeff_h13W.dat', h13W, delimiter=',')
    np.savetxt('Coeff_h13b.dat', h13b, delimiter=',')
    np.savetxt('Coeff_h14W.dat', h14W, delimiter=',')
    np.savetxt('Coeff_h14b.dat', h14b, delimiter=',')
    np.savetxt('Coeff_h15W.dat', h15W, delimiter=',')
    np.savetxt('Coeff_h15b.dat', h15b, delimiter=',')
    np.savetxt('Coeff_h16W.dat', h16W, delimiter=',')
    np.savetxt('Coeff_h16b.dat', h16b, delimiter=',')
    np.savetxt('Coeff_h17W.dat', h17W, delimiter=',')
    np.savetxt('Coeff_h17b.dat', h17b, delimiter=',')
    # np.savetxt('Coeff_h18W.dat', h18W, delimiter=',')
    # np.savetxt('Coeff_h18b.dat', h18b, delimiter=',')
    # np.savetxt('Coeff_h19W.dat', h19W, delimiter=',')
    # np.savetxt('Coeff_h19b.dat', h19b, delimiter=',')
    # np.savetxt('Coeff_h20W.dat', h20W, delimiter=',')
    # np.savetxt('Coeff_h20b.dat', h20b, delimiter=',')
    np.savetxt('Coeff_outW.dat', outW, delimiter=',')
    np.savetxt('Coeff_outb.dat', outb, delimiter=',')

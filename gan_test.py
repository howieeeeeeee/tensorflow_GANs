# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1./tf.sqrt(in_dim/2.)
    return tf.random_normal(shape=size,stddev=xavier_stddev)

#X = tf.placeholder(tf.float32,shape = [None,784])
#D_W1 = tf.Variable(xavier_init([784, 128]),name='D_W1')
#D_b1 = tf.Variable(tf.zeros(shape=[128]),name='D_b1')
#
#D_W2 = tf.Variable(xavier_init([128, 1]),name='D_W2')
#D_b2 = tf.Variable(tf.zeros(shape=[1]),name='D_b2')

Z = tf.placeholder(tf.float32,shape = [None,100])
G_W1 = tf.Variable(xavier_init([100, 128]),name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]),name='G_W1')

G_W2 = tf.Variable(xavier_init([128, 784]),name='G_W1')
G_b2 = tf.Variable(tf.zeros(shape=[784]),name='G_W1')

def sample_Z(m,n):
    return np.random.uniform(-1.,1.,size=[m, n])

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z,G_W1)+G_b1)
    G_log_prob = tf.matmul(G_h1,G_W2)+G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    
    return G_prob

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

G_sample = generator(Z)
#D_real, D_logit_real = discriminator(X)
#D_fake, D_logit_fake = discriminator(G_sample)
Z_dim = 100

saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('out/')
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            saver.restore(sess,os.path.join(ckpt.model_checkpoint_path))
            
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %(ckpt.model_checkpoint_path,global_step))
    else:
        print('No checkpoint file found')
#        return
    
    samples = sess.run(G_sample, feed_dict = {Z: sample_Z(16,Z_dim)})
    fig = plot(samples)
    plt.savefig('out/test.png', bbox_inches='tight')
    plt.close(fig)
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains ready to train architectures using standard training and stabilized ones.

Author: Alexandre Péré

"""

from network import BaseNetwork
import tensorflow as tf

class MnistNetwork(BaseNetwork):
    """
    A Simple Mnist classifier that follows the architectures proposed in the deep mnist tutorial at:
    ttps://www.tensorflow.org/tutorials/mnist/pros/
    """

    def _construct_arch(self):

        with self._tf_graph.as_default():
            
            # Define Network # =========================================================================================
            # Input #
            with tf.name_scope('Input') as scope:
                self._net_input = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='Input')
                x_image = tf.reshape(self._net_input, [-1, 28, 28, 1])
            
            # Conv
            W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name='W1')
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name = 'b1')
            with tf.name_scope('Conv1') as scope:
                h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

            # Max pool
            with tf.name_scope('MaxPool1') as scope:
                h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Conv
            W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='W2')
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b2')
            with tf.name_scope('Conv2') as scope:
                h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

            # Max pool
            with tf.name_scope('MaxPool1') as scope:
                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Dense                
            W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1), name='W3')
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b3')
            with tf.name_scope('Dense1') as scope:
                h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
                
            W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='W4')
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), name='b4')
            with tf.name_scope('Dense2') as scope:
                y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

            # Output # 
            with tf.name_scope('Output') as scope:
                self._net_output = tf.nn.softmax(y_conv, name='Output')
                self._net_label = tf.placeholder(tf.float32, shape=[None, 10], name='Label')

            # Define Loss # ============================================================================================
            with tf.name_scope('Loss') as scope:
                cross_entropy = tf.reduce_mean(-tf.reduce_sum(self._net_label * tf.log(self._net_output), 
                                               reduction_indices=[1]), name='CrossEntropy')
                weights = tf.trainable_variables()
                weights_decay = tf.add_n([tf.nn.l2_loss(v) for v in weights]) * 0.001
                self._net_loss = tf.add(cross_entropy, weights_decay, name='NetLoss')
                tf.summary.scalar('Loss', self._net_loss)
                
            # Define Accuracy # ========================================================================================
            with tf.name_scope('Accuracy') as scope:
                correct_prediction = tf.equal(tf.argmax(self._net_output, 1), tf.argmax(self._net_label, 1))
                self._net_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('Accuracy', self._net_accuracy)

            # Define Optimizer # =======================================================================================
            self._net_optimize = tf.train.AdamOptimizer(1e-4).minimize(self._net_loss)
            
            # Define Train Dict # ======================================================================================
            self._net_train_dict = dict()
            self._net_test_dict = dict()
            
            # Define Summaries # =======================================================================================
            self._net_summaries = tf.summary.merge_all()


class CifarNet(BaseNetwork):
    """
    Cifar-10  architecture, based on implementation proposed in CIFAR 10 tensorflow tutorial available at:
    https://www.tensorflow.org/versions/r0.10/tutorials/deep_cnn/
    """

    def _construct_arch(self):
        with self._tf_graph.as_default():
            
            # Define Parameters # ======================================================================================
            dropout_prob = tf.placeholder(tf.float32, shape=[], name='DropoutProb')
            beta = tf.placeholder(tf.float32, shape=[], name='Beta')
            learning_rate = tf.placeholder(tf.float32, shape=[], name='LearningRate')
            tf.summary.scalar('LearningRate', learning_rate)
            
            # Define Network # =========================================================================================
            # Input #
            with tf.name_scope('Input') as scope:
                self._net_input = tf.placeholder(tf.float32, shape=[None, 32 * 32 * 3], name='Input')
                x_image = tf.reshape(self._net_input, [-1, 32, 32, 3])

            # Conv1 # 
            W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=0.05), dtype=tf.float32, name='W1')
            b1 = tf.Variable(tf.constant(0.1, shape=[64]), dtype=tf.float32, name='b1')
            with tf.name_scope('Conv1') as scope:
                a1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')
                a1 = tf.add(a1, b1)
                a1 = tf.nn.relu(a1)
            
            # MaxPool1 #
            with tf.name_scope('MaxPool1') as scope:
                a1 = tf.nn.max_pool(a1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            # Conv2 # 
            W2 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.05), dtype=tf.float32, name='W2')
            b2 = tf.Variable(tf.constant(0.1, shape=[64]), dtype=tf.float32, name='b2')
            with tf.name_scope('Conv2') as scope:
                a2 = tf.nn.conv2d(a1, W2, strides=[1, 1, 1, 1], padding='SAME')
                a2 = tf.add(a2, b2)
                a2 = tf.nn.relu(a2)
           
            # MaxPool2 #
            with tf.name_scope('MaxPool2') as scope:
                a2 = tf.nn.max_pool(a2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='LastNotDense')

            # Full3 # 
            W3 = tf.Variable(tf.truncated_normal([7*7*64, 384], stddev=0.04), dtype=tf.float32, name='W3')
            b3 = tf.Variable(tf.constant(0.1, shape=[384]), dtype=tf.float32, name='b3')
            with tf.name_scope('Dense3') as scope:
                a3 = tf.reshape(a2, [-1, 7*7*64], name='FirstDense')
                a3 = tf.add(tf.matmul(a3, W3), b3)
                a3 = tf.nn.relu(a3)
                a3 = tf.nn.dropout(a3,dropout_prob)

            # Full4 # 
            W4 = tf.Variable(tf.truncated_normal([384, 192], stddev=0.04), dtype=tf.float32, name='W4')
            b4 = tf.Variable(tf.constant(0.1, shape=[192]), dtype=tf.float32, name='b4')
            with tf.name_scope('Dense4') as scope:
                a4 = tf.add(tf.matmul(a3, W4), b4)
                a4 = tf.nn.relu(a4)
                a4 = tf.nn.dropout(a4,dropout_prob)
            
            # Full5 # 
            W5 = tf.Variable(tf.truncated_normal([192, 10], stddev=1/192), dtype=tf.float32, name='W5')
            b5 = tf.Variable(tf.constant(0.1, shape=[10]), dtype=tf.float32, name='b5')
            with tf.name_scope('Dense5') as scope:
                a5 = tf.add(tf.matmul(a4, W5), b5, name='BeforeSoftMax')
            
            # Output # 
            with tf.name_scope('Output') as scope:
                self._net_output = tf.nn.softmax(a5, name='Output')
                self._net_label = tf.placeholder(tf.float32, shape=[None, 10])

            # Define Loss # ============================================================================================
            with tf.name_scope('Loss') as scope:
                cross_entropy = tf.reduce_mean(-tf.reduce_sum(self._net_label * tf.log(self._net_output), 
                                               reduction_indices=[1]), name='CrossEntropy')
                weights = tf.trainable_variables()
                weights_decay = tf.scalar_mul(beta,tf.add_n([tf.nn.l2_loss(v) for v in weights]))
                self._net_loss = tf.add(cross_entropy, weights_decay, name='Loss')
                tf.summary.scalar('Loss', self._net_loss)
            
            # Define Pert Tools # ======================================================================================
            with tf.name_scope('Pert') as scope:
                unitary_cross_entropy = -tf.mul(self._net_label, tf.log(self._net_output), name='UnitCrossEntropy')
                unit_im_grad = tf.gradients(unitary_cross_entropy, self._net_input, name='UnitImGrad')
            
            # Define Accuracy # ========================================================================================
            with tf.name_scope('Accuracy') as scope:
                correct_prediction = tf.equal(tf.argmax(self._net_output, 1), tf.argmax(self._net_label, 1))
                self._net_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
                tf.summary.scalar('Accuracy', self._net_accuracy)
                
            # Define Optimizer # =======================================================================================
            #self._net_optimize = tf.train.AdadeltaOptimizer(1e-3).minimize(self._net_loss)
            self._net_optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(self._net_loss)
            
            # Define Dicts # ===========================================================================================
            self._net_train_dict = {dropout_prob:0.5, beta:0.004, learning_rate:1e-5}
            self._net_test_dict = {dropout_prob:1, beta:0.004, learning_rate:1e-5}
            
            # Define Summaries # =======================================================================================
            self._net_summaries = tf.summary.merge_all()

            
class StabilizedCifarNet(BaseNetwork):
    """
    Cifar-10  architecture, based on implementation proposed in CIFAR 10 tensorflow tutorial available at:
    https://www.tensorflow.org/versions/r0.10/tutorials/deep_cnn/
    Implementing the Stability training as proposed in https://arxiv.org/pdf/1604.04326v1.pdf
    """

    def _construct_arch(self):
        
        with self._tf_graph.as_default():
            
            # Define Parameters # ======================================================================================
            stdv = tf.placeholder(tf.float32, shape=[], name='NoiseStdv')
            dropout_prob = tf.placeholder(tf.float32, shape=[], name='DropoutProb')
            alpha = tf.placeholder(tf.float32, shape=[], name='Alpha')
            beta = tf.placeholder(tf.float32, shape=[], name='Beta')
            learning_rate = tf.placeholder(tf.float32, shape=[], name='LearningRate')
            tf.summary.scalar('LearningRate', learning_rate)
                        
            # Define Network # =========================================================================================
            # Input # 
            with tf.name_scope('Input') as scope:
                self._net_input = tf.placeholder(tf.float32, shape=[None, 32 * 32 * 3], name='Input')
                x_image = tf.reshape(self._net_input, [-1, 32, 32, 3])            
                test_unperturbed = tf.identity(x_image, name='UnPerturbed')
                prtrb = tf.random_normal(tf.shape(x_image), stddev=stdv, dtype=tf.float32)
                test_pert = tf.identity(prtrb, name='Pert')
                x_prtrb = tf.stop_gradient(tf.add(x_image, prtrb))
                test_perturbed = tf.identity(x_prtrb, name='Perturbed')
            
            # Conv1 # 
            W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=0.05), dtype=tf.float32, name='W1')
            b1 = tf.Variable(tf.constant(0.1, shape=[64]), dtype=tf.float32, name='b1')
            with tf.name_scope('UnPertConv1') as scope:
                a1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')
                a1 = tf.add(a1, b1)
                a1 = tf.nn.relu(a1)
            with tf.name_scope('PertConv1') as scope:
                a1_p = tf.stop_gradient(tf.nn.conv2d(x_prtrb, W1, strides=[1, 1, 1, 1], padding='SAME'))
                a1_p = tf.stop_gradient(tf.add(a1_p, b1))
                a1_p = tf.stop_gradient(tf.nn.relu(a1_p))
                
            # MaxPool1 #
            with tf.name_scope('UnPertMaxPool1') as scope:
                a1 = tf.nn.max_pool(a1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            with tf.name_scope('PertMaxPool1') as scope:
                a1_p = tf.stop_gradient(tf.nn.max_pool(a1_p, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID'))
            
            # Conv2 # 
            W2 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.05), dtype=tf.float32, name='W2')
            b2 = tf.Variable(tf.constant(0.1, shape=[64]), dtype=tf.float32, name='b2')
            with tf.name_scope('UnPertConv2') as scope:
                a2 = tf.nn.conv2d(a1, W2, strides=[1, 1, 1, 1], padding='SAME')
                a2 = tf.add(a2, b2)
                a2 = tf.nn.relu(a2)
            with tf.name_scope('PertConv2') as scope:
                a2_p = tf.stop_gradient(tf.nn.conv2d(a1_p, W2, strides=[1, 1, 1, 1], padding='SAME'))
                a2_p = tf.stop_gradient(tf.add(a2_p, b2))
                a2_p = tf.stop_gradient(tf.nn.relu(a2_p))
                
            # MaxPool2 # 
            with tf.name_scope('UnPertMaxPool2') as scope:
                a2 = tf.nn.max_pool(a2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            with tf.name_scope('PertMaxPool2') as scope:
                a2_p = tf.stop_gradient(tf.nn.max_pool(a2_p, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID'))
            
            # Full3 # 
            W3 = tf.Variable(tf.truncated_normal([7*7*64, 384], stddev=0.04), dtype=tf.float32, name='W3')
            b3 = tf.Variable(tf.constant(0.1, shape=[384]), dtype=tf.float32, name='b3')
            with tf.name_scope('UnPertDense3') as scope:
                a3 = tf.reshape(a2, [-1, 7*7*64])
                a3 = tf.add(tf.matmul(a3, W3), b3)
                a3 = tf.nn.relu(a3)
                a3 = tf.nn.dropout(a3, dropout_prob)
            with tf.name_scope('PertDense3') as scope:
                a3_p = tf.stop_gradient(tf.reshape(a2_p, [-1, 7*7*64]))
                a3_p = tf.stop_gradient(tf.add(tf.matmul(a3_p, W3), b3))
                a3_p = tf.stop_gradient(tf.nn.relu(a3_p))
                a3_p = tf.stop_gradient(tf.nn.dropout(a3_p, dropout_prob))
            
            # Full4 # 
            W4 = tf.Variable(tf.truncated_normal([384, 192], stddev=0.04), dtype=tf.float32, name='W4')
            b4 = tf.Variable(tf.constant(0.1, shape=[192]), dtype=tf.float32, name='b4')
            with tf.name_scope('UnPertDense4') as scope:
                a4 = tf.add(tf.matmul(a3, W4), b4)
                a4 = tf.nn.relu(a4)
                a4 = tf.nn.dropout(a4, dropout_prob)
            with tf.name_scope('PertDense4') as scope:
                a4_p = tf.stop_gradient(tf.add(tf.matmul(a3_p, W4), b4))
                a4_p = tf.stop_gradient(tf.nn.relu(a4_p))
                a4_p = tf.stop_gradient(tf.nn.dropout(a4_p, dropout_prob))

            # Full5 # 
            W5 = tf.Variable(tf.truncated_normal([192, 10], stddev=1/192), dtype=tf.float32, name='W5')
            b5 = tf.Variable(tf.constant(0.1, shape=[10]), dtype=tf.float32, name='b5')
            with tf.name_scope('UnPertDense5') as scope:
                a5 = tf.add(tf.matmul(a4, W5), b5, name='UnPerturbed')
            with tf.name_scope('PertDense5') as scope:
                a5_p = tf.stop_gradient(tf.add(tf.matmul(a4_p, W5), b5, name='Perturbed'))
            
            # Output # 
            with tf.name_scope('Output') as scope:
                output_image = tf.nn.softmax(a5)
                output_prtrb = tf.stop_gradient(tf.nn.softmax(a5_p))
                self._net_label = tf.placeholder(tf.float32, shape=[None, 10])
                self._net_output = output_image

            # Define Loss # ============================================================================================
            with tf.name_scope('Loss') as scope:
                cross_entropy = tf.reduce_mean(-tf.reduce_sum(self._net_label * tf.log(self._net_output), 
                                               reduction_indices=[1]), name='CrossEntropy')
                stability_term = tf.stop_gradient(tf.reduce_mean(-tf.reduce_sum(output_image * tf.log(output_prtrb),
                                                                 reduction_indices=[1])), name='Stability')
                loss = tf.add(cross_entropy, tf.scalar_mul(alpha, stability_term))
                weights = tf.trainable_variables()
                weights_decay = tf.scalar_mul(beta,tf.add_n([tf.nn.l2_loss(v) for v in weights]))
                self._net_loss = tf.add(loss, weights_decay, name='Loss')
                tf.summary.scalar('Loss', self._net_loss)
                tf.summary.scalar('Stability', stability_term)
                
            # Define Accuracy # ========================================================================================
            with tf.name_scope('Accuracy') as scope:
                correct_prediction = tf.equal(tf.argmax(self._net_output, 1), tf.argmax(self._net_label, 1))
                self._net_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
                tf.summary.scalar('Accuracy', self._net_accuracy)

            # Define Optimizer # =======================================================================================
            #self._net_optimize = tf.train.AdadeltaOptimizer(1e-4).minimize(self._net_loss)
            self._net_optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(self._net_loss)

            # Define Train Dict # ======================================================================================
            self._net_train_dict = {dropout_prob:0.5, stdv:0.05, alpha: 0.01, beta:0.004, learning_rate:0.1}
            self._net_test_dict = {dropout_prob:1., stdv:0.05, alpha:0.01, beta: 0.004, learning_rate:0.1}
            
            # Define Summaries # =======================================================================================
            self._net_summaries = tf.summary.merge_all()

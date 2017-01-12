#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains base implementation of a NN classifier trained using supervised learning.

Author: Alexandre Péré

"""

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy
import time
import os
import pickle
import scipy.io

class BaseNetwork(object):
    """
    This defines the basic network structure we use.
    """

    def __init__(self, path_to_logs=os.getcwd()):
        """
        The initializer of the BasicNetwork object.
        
        Attributes:
            + self._tf_graph: the tf graph containing the network structure
            + self._tf_session: the tf session used to compute operations relative to the object
            + self._tf_fw: the tf file writer to display the graph in tensorboard
            + self._net_loss: the tf expression of loss attached to the graph
            + self._net_optimize: the tf optimization method
            + self._net_input: the tf input placeholder
            + self._net_label: the tf labels placeholder
            + self._net_output: the tf net outuput
            + self._net_accuracy: the tf accuracy method 
            + self._net_train_dict: the dictionnary added for training
            + self._net_test_dict: the dictionnary added for testing
            + self._net_summaries: the tensoroard merged summaries
            + self._net_history: a list containing training records (arrays [time, train accuracy, test accuracy])
            + self._logs_path: the path to tensorboard logs files
        """
        
        # Initialize super 
        object.__init__(self)
           
        # We initialize the variables of the object
        self._tf_graph = tf.Graph()
        self._tf_session =None
        self._tf_fw = None
        self._net_loss = None
        self._net_optimize = None
        self._net_input = None
        self._net_label = None
        self._net_output = None
        self._net_accuracy = None
        self._net_train_dict = dict()
        self._net_test_dict = dict()
        self._net_summaries = None
        self._net_history = list()
        self._net_summaries_history = list()
        self._net_summary_parser = summary_pb2.Summary()
        self._logs_path = path_to_logs
        
        # We construct and initialize everything
        self._construct_arch()
        self._initialize_fw()
        self._initialize_session()
        self._initialize_weights()     

    def train(self, X_train, y_train, 
              X_test, y_test, 
              iterations=0, 
              criterion=0,
              train_batch_size=100, 
              test_batch_size=100,
              callback=None):
        """
        The public training method. A network can be trained for a specified number of iterations using the _iterations_
        parameter, or with a stopping criterion over the training accuracy using the _criterion_ argument.
        
        Parameters:
            + X_train: a numpy array containing training input data
            + y_train: a numpy array containing training output classes
            + X_test: a numpy array containing testing input data
            + y_test: a numpy array containing testing output classes
            + iterations: number of iterations to perform
            + criterion: stopping criterion over training accuracy
            + train_batch_size: the batch size for training data
            + test_batch_size: the batch size for testing data
            + callback: a method to be called before each printing iteration
        """
        
        # We check that the number of iterations set is greater than 100 if iterations is used
        if (criterion == 0 and iterations<100):
            raise Warning("Number of iterations must be superior to 100")
        
        # We initialize history if the network is fresh
        if len(self._net_history)==0:
            self._net_history.append([0., 0., 0.])
            start_time = 0.
        else:
            start_time = max(numpy.asarray(self._net_history)[:,0])
        start_tick = time.time()

        # Training with iterations
        if iterations != 0 and criterion == 0:
            for iter in range(iterations):
                # We get the random indexes to use in the batch
                train_idx = numpy.random.permutation(X_train.shape[0])
                train_idx = train_idx[0:train_batch_size]
                # We execute the gradient descent step
                input_dict = {self._net_input: X_train[train_idx], self._net_label: y_train[train_idx]}
                input_dict.update(self._net_train_dict)
                self._net_optimize.run(feed_dict=input_dict, session=self._tf_session)
                # If the iteration is a multiple of 100, we do things
                if (iter % 100 == 0) and (iter > 0):
                    # We compute the train accuracy over the batch
                    input_dict = {self._net_input: X_train[train_idx], self._net_label: y_train[train_idx]}
                    input_dict.update(self._net_test_dict)
                    train_accuracy = self._net_accuracy.eval(feed_dict=input_dict, session=self._tf_session)
                    # We compute the test accuracy over the batch
                    test_idx = numpy.random.permutation(X_test.shape[0])
                    test_idx = test_idx[0:test_batch_size]
                    input_dict = {self._net_input: X_test[test_idx], self._net_label: y_test[test_idx]}
                    input_dict.update(self._net_test_dict)
                    test_accuracy = self._net_accuracy.eval(feed_dict=input_dict, session=self._tf_session)
                    # We update tensorboard summaries
                    summary = self._net_summaries.eval(feed_dict=input_dict,session=self._tf_session)
                    self._net_summary_parser.ParseFromString(summary)
                    self._net_summaries_history.append({str(val.tag):val.simple_value for val in self._net_summary_parser.value})
                    self._tf_fw.add_summary(summary,iter)
                    self._tf_fw.flush()
                    # We write the record to the history
                    self._net_history.append([(time.time() - start_tick) + start_time, train_accuracy, test_accuracy])
                    # We execute the callback if it exists
                    if callback is not None: callback(self)
                        
        # Training with criterion
        elif criterion != 0 and iterations == 0:
            iter = 0
            train_accuracy = 0
            while train_accuracy < criterion:
                iter += 1
                # We get the random indexes to use in the batch
                train_idx = numpy.random.permutation(X_train.shape[0])
                train_idx = train_idx[0:train_batch_size]
                # We execute the gradient descent step
                input_dict = {self._net_input: X_train[train_idx], self._net_label: y_train[train_idx]}
                input_dict.update(self._net_train_dict)
                self._net_optimize.run(feed_dict=input_dict, session=self._tf_session)
                # If the iteration is a multiple of 100, we do things
                if (iter % 100 == 0) and (iter > 0): 
                    # We compute the train accuracy over the batch
                    input_dict = {self._net_input: X_train[train_idx], self._net_label: y_train[train_idx]}
                    input_dict.update(self._net_test_dict)
                    train_accuracy = self._net_accuracy.eval(feed_dict=input_dict, session=self._tf_session)
                    # We compute the test accuracy over the batch
                    test_idx = numpy.random.permutation(X_test.shape[0])
                    test_idx = test_idx[0:test_batch_size]
                    input_dict = {self._net_input: X_test[test_idx], self._net_label: y_test[test_idx]}
                    input_dict.update(self._net_test_dict)
                    test_accuracy = self._net_accuracy.eval(feed_dict=input_dict, session=self._tf_session)
                    # We update tensorboard summaries
                    summary = self._net_summaries.eval(feed_dict=input_dict,session=self._tf_session)
                    self._net_summary_parser.ParseFromString(summary)
                    self._net_summaries_history.append({str(val.tag):val.simple_value for val in self._net_summary_parser.value})
                    self._tf_fw.add_summary(summary,iter)
                    self._tf_fw.flush()
                    self._net_summaries_history.append(summary)
                    # We write the record to the history
                    self._net_history.append([(time.time() - start_tick) + start_time, train_accuracy, test_accuracy])
                    # We execute the callback if it exists
                    if callback is not None: callback(self)                    
               
        # Ambiguous arguments
        else:
            raise Warning("Ambiguous Arguments. You can either set a number of iterations or a stopping criterion.")

    def test(self, X_test, y_test, top=1):
        """
        The public testing method.
        
        Parameters:
            + X_test: a numpy array containing testing data
            + y_test: a numpy array containing testing classes
            + top: compute the top-n accuracy
        
        Returns:
            + accuracy over the test set
        """
        
        # We initialize the test acc var
        test_acc = 0.
        nb_batch = X_test.shape[0]
        
        # We loop through the samples to compute accuracy sum
        for itr in range(0, nb_batch):
            input_dict = {self._net_input: X_test[itr:itr + 1], self._net_label: y_test[itr:itr + 1]}
            input_dict.update(self._net_test_dict)
            outpt = self._net_output.eval(feed_dict=input_dict, session=self._tf_session)
            true_label = numpy.argsort(y_test[itr])[-1]
            top_n_out = numpy.argsort(outpt[0])[-top:]
            if true_label in top_n_out:
                test_acc +=1
            
        # We divide by the number of samples to get the accuracy over the test set
        test_acc /= nb_batch
        
        return test_acc

    def evaluate_output(self, X):
        """
        The public output evaluation method. 
        
        Parameters:
            + X: a numpy array containing input data
            
        Returns:
            + a numpy array containing the evaluations
        """
        
        # We instantiate the output array
        output_shape = [dim.value for dim in self._net_output.get_shape()]
        out_arr = list()
        
        # We loop through the samples to evaluate the network value
        for iter in range(0, X.shape[0]):
            input_dict = {self._net_input: X[iter:iter+1]}
            input_dict.update(self._net_test_dict)
            tensor_to_eval = self._net_output.eval(feed_dict=input_dict, session=self._tf_session)
            out_arr.append(tensor_to_eval.eval(feed_dict=input_dict, session=self._tf_session))
            
        return numpy.asarray(out_arr)
    
    def evaluate_tensor(self, name, initial_dict='train', update_dict=None):
        """
        The public tensor evaluation method. You can eval any tensor given an input dict. The initial dict 
        is basically fixed to be the train dict.
        
        Parameters:
            + name: the name of the tensor to evaluate
            + initial_dict: 'train' to use train_dict as initial dict, 'test' to use test dict as initial dict
            + update_dict: some input dict of your own to update the initial_dict
            
        Returns:
            + a numpy array containing the evaluations
        """
        
        # We retrieve the tensor by name
        tensor_to_eval = self.get_tensor(name)
        
        # We set the input dict
        if initial_dict=='train':
            input_dict = self._net_train_dict
        elif initial_dict=='test':
            input_dict = self._net_test_dict
        if update_dict is not None:    
            input_dict.update(update_dict)
            
        # We evaluate the tensor
        out_arr = tensor_to_eval.eval(feed_dict=input_dict, session=self._tf_session)
            
        return out_arr
    
    def update_feed_dict_value(self, key, value, which):
        """
        The public feed dict update method. Used to update the learning rate during training.
        
        Parameters:
            + key: the dict key to update
            + value: the dict new value
            + which: if 'test' change test dict, if 'train' change train dict, if 'both' change both
        """
        
        if which=="test":
            self._net_test_dict[key] = value;
        elif which=="train":
            self._net_train_dict[key] = value;
        elif which=='both':
            self._net_train_dict[key] = value;
            self._net_test_dict[key] = value;

    def save(self, path):
        """
        The public saving method, which allows to save a trained network. Tensorflow do not save on a single file, hence
        path doesn't need to have an extension.
        
        Parameters:
            + path: Path to files like '/tmp/model'
        """
        
        # We save tensorflow objects
        with self._tf_graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._tf_session, os.path.abspath(path))
            
        # We save history list
        with open(path + '.hst', 'wb') as file:
            pickle.dump(self._net_history, file)
            
    def save_mat_weights(self, path=os.getcwd()):
        """
        Public saving method that exports weights as matlab .mat files, with 1 file for each tensor, and with weight name
        as filename.
        
        Parameters:
            + path: Path to folder that will contain .mat files
        """
       
        with self._tf_graph.as_default():
            variables = tf.trainable_variables()
            for var in variables:
                vararr = var.eval(session=self._tf_session)
                varname = var.name[0:-2]
                filename = ('%s/%s.mat'%(path,varname))
                scipy.io.savemat(filename,{varname:vararr})              

    def load(self, path):
        """
        The public loading method, which allows to restore a trained network. Tensorflow do not save on a single file,
        hence path doesn't need to have an extension.
        
        Parameters:
             + path: Path to files like '/tmp/model'
        """
        
        # We load the tensorflow objects
        with self._tf_graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._tf_session, os.path.abspath(path))
            self._tf_fw.add_graph(self._tf_graph)
            
        # We load history list
        with open (path + '.hst', 'rb') as file:
            self._net_history = pickle.load(file)
    
    def get_history(self):
        """
        This public method allows to retrieve the whole history of the network. 
        
        Returns:
            + a numpy array of size [nb,3] with nb the number of iterations divided by 100. Each record contains 
              cumuled duration, training accuracy, and testing accuracy
        """
        
        return numpy.asarray(self._net_history)
    
    def get_summaries(self, name=None):
        """
        This public method allows to retrieve the recent summaries of the network.
        
        Parameters:
            + name: if the name of the summary you want to retrieve, if not given, everything is returned
                
        Returns:
            + a list containing merged summaries if no name is provided, and an array containing the data otherwise.
        """
        if name is None:
            return self._net_summaries_history
        else:
            length = len(self._net_summaries_history)
            array = numpy.zeros([length,1])
            for i in range(0,length):
                array[i] = self._net_summaries_history[i][name]
            return array
        
    def get_tensor(self, name):
        """
        This public method allows to catch a tensor by its name in the architecture.
        
        Parameters:
            + name: the name of the tensor ex: 'Conv1/W1:0'
        
        Returns:
            + The tensor
        """
        
        return self._tf_graph.get_tensor_by_name(name)
        
    def _initialize_weights(self):
        """
        The private weights initialization method.
        """
        
        with self._tf_graph.as_default():
            self._tf_session.run(tf.global_variables_initializer())
            
    def _initialize_fw(self):
        """
        The private filewriter initialization method.
        """
        
        self._tf_fw = tf.summary.FileWriter(self._logs_path, graph=self._tf_graph)
        tf.train.SummaryWriterCache.clear()
        
    def _initialize_session(self):
        """
        The private session initialization method.
        """
   
        self._tf_session = tf.Session(graph=self._tf_graph)
        

    def _construct_arch(self):
        """
        The private architecture construction method. Should be reimplemented, and define the computations of the
        following attributes:
            + self._net_input: the input tf placeholder
            + self._net_output: the output layer of the network
            + self._net_label: the labels tf placeholder
            + self._net_loss: the loss used to train the network (containing weights decay)
            + self._net_optimize: the optimization method to use for training
            + self._net_accuracy: the accuracy measure used to monitor the network performance
        """

        raise NotImplementedError("Virtual Method Was called")

        # with self._tf_graph.as_default():
        #
        #     # Define Network # =======================================================================================
        #
        #     # Input # ------------------------------------------------------------------------------------------------
        #     self._net_input = tf.placeholder(tf.float32, shape=[None, "Put Here Input Dim"], name='input')
        #
        #     # Output # -----------------------------------------------------------------------------------------------
        #     self._net_output = tf.nn.softmax("Put Here output layer", name='output')
        #     self._net_label = tf.placeholder(tf.float32, shape=[None, "Put Here Output Dim"], name='label')
        #
        #     # Define Loss # ==========================================================================================
        #     self._net_loss = tf.add(cross_entropy, weights_decay)
        #
        #     # Define Optimizer # =====================================================================================
        #     self._net_optimize = tf.train.AdamOptimizer(1e-4).minimize(self._net_loss)
        #
        #     # Define Accuracy # ======================================================================================
        #     correct_prediction = tf.equal(tf.argmax(self._net_output, 1), tf.argmax(self._net_label, 1))
        #     self._net_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


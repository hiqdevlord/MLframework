'''
Created on Feb 18, 2013

@author: Ash Booth
'''

import theano
import theano.tensor as T
import numpy

class DeepLearn(object):
    '''
    Class that deals with deep learning stuff.
    
    Code adapted from:
    http://deeplearning.net/tutorial/
    '''


    def __init__(self):
        '''
        NLL is a symbolic variable ; to get the actual value of NLL, this symbolic
        expression has to be compiled into a Theano function.
        
        N.B. T.arange(y.shape[0]) is a vector of integers [0,1,2,...,len(y)].
        Indexing a matrix M by the two vectors [0,1,...,K], [a,b,...,k] returns the
        elements M[0,a], M[1,b], ..., M[K,k] as a vector.  Here, we use this
        syntax to retrieve the log-probability of the correct labels, y.
        '''
        
    def shared_dataset(self, data_xy):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        it's needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')
        
        
    def __miniB_stoch_grad_dec(self, x_batch, y_batch, params, learning_rate, lambda_1, lambda_2):
        '''
        Minibatch Stochastic Gradient Descent
        assume loss is a symbolic description of the loss function given
        the symbolic variables params (shared variable), x_batch, y_batch;
        '''
        p = T.dscalar('p')
        y = T.dscalar('y')
        NLL = -T.sum(T.log(p)[T.arange(y.shape[0]), y])
        # symbolic Theano variable that represents the L1 regularization term
        L1  = T.sum(abs(params))
        # symbolic Theano variable that represents the squared L2 term
        L2_sqr = T.sum(params ** 2)
        # the loss
        loss = NLL + lambda_1 * L1 + lambda_2 * T.sqrt(L2_sqr)
        
        d_loss_wrt_params = T.grad(loss, params)
        
        # compile the MSGD step into a theano function
        updates = [(params, params - learning_rate * d_loss_wrt_params)]
        MSGD = theano.function([x_batch,y_batch], loss, updates=updates)
        
        for (x_batch, y_batch) in train_batches:
            # here x_batch and y_batch are elements of train_batches and
            # therefore numpy arrays; function MSGD also updates the params
            print('Current loss is ', MSGD(x_batch, y_batch))
            if stopping_condition_is_met:
                return params
        
        
    def fit(self, train_inputs, train_targets):
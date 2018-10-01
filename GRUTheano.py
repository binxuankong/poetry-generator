import numpy
import theano
import theano.tensor as T
from theano.gradient import grad_clip
import operator
import sys

class GRU:

  def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
    # Assign instance variables
    self.word_dim = word_dim
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate
    # Randomly initialize the network parameters
    E = numpy.random.uniform(-numpy.sqrt(1./word_dim), numpy.sqrt(1./word_dim), (hidden_dim, word_dim))
    U = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
    W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
    V = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (word_dim, hidden_dim))
    b = numpy.zeros((6, hidden_dim))
    c = numpy.zeros(word_dim)
    # Theano: Created shared variables
    self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
    self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
    self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
    self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
    self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
    self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
    # SGD / rmsprop: Initialize parameters
    self.mE = theano.shared(name='mE', value=numpy.zeros(E.shape).astype(theano.config.floatX))
    self.mU = theano.shared(name='mU', value=numpy.zeros(U.shape).astype(theano.config.floatX))
    self.mV = theano.shared(name='mV', value=numpy.zeros(V.shape).astype(theano.config.floatX))
    self.mW = theano.shared(name='mW', value=numpy.zeros(W.shape).astype(theano.config.floatX))
    self.mb = theano.shared(name='mb', value=numpy.zeros(b.shape).astype(theano.config.floatX))
    self.mc = theano.shared(name='mc', value=numpy.zeros(c.shape).astype(theano.config.floatX))
    # Store the Theano graph here
    self.theano = {}
    self.__theano_build__()

  def __theano_build__(self):
    E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c
    x = T.ivector('x')
    y = T.ivector('y')
    def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
      # Word embedding layer
      x_e = E[:,x_t]
      # GRU Layer 1
      z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
      r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
      c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
      s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
      # GRU Layer 2
      z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
      r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
      c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
      s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
      # Final output calculation
      # Theano's softmax returns a matrix with one row, we only need the row
      o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]
      return [o_t, s_t1, s_t2]
    [o, s, s2], updates = theano.scan(
      forward_prop_step,
      sequences=x,
      truncate_gradient=self.bptt_truncate,
      outputs_info=[None, dict(initial=T.zeros(self.hidden_dim)), dict(initial=T.zeros(self.hidden_dim))])
    prediction = T.argmax(o, axis=1)
    o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
    # Total cost (could add regularization here)
    cost = o_error
    # Gradients
    dE = T.grad(cost, E)
    dU = T.grad(cost, U)
    dW = T.grad(cost, W)
    db = T.grad(cost, b)
    dV = T.grad(cost, V)
    dc = T.grad(cost, c)
    # Assign functions
    self.forward_propagation = theano.function([x], o)
    self.predict = theano.function([x], prediction)
    self.ce_error = theano.function([x, y], cost)
    self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc])
    # SGD parameters
    learning_rate = T.scalar('learning_rate')
    decay = T.scalar('decay')
    # rmsprop cache updates
    mE = decay * self.mE + (1 - decay) * dE ** 2
    mU = decay * self.mU + (1 - decay) * dU ** 2
    mW = decay * self.mW + (1 - decay) * dW ** 2
    mV = decay * self.mV + (1 - decay) * dV ** 2
    mb = decay * self.mb + (1 - decay) * db ** 2
    mc = decay * self.mc + (1 - decay) * dc ** 2
    self.sgd_step = theano.function(
      [x, y, learning_rate, theano.In(decay, value=0.9)],
      [], 
      updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
               (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
               (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
               (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
               (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
               (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
               (self.mE, mE),
               (self.mU, mU),
               (self.mW, mW),
               (self.mV, mV),
               (self.mb, mb),
               (self.mc, mc)
              ])

  # Outer SGD Loop
  # - learning_rate: Initial learning rate for SGD
  # - nepoch: Number of times to iterate through the complete dataset
  def train_with_sgd(self, x_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9, callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
      # For each training example...
      for i in numpy.random.permutation(len(y_train)):
        # One SGD step
        self.sgd_step(x_train[i], y_train[i], learning_rate, decay)
        num_examples_seen += 1
        # Optionally do callback
        if (callback and callback_every and num_examples_seen % callback_every == 0):
          callback(self, num_examples_seen)

  def save_model_parameters(self, outfile):
    numpy.savez(outfile,
                E = self.E.get_value(),
                U = self.U.get_value(),
                W = self.W.get_value(),
                V = self.V.get_value(),
                b = self.b.get_value(),
                c = self.c.get_value())
    print("Saved model parameters to %s." %outfile)

  def load_model_parameters(self, path):
    numpyzfile = numpy.load(path)
    E, U, W, V, b, c = numpyzfile["E"], numpyzfile["U"], numpyzfile["W"], numpyzfile["V"], numpyzfile["b"], numpyzfile["c"]
    print("Building GRU using model parameters from %s with hidden_dim=%d word_dim=%d" %(path, E.shape[0], E.shape[1]))
    sys.stdout.flush()
    self.hidden_dim = E.shape[0]
    self.word_dim = E.shape[1]
    self.E.set_value(E)
    self.U.set_value(U)
    self.W.set_value(W)
    self.V.set_value(V)
    self.b.set_value(b)
    self.c.set_value(c)
    return "Built GRU using model parameters from %s with hidden_dim=%d word_dim=%d" %(path, E.shape[0], E.shape[1])
import numpy
import theano
import theano.tensor as T
import operator
import sys

class RNN:

  def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
    # Assign instance variables
    self.word_dim = word_dim
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate
    # Randomly initialize the network parameters
    U = numpy.random.uniform(-numpy.sqrt(1./word_dim), numpy.sqrt(1./word_dim), (hidden_dim, word_dim))
    V = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (word_dim, hidden_dim))
    W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    # Theano: Created shared variables
    self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
    self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
    self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
    # Store the Theano graph here
    self.theano = {}
    self.__theano_build__()

  def __theano_build__(self):
    U, V, W = self.U, self.V, self.W
    x = T.ivector('x')
    y = T.ivector('y')
    def forward_prop_step(x_t, s_t_prev, U, V, W):
      s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
      o_t = T.nnet.softmax(V.dot(s_t))
      return [o_t[0], s_t]
    [o,s], updates = theano.scan(
      forward_prop_step,
      sequences=x,
      outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
      non_sequences=[U, V, W],
      truncate_gradient=self.bptt_truncate,
      strict=True)
    prediction = T.argmax(o, axis=1)
    o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
    # Gradients
    dU = T.grad(o_error, U)
    dV = T.grad(o_error, V)
    dW = T.grad(o_error, W)      
    # Assign functions
    self.forward_propagation = theano.function([x], o)
    self.predict = theano.function([x], prediction)
    self.ce_error = theano.function([x, y], o_error)
    self.bptt = theano.function([x, y], [dU, dV, dW])
    # SGD
    learning_rate = T.scalar('learning_rate')
    self.sgd_step = theano.function([x,y,learning_rate], [], 
                    updates=[(self.U, self.U - learning_rate * dU),
                             (self.V, self.V - learning_rate * dV),
                             (self.W, self.W - learning_rate * dW)])

  def calculate_total_loss(self, X, Y):
      return numpy.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
  def calculate_loss(self, X, Y):
      # Divide calculate_loss by the number of words
      num_words = numpy.sum([len(y) for y in Y])
      return self.calculate_total_loss(X,Y)/float(num_words)

  def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = numpy.zeros(self.U.shape)
    dLdV = numpy.zeros(self.V.shape)
    dLdW = numpy.zeros(self.W.shape)
    delta_o = o
    delta_o[numpy.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in numpy.arange(T)[::-1]:
      dLdV += numpy.outer(delta_o[t], s[t].T)
      # Initial delta calculation
      delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
      # Backpropagation through time (for at most self.bptt_truncate steps)
      for bptt_step in numpy.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
        # print("Backpropagation step t=%d bptt step=%d " %(t, bptt_step))
        dLdW += numpy.outer(delta_t, s[bptt_step-1])              
        dLdU[:,x[bptt_step]] += delta_t
        # Update delta for next step
        delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]

  def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    self.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = self.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
      # Get the actual parameter value from the mode, e.g. self.W
      parameter_T = operator.attrgetter(pname)(self)
      parameter = parameter_T.get_value()
      print("Performing gradient check for parameter %s with size %d." %(pname, numpy.prod(parameter.shape)))
      # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
      it = numpy.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
      while not it.finished:
        ix = it.multi_index
        # Save the original value so we can reset it later
        original_value = parameter[ix]
        # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
        parameter[ix] = original_value + h
        parameter_T.set_value(parameter)
        gradplus = self.calculate_total_loss([x],[y])
        parameter[ix] = original_value - h
        parameter_T.set_value(parameter)
        gradminus = self.calculate_total_loss([x],[y])
        estimated_gradient = (gradplus - gradminus)/(2*h)
        parameter[ix] = original_value
        parameter_T.set_value(parameter)
        # The gradient for this parameter calculated using backpropagation
        backprop_gradient = bptt_gradients[pidx][ix]
        # calculate The relative error: (|x - y|/(|x| + |y|))
        relative_error = numpy.abs(backprop_gradient - estimated_gradient)/(numpy.abs(backprop_gradient) + numpy.abs(estimated_gradient))
        # If the error is to large fail the gradient check
        if relative_error > error_threshold:
          print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
          print("+h Loss: %f" % gradplus)
          print("-h Loss: %f" % gradminus)
          print("Estimated_gradient: %f" % estimated_gradient)
          print("Backpropagation gradient: %f" % backprop_gradient)
          print("Relative Error: %f" % relative_error)
          return 
        it.iternext()
      print("Gradient check for parameter %s passed." %pname)

  # Performs one step of SGD.
  def sgd_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW

  # Outer SGD Loop
  # - learning_rate: Initial learning rate for SGD
  # - nepoch: Number of times to iterate through the complete dataset
  # - evaluate_loss_after: Evaluate the loss after this many epochs
  def train_with_sgd(self, x, y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
      # Optionally evaluate the loss
      if (epoch % evaluate_loss_after == 0):
        loss = self.calculate_loss(x, y)
        losses.append((num_examples_seen, loss))
        # Adjust the learning rate if loss increases
        if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
          learning_rate = learning_rate * 0.5 
          print("Setting learning rate to %f" %learning_rate)
        sys.stdout.flush()
        # For each training example...
      for i in range(len(y)):
        # One SGD step
        self.sgd_step(x[i], y[i], learning_rate)
        num_examples_seen += 1
    return losses

  def save_model_parameters(self, outfile):
    numpy.savez(outfile,
                U = self.U.get_value(),
                V = self.V.get_value(),
                W = self.W.get_value())
    print("Saved model parameters to %s." %outfile)
   
  def load_model_parameters(self, path):
    npzfile = numpy.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    print("Building RNN using model parameters from %s with hidden_dim=%d word_dim=%d" %(path, U.shape[0], U.shape[1]))
    sys.stdout.flush()
    self.hidden_dim = U.shape[0]
    self.word_dim = U.shape[1]
    self.U.set_value(U)
    self.V.set_value(V)
    self.W.set_value(W)
    return "Built RNN using model parameters from %s with hidden_dim=%d word_dim=%d" %(path, U.shape[0], U.shape[1])
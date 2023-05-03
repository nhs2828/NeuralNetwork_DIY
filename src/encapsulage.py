import numpy as np
import copy as cp
from tqdm import tqdm

def logsoftmax(y):
  return np.exp(y)/np.sum(np.exp(y), axis = 1)[:, np.newaxis]


import numpy as np
import copy as cp

def logsoftmax(y):
  return np.exp(y)/np.sum(np.exp(y), axis = 1)[:, np.newaxis]


class Sequentiel():
  def __init__(self, *module):
    self.net = module

  def forward(self, input):
    z = input
    self.input = [input]
    for m in self.net:
      z = m.forward(z)
      self.input.append(z)
    return z
  
  def backward(self,loss, Y,gradient_step=0.001):
    delta_ = loss.backward(Y,self.input[-1])
    #print(len(self.input), len(self.net))
    for i in range(len(self.net)-1,-1,-1): # BACKKKKWARDDDDDDDD
      #print(type(self.net[i]).__name__)
      if type(self.net[i]).__name__ not in ['Linear', 'Conv1D']: # si fonction d'activation
        #print(f"SEQ ACTI {type(self.net[i]).__name__} input {self.input[i].shape} delta {delta_.shape}")
        delta_ = self.net[i].backward_delta(self.input[i], delta_)
      else: #Lineaire + convo
        #print("LIN")
        #print(f"SEQ LIN {type(self.net[i]).__name__} input {self.input[i].shape} delta {delta_.shape}")
        self.net[i].backward_update_gradient(self.input[i],delta_)
        delta_ = self.net[i].backward_delta(self.input[i], delta_)
        self.net[i].update_parameters(gradient_step=gradient_step)


class Optim():
  def __init__(self, net, loss, eps):
    self.net = net
    self.loss = loss
    self.eps = eps

  def step(self, batch_x, batch_y):
    y_hat = self.net.forward(batch_x)
    loss = self.loss.forward(batch_y, y_hat)
    self.net.backward(self.loss, batch_y,gradient_step=self.eps)
    return y_hat, loss
  
  def score(self, X, Y):
    if Y.shape[1] != 1:  #  OneHot
      Y = Y.argmax(axis=1)
      Y_hat = np.argmax(self.net.forward(X), axis=1)
    else:
      Y_hat = np.where(self.net.forward(X)>0.5,1,0)
    return np.where(Y_hat == Y, 1, 0).mean()

def SGD(net, data, loss, eps=1e-4, batch_taille=40, nb_iter=5):
    """Apprentissage du réseaux en utilisant mini-batch et descente de gradient
    Arg:
    net: le réseau de neurones
    data: le jeu de données X, Y
    loss: fonction cout
    eps: pas de gradient
    batch_taille: la taille de chaque batch
    nb_inter: nombre d'itérations
    """
    Opt = Optim(net, loss, eps)
    n = len(data)
    data_train = cp.deepcopy(data)
    X, Y = np.array(data_train[:,0].tolist()), np.array(data_train[:,1].tolist())
    l_loss = []
    l_score = []
    for i in tqdm(range(nb_iter)): # pour chaque epochs
        np.random.shuffle(data_train)
        tab_mini_batches = [data_train[k:k+batch_taille] for k in range(0, n, batch_taille)] #decouper en mini batches
        cout = 0
        for mini_batch in tab_mini_batches: # apprentissage sur chaque mini batch
            batch_x, batch_y = np.array(mini_batch[:,0].tolist()), np.array(mini_batch[:,1].tolist()) # array de array -> array de liste
            _, c = Opt.step(batch_x, batch_y)
            cout += c.sum()
        l_loss.append(cout)
        l_score.append(Opt.score(X, Y))
    return l_loss, l_score
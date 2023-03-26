class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss(Loss):
    def forward(self, y, yhat):
        """ Calculer le coût en fonction des 2 entrées
        """
        return np.linalg.norm(y-yhat, axis=1)**2

    def backward(self, y, yhat):
        """ Calculer le gradient du cout par rapport yhat
        """
        return -2*(y-yhat)


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

class Linear(Module):
    def __init__(self, input, output):
        """ Une couche linéaire dans le réseau de neurones

        Args:
            input (int): le nombre d'entrées
            output (int): le nombre de sorties
        """
        super().__init__()
        self._parameters = np.random.randn(output, input) # W
        #self._parameters = np.random.randn(input, output)
        #print(self._parameters.shape)
        self._gradient = np.zeros_like(self._parameters)

    def zero_grad(self):
        """ Réinitialiser à 0 le gradient
        """
        self._gradient = np.zeros_like(self._parameters)
    
    def forward(self, X):
        """ calculer les sorties du module pour les entrées passées en paramètre 
        """
        return np.dot(X,self._parameters.T) # <x,w>
        #return np.dot(X,self._parameters)
    
    
    def update_parameters(self, gradient_step=1e-3):
        """ Mettre à jour les paramètres du module selon le gradient accumulé 
            jusqu’à son appel avec un pas de gradient_step
        """ 
        self._parameters -= gradient_step*self._gradient
        self._parameters /= np.linalg.norm(self._parameters)
        self.zero_grad()
        
    def backward_update_gradient(self, input, delta):
        """ On est dans la couche h, calculer le gradient du coût par
            rapport aux paramètres et l’additionner à la variable _gradient
            - en fonction de l’entrée input et des δ de la couche suivante delta

        Args:
            input (array): z_h-1
            delta (_type_): _description_
        """
        #print(f"LIN MAJ GRAD input {input.shape} delta {delta.shape}")
        #print("BW_grad_Lin",delta.T.shape, input.shape)
        gradient = delta.T@input
        #print(gradient.shape)
        #print(self._gradient.shape)
        #gradient = np.dot(input.T, delta)
        # print(gradient.shape)
        # print(self._gradient.shape)
        self._gradient += gradient
        

    def backward_delta(self, input, delta):
        """ calculer le gradient du coût par rapport aux entrées 
            en fonction de l’entrée input et des deltas de la couche
            suivante delta
        """
        #print(f"LIN input {input.shape} delta {delta.shape}")
        return np.dot(delta, self._parameters)
        #return np.dot(delta, self._parameters.T)
    

class TanH(Module): # (e(z)-e(-z)) / (e(z)+e(-z))
    def __init__(self):
        """ Une couche tanH dans le réseau de neurones
        """
        super().__init__()
        

    def zero_grad(self):
        """ Réinitialiser à 0 le gradient
        """
        pass
    
    def forward(self, X):
        """ calculer les sorties du module pour les entrées passées en paramètre 
        """
        return  np.tanh(X)
    
    
    def update_parameters(self, gradient_step=1e-3):
        """ Mettre à jour les paramètres du module selon le gradient accumulé 
            jusqu’à son appel avec un pas de gradient_step
        """ 
        pass
        
    def backward_update_gradient(self, input, delta):
        """ On est dans la couche h, calculer le gradient du coût par
            rapport aux paramètres et l’additionner à la variable _gradient
            - en fonction de l’entrée input et des δ de la couche suivante delta

        Args:
            input (array): z_h-1
            delta (_type_): _description_
        """
        pass
        

    def backward_delta(self, input, delta):
        """ calculer le gradient du coût par rapport aux entrées 
            en fonction de l’entrée input et des deltas de la couche
            suivante delta
        """
        #print(f"TANH input {input.shape} delta {delta.shape}")
        dzda = (2/(np.exp(input)-np.exp(-input)))**2
        #print(input.shape)
        #print(dzda.shape)
        return delta*dzda     
    
class Sigmoide(Module): # 1 / (1+e(-z))
    def __init__(self):
        """ Une couche tanH dans le réseau de neurones
        """
        super().__init__()
        

    def zero_grad(self):
        """ Réinitialiser à 0 le gradient
        """
        pass
    
    def forward(self, X):
        """ calculer les sorties du module pour les entrées passées en paramètre 
        """
        return  1/(1+np.exp(-X))
    
    
    def update_parameters(self, gradient_step=1e-3):
        """ Mettre à jour les paramètres du module selon le gradient accumulé 
            jusqu’à son appel avec un pas de gradient_step
        """ 
        pass
        
    def backward_update_gradient(self, input, delta):
        """ On est dans la couche h, calculer le gradient du coût par
            rapport aux paramètres et l’additionner à la variable _gradient
            - en fonction de l’entrée input et des δ de la couche suivante delta

        Args:
            input (array): z_h-1
            delta (_type_): _description_
        """
        pass
        

    def backward_delta(self, input, delta):
        """ calculer le gradient du coût par rapport aux entrées 
            en fonction de l’entrée input et des deltas de la couche
            suivante delta
        """
        #print(f"SIG input {input.shape} delta {delta.shape}")
        dzda = np.exp(-input)/(1+np.exp(-input))**2
        #print(input.shape)
        #print(dzda.shape)
        return delta*dzda     

def f_sig_seq(seq, X):
  A = seq.forward(X)
  return np.where(A>0.5,1,-1)


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
      if type(self.net[i]).__name__ != 'Linear': # si fonction d'activation
        #print(f"SEQ ACTI {type(self.net[i]).__name__} input {self.input[i].shape} delta {delta_.shape}")
        delta_ = self.net[i].backward_delta(self.input[i], delta_)
      else: #Lineaire
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
  opt = Optim(net, loss, eps)
  n = len(data)
  data_train = cp.deepcopy(data)
  for i in range(nb_iter): # pour chaque epochs
    np.random.shuffle(data_train)
    tab_mini_batches = [data_train[k:k+batch_taille] for k in range(0, n, batch_taille)] #decouper en mini batches
    for mini_batch in tab_mini_batches: # apprentissage sur chaque mini batch
      batch_x, batch_y = np.array(mini_batch[:,0].tolist()), np.array(mini_batch[:,1].tolist()) # array de array -> array de liste
      opt.step(batch_x, batch_y)
import numpy as np

class Module(object):
    def __init__(self):
        self._parameters = {}
        self._gradient = {}

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
    def __init__(self, input, output, mode=0):
        """ Une couche linéaire dans le réseau de neurones

        Args:
            input (int): le nombre d'entrées
            output (int): le nombre de sorties
            mode (int): 0 -> distribution normal, 1 -> xavier uniforme
        """
        super().__init__()
        if mode == 0:
            self._parameters["w"] = np.random.randn(output, input)
            self._parameters["b"] = np.random.randn(output)
        else:
            borne = np.sqrt(6 / input)
            self._parameters["w"] = np.random.uniform(-borne, borne, (output, input))
            self._parameters["b"] = np.random.uniform(-borne, borne, (output))
        self._gradient['w'] = np.zeros_like(self._parameters['w'])
        self._gradient['b'] = np.zeros_like(self._parameters['b'])
        
    def zero_grad(self):
        """ Réinitialiser à 0 le gradient
        """
        self._gradient["w"] = np.zeros_like(self._parameters["w"])
        self._gradient["b"] = np.zeros_like(self._parameters["b"])
    
    def forward(self, X):
        """ calculer les sorties du module pour les entrées passées en paramètre 
        """
        #return np.dot(X,self._parameters.T)
        return np.dot(X,self._parameters["w"].T) + self._parameters["b"]
    
    
    def update_parameters(self, gradient_step=1e-3):
        """ Mettre à jour les paramètres du module selon le gradient accumulé 
            jusqu’à son appel avec un pas de gradient_step
        """ 
        self._parameters['w'] -= gradient_step*self._gradient['w']
        self._parameters['b'] -= gradient_step*self._gradient['b']
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
        #gradient = delta.T@input
        gradient = delta.T@input
        self._gradient['w'] += gradient
        self._gradient['b'] += delta.sum(axis=0)
        

    def backward_delta(self, input, delta):
        """ calculer le gradient du coût par rapport aux entrées 
            en fonction de l’entrée input et des deltas de la couche
            suivante delta
        """
        #print(f"LIN input {input.shape} delta {delta.shape} w {self._parameters['w'].shape}")
        #return delta@self._parameters
        return delta@self._parameters['w']

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
        return delta * (1 - np.tanh(input) ** 2) 
    
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
        return  delta * (1/(1+np.exp(-input))) * (1 - (1/(1+np.exp(-input))))

class ReLU(Module):
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
        return  np.maximum(0, X)
    
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
        return  delta * np.where(self.forward(input)>0 ,1,0)

def f_sig_seq(seq, X):
  A = seq.forward(X)
  return np.where(A>0.5,1,-1)

class Conv1D(Module):
  def __init__(self, k_size, chan_in, chan_out, stride=1, mode=0):
    self.k_size = k_size
    self.chan_in = chan_in
    self.chan_out = chan_out
    self.stride = stride
    if mode == 0:
        self._parameters = np.random.randn(chan_out, k_size*chan_in) # nb_couche X ksize*C
    else:
        borne = np.sqrt(6 / (self.chan_in + self.chan_out))
        self._parameters = np.random.uniform(-borne, borne, (self.k_size, self.chan_in, self.chan_out))
    self._gradient = np.zeros_like(self._parameters)



  def zero_grad(self):
    self._gradient = np.zeros_like(self._parameters)

  def forward(self, X):
    n_batch, d, C = X.shape
    dout = (d - self.k_size)//self.stride + 1 # une sortie est 2D d_out X chan_out
    o = np.zeros((b,dout,self.chan_out))
    for i in range(0,dout,self.stride):
        fenetre_X = X[:, i*self.stride:i*self.stride+self.k_size, :]
        o[:, i, :] = np.einsum('bkc,kco->bo', fenetre_X,self._parameters)
    return o
  
  def backward_update_gradient(self, input, delta):
    #print("COVO delta grad, delta shape", delta.shape)
    n_batch, d, C = input.shape
    dout = (d - self.k_size)//self.stride + 1
    for i in range(0,dout,self.stride):
        fenetre_X = X[:, i*self.stride:i*self.stride+self.k_size, :]
        d = delta[:,i,:]
        self._gradient += np.einsum('bkc,bo->kco', fenetre_X,d)


  def backward_delta(self, input, delta):
    #print("COVO delta, delta shape", delta.shape)
    n_batch, d, C = input.shape
    dout = (d - self.k_size)//self.stride + 1
    delta_h = np.zeros_like(input)
    for i in range(0,dout,self.stride):
        d = delta[:,i,:]
        delta_h[:, i*self.stride:i*self.stride+self.k_size, :] += np.einsum('bo,kco->bkc', d,W)
    return delta_h

  def update_parameters(self, gradient_step=1e-3):
      self._parameters -= gradient_step*self._gradient
      self.zero_grad()

    
class Flatten(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return X.reshape(X.shape[0], -1)
    
    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, gradient_step):
        pass

class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride
        self.cache = None

    def forward(self, X):
        n_batch, d, C = X.shape
        o = np.zeros((n_batch, (d - self.k_size) // self.stride + 1, C))
        cache = []
        for i in range(0, (d - self.k_size) // self.stride + 1, self.stride):
            x_window = X[:, i * self.stride : i * self.stride + self.k_size, :]
            #print("ici",x_window)
            x_max = np.max(x_window, axis=1)
            o[:, i, :] = x_max
            #print("o", o)
            cache.append((x_window, x_max))
        self.cache = cache
        #print("F", o.shape)
        return o

    def backward_delta(self, input, delta):
        #print("MAX pool delta shape", delta.shape)
        n_batch, d, C = input.shape
        delta_h = np.zeros_like(input)
        for i, (x_window, x_max) in enumerate(self.cache):
            delta_window = (x_window == x_max[:, None, :]) * delta[:, i, None, :]
            delta_h[:, i * self.stride : i * self.stride + self.k_size, :] += delta_window
        #print("B",delta_h.shape)
        return delta_h

    def update_parameters(self, gradient_step):
        pass  

    def zero_grad(self):
        pass  

    def backward_update_gradient(self, input, delta):
        pass  



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
        self.zero_grad()
        
    def backward_update_gradient(self, input, delta):
        """ On est dans la couche h, calculer le gradient du coût par
            rapport aux paramètres et l’additionner à la variable _gradient
            - en fonction de l’entrée input et des δ de la couche suivante delta

        Args:
            input (array): z_h-1
            delta (_type_): _description_
        """
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
        dzda = np.exp(-input)/(1+np.exp(-input))**2
        #print(input.shape)
        #print(dzda.shape)
        return delta*dzda     

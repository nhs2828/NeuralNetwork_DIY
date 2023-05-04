import numpy as np

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
        

class SMCELoss(Loss):
    def forward(self, y, yhat):
        """Coût cross-entropique
        y: indice de la classe à prédire, exp pour 1 exemple [0,0,0,1] -> classe 3
        y_hat: le vecteur de prédiction -> exp pour 1 exemple [0.1, 0.3, 0.4, 0.2]
        """
        #-yhat_y + log(sum_i(exp(yhat_i)))
        return -np.sum(y*yhat, axis = 1) + np.log(np.sum(np.exp(yhat), axis = 1))

    def backward(self, y, yhat):
        """ Calculer le gradient du cout par rapport yhat
        """
        exp_sum_exp = np.exp(yhat)/np.sum(np.exp(yhat), axis = 1)[:, np.newaxis] # exp/sum(exp(yhat_i))
        tmp_2 = np.where(y==1, -1, 0)
        return exp_sum_exp + tmp_2 # ou y = 1 -> -1


class BCELoss(Loss):
    def forward(self, y, yhat):
        """Coût cross-entropique binaire
        y: indice de la classe à prédire, exp pour 1 exemple [0,0,0,1] -> classe 3
        y_hat: le vecteur de prédiction -> exp pour 1 exemple [0.1, 0.3, 0.4, 0.2]
        """
        yhat = np.where(yhat == 0, 1e-9, yhat)
        yhat = np.where(yhat == 1, 0.999999999, yhat)
        return -(y*np.log(yhat) + (1-y)*np.log(1-yhat))

    def backward(self, y, yhat):
        """ Calculer le gradient du cout par rapport yhat
        """
        yhat = np.where(yhat == 0, 1e-9, yhat)
        yhat = np.where(yhat == 1, 0.999999999, yhat)
        return -(y/yhat + (y-1)/(1-yhat))

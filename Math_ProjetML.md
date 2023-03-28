# Linéaire
## MSELoss
### Forward
$MSE(y, \hat y) = ||\hat y - y||^2$

### Backward
$\frac{\partial MSE}{\partial \hat y} = \frac{\partial \sum_i^N||\hat y_i - y_i||^2}{\partial \hat y}$

On utilise le denominator layout

$2\left(\begin{array}{cccc} \hat y_{11}-y_{11} & \hat y_{12}-y_{12} & ... &\hat y_{1d}-y_{1d}\\ \hat y_{21}-y_{21} & \hat y_{22}-y_{22} & ... &\hat y_{2d}-y_{2d}\\ ...&...&...&...&\\\hat y_{b1}-y_{b1} & \hat y_{b2}-y_{b2} & ... &\hat y_{bd}-y_{bd} \end{array}\right) = 2\left(\begin{array}{c} \hat y_1-y_1\\ \hat y_2-y2\\ ... \\\hat y_b-y_b\end{array}\right)$

Donc:

$\frac{\partial MSE}{\partial \hat y} = -2(y - \hat y)$

## Module Linéaire
### Backward update gradient

$\frac{\partial Loss}{\partial W^h} = \frac {\partial Loss}{\partial z^h}\frac{\partial z^h}{\partial W^h}=\delta^h\frac{\partial z^h}{\partial W^h}$

On utilise le denominator layout

$z^h = \left(\begin{array}{cccc}z^h_1 & z^h_2 & ... & z^h_{d'}\end{array}\right) = \left(\begin{array}{cccc}\sum_i^d z^{h-1}_iw^h_{i1} & \sum_i^d z^{h-1}_iw^h_{i2} & ... & \sum_i^d z^{h-1}_iw^h_{id'}\end{array}\right)$
$W^h=\left(\begin{array}{c} w^h_1\\ w^h_2 \\ ... \\ w^h_{d'} \end{array}\right)=\left(\begin{array}{cccc} w^h_{11} & w^h_{21} & ... & w^h_{d1}\\ w^h_{12} & w^h_{22} & ... & w^h_{d2} \\ ... & ... & ... & ... \\ w^h_{1d'} & w^h_{2d'} & ... & w^h_{dd'}\end{array}\right)$

$\frac{\partial z^h}{\partial W^h} = \left(\begin{array}{cccccccccccc} z^{h-1}_1 & 0 & ... & 0 & z^{h-1}_2 & 0 & ... & 0 & z^{h-1}_d & 0 & 0 & 0 \\ 0 & z^{h-1}_1 & ... & 0 & 0 & z^{h-1}_2 & ... & 0 & 0 & z^{h-1}_d & 0 & 0 \\ ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... & ... \\ 0 & 0 & ... & z^{h-1}_1 & 0 & 0 & ... & z^{h-1}_2 & 0 & 0 & 0 & z^{h-1}_d \end{array} \right)$

$\frac{\partial z^h}{\partial W^h} = \left(\begin{array}{cccc} z^{h-1} & 0 & ... & 0 \\ 0 & z^{h-1} & ... & 0  \\ ... & ... & ... & ... \\ 0 & 0 & ... & z^{h-1}\end{array} \right)$
Dimension $(d',dd')$
$\delta^h = \left(\begin{array}{c} \delta^h_1 \\ \delta^h_2 \\ ... \\ \delta^h_{d'}\end{array}\right)$

$\frac{\partial Loss}{\partial W^h} =\delta^h\frac{\partial z^h}{\partial W^h}=\left(\begin{array}{cccc} \delta^h_1 & \delta^h_2 & ... & \delta^h_{d'}\end{array}\right)\left(\begin{array}{cccc} z^{h-1} & 0 & ... & 0 \\ 0 & z^{h-1} & ... & 0  \\ ... & ... & ... & ... \\ 0 & 0 & ... & z^{h-1}\end{array} \right)=(\delta^h)^Tz^{h-1}$
Dimension $(N,d') \times (d',dd') = (N, dd')$


### Backward delta

$\frac{\partial Loss}{\partial z^{h-1}} = \frac {\partial Loss}{\partial z^h}\frac{\partial z^h}{\partial z^{h-1}}=\delta^h\frac{\partial z^h}{\partial z^{h-1}}$

$z^{h-1} = \left(\begin{array}{c} z^{h-1}_1 \\ z^{h-1}_2 \\ ... \\ z^{h-1}_{d}\end{array}\right)$
$z^h = \left(\begin{array}{cccc}z^h_1 & z^h_2 & ... & z^h_{d'}\end{array}\right) = \left(\begin{array}{cccc}\sum_i^d z^{h-1}_iw^h_{i1} & \sum_i^d z^{h-1}_iw^h_{i2} & ... & \sum_i^d z^{h-1}_iw^h_{id'}\end{array}\right)$

$\frac{\partial z^h}{\partial z^{h-1}} = \left(\begin{array}{cccc} w^{h}_{11} & w^{h}_{12} & ... & w^{h}_{1d'} \\ w^{h}_{21} & w^{h}_{22} & ... & w^{h}_{2d'}  \\ ... & ... & ... & ... \\ w^{h}_{d1} & w^{h}_{d2} & ... & w^{h}_{dd'}\end{array} \right) = (W^h)^T$ 

$\frac{\partial Loss}{\partial z^{h-1}} =\delta^h\frac{\partial z^h}{\partial z^{h-1}} = \delta^h(W^h)^T$
Comme $z=<x.w>$, donc on utilise  la transition de $W$. Or:
$\frac{\partial Loss}{\partial z^{h-1}} =\delta^h\frac{\partial z^h}{\partial z^{h-1}} = \delta^hW^h$

# Non-linéaire

## Tangente hyperbolique
### Forward
$tanH (x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$

### Backward
```mermaid
flowchart LR
ModuleLinéaire --> a
a --> TanH
TanH --> z
```

$\frac{\partial Loss}{\partial a^h} = \frac {\partial Loss}{\partial z^h}\frac{\partial z^h}{\partial a^h}=\delta^h\frac{\partial z^h}{\partial a^h}$

$\frac{\partial z^h}{\partial a^h}=\frac{\partial\frac{e^{a^h} - e^{-a^h}}{e^{a^h} + e^{-a^h}}}{\partial a^h}=\frac{(e^{a^h} - e^{-a^h})'(e^{a^h} + e^{-a^h}) - (e^{a^h} + e^{-a^h})'(e^{a^h} - e^{-a^h})}{(e^{a^h} + e^{-a^h})^2}=\frac{(e^{a^h} + e^{-a^h})^2 - (e^{a^h} - e^{-a^h})^2}{(e^{a^h} + e^{-a^h})^2}=1-\frac{(e^{a^h} - e^{-a^h})^2}{(e^{a^h} + e^{-a^h})^2}$


Or:
$\frac{\partial z^h}{\partial a^h}=1 -tanH^2 (a^h)$

Donc:
$\frac{\partial Loss}{\partial a^h} =\delta^h\frac{\partial z^h}{\partial a^h} = \delta^h(1 -tanH^2 (a^h))$

## Sigmoide
### Forward
$\sigma(x) = \frac{1}{1+e^{-x}}$

## Backward
```mermaid
flowchart LR
ModuleLinéaire --> a
a --> Sigmoide
Sigmoide --> z
```

$\frac{\partial Loss}{\partial a^h} = \frac {\partial Loss}{\partial z^h}\frac{\partial z^h}{\partial a^h}=\delta^h\frac{\partial z^h}{\partial a^h}$

$\frac{\partial z^h}{\partial a^h}=\frac{\partial\frac{1}{1+e^{-a^h}}}{\partial a^h} = \frac{-(1+e^{-a^h})'}{(1+e^{-a^h})^2}=\frac{e^{-a^h}}{(1+e^{-a^h})^2}=\frac{1}{1+e^{-a^h}}\frac{e^{-a^h}}{1+e^{-a^h}}=\frac{1}{1+e^{-a^h}}\frac{1+e^{-a^h}-1}{1+e^{-a^h}}=\frac{1}{1+e^{-a^h}}(1-\frac{1}{1+e^{-a^h}})=\sigma(a^h)(1-\sigma(a^h))$

# Softmax et Cout Entropique
## Soft-max

$Softmax(x)=\frac{e^x}{\sum_i^d e^i}$

## Coût Cross-Entropique

Soit $y$ le vecteur supervision codé en one-hot.
Par exemple, si $y_i$ est de 3ème classe.
$y_i=\left(\begin{array}{c} 0 \\ 0 \\ 1 \\ 0 \end{array}\right)$
Et $\hat y_i$ est la vecteur de prédiction
$\hat y_i=\left(\begin{array}{c} 0.2 \\ 0.4 \\ 0.3 \\ 0.1 \end{array}\right)$

$CE(y_i, \hat y_i) = -<y_i. \hat y_i> = -(0*0.2+0*0.4+1*0.3+0*0.1) = -0.3$

Nous allons noter $y$ comme l'indice de la classe à prédire.
$CE(y, \hat y) = -\hat y_y$

## Combinaison de Softmax et coût cross-entropique

Afin d’éviter des instabilités numériques, on enchaîne un $Softmax$ passé au logarithme (l$ogSoftMax$) et un coût cross entropique comme une combinaison.

### Forward
$CE(y,\hat y) = -\log\frac{e^{\hat y_y}}{\sum_{i=1}^K e^{\hat y_i}} = -\hat y_y + \log \sum_{i=1}^K e^{\hat y_i}$

### Backward
$\frac{\partial Loss}{\partial \hat y}=\frac{\partial \sum_{i=1}^N -\hat y_{i,y_i} + log \sum_{j=1}^K e^{\hat y_{i,j}}}{\partial \hat y}$

On utilise le denominator layout

$\hat y = \left(\begin{array}{c} \hat y_1 \\ \hat y_2 \\ ... \\ \hat y_N \end{array}\right)=\left(\begin{array}{cccc} \hat y_{11} & \hat y_{12} & ... & \hat y_{1K} \\ \hat y_{21} & \hat y_{22} & ... & \hat y_{2K} \\ ... \\ \hat y_{N1} & \hat y_{N2} & ... & \hat y_{NK} \end{array}\right)$

$\frac{\partial Loss}{\partial \hat y_{kf}}=\frac{\partial \sum_{i=1}^N -\hat y_{i,y_i} + \log \sum_{j=1}^K e^{\hat y_{i,j}}}{\partial \hat y_{kf}}=\frac{\partial (-\hat y_{k,y_k} + \log \sum_{j=1}^K e^{\hat y_{k,j}})}{\partial \hat y_{kf}}$


Si $y_k = f$ :
$\frac{\partial Loss}{\partial \hat y_{kf}}=-1 + \frac{e^{\hat y_{kf}}}{\sum_{j=1}^K e^{\hat y_{k,j}}}$

Si $y_k \neq f$ :
$\frac{\partial Loss}{\partial \hat y_{kf}}=\frac{e^{\hat y_{kf}}}{\sum_{j=1}^K e^{\hat y_{k,j}}}$

Donc:

$\frac{\partial Loss}{\partial \hat y}=\left(\begin{array}{cccc} -y_{11} + \frac{e^{\hat y_{11}}}{\sum_{j=1}^K e^{\hat y_{1,j}}} & -y_{12} +\frac{e^{\hat y_{12}}}{\sum_{j=1}^K e^{\hat y_{1,j}}} & ... & -y_{1K} +\frac{e^{\hat y_{1K}}}{\sum_{j=1}^K e^{\hat y_{1,j}}} \\ -y_{21} + \frac{e^{\hat y_{21}}}{\sum_{j=1}^K e^{\hat y_{2,j}}} & -y_{22} + \frac{e^{\hat y_{22}}}{\sum_{j=1}^K e^{\hat y_{2,j}}} & ... & -y_{2K} + \frac{e^{\hat y_{2K}}}{\sum_{j=1}^K e^{\hat y_{2,j}}} \\ ... & ... & ... & ... \\ -y_{N1} + \frac{e^{\hat y_{N1}}}{\sum_{j=1}^K e^{\hat y_{N,j}}} & -y_{N2} + \frac{e^{\hat y_{N2}}}{\sum_{j=1}^K e^{\hat y_{N,j}}} & ... & -y_{NK} + \frac{e^{\hat y_{NK}}}{\sum_{j=1}^K e^{\hat y_{N,j}}} \end{array}\right)$

$\frac{\partial Loss}{\partial \hat y}= -y + Softmax(\hat y)$

# Se compresser
## Encodage
```mermaid
flowchart LR
N1(Linéaire.256_100) --> N2(TanH) --> N3(Linéaire.100_10) --> N4(Tanh)
```
## Décodage
```mermaid
flowchart LR
N1(Linéaire.10_100) --> N2(TanH) --> N3(Linéaire.100_256) --> N4(Sigmoide)
```
## Coût Cross-entropique binaire
### Forward
$BCE(y, \hat y) = -(y\log (\hat y) + (1-y)\log (1-\hat y))$

### Backward
$\frac{\partial Loss}{\partial \hat y} = -(\frac{y}{\hat y} + \frac{y-1}{1-\hat y})$





























































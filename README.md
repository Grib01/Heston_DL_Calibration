# Reseaux DNN pour la Calibration d'un Modele de Heston

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

## Description

Ce projet propose une approche par **Deep Learning** pour la **calibration du modele de Heston**, un modele de volatilite stochastique largement utilise en finance quantitative. Plutot que de recourir aux methodes d'optimisation numerique traditionnelles (lentes et parfois instables), nous entrainons un reseau de neurones profond (DNN) capable d'inverser directement une surface de volatilite implicite en ses cinq parametres de Heston.

## Auteurs

- **Mathis Ribordy**
- **Clement Rougeron**

*Projet realise dans le cadre du cours "Calibration de modeles" - Master 2*

---

## Table des matieres

1. [Contexte et modele de Heston](#contexte-et-modele-de-heston)
2. [Structure du projet](#structure-du-projet)
3. [Methodologie](#methodologie)
4. [Architecture des reseaux](#architecture-des-reseaux)
5. [Parametres](#parametres)
6. [Installation et utilisation](#installation-et-utilisation)
7. [Resultats](#resultats)
8. [References](#references)

---

## Contexte et modele de Heston

### Limites du modele de Black-Scholes

Le modele de Black-Scholes suppose une volatilite constante, ce qui ne reflete pas la realite des marches financiers. A l'inverse, le **modele de Heston** permet de capturer :
- Un niveau de volatilite variable
- La presence de skew (asymetrie du smile)
- Une structure par terme realiste

### Dynamique du modele de Heston

Sous la mesure risque-neutre, la dynamique du prix du sous-jacent S_t et de sa variance v_t est regie par :

```
dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW1_t

dv_t = kappa * (theta - v_t) * dt + sigma * sqrt(v_t) * dW2_t
```

### Parametres a calibrer

| Parametre | Description |
|-----------|-------------|
| `v0` | Variance initiale |
| `kappa` | Vitesse de retour a la moyenne |
| `theta` | Niveau de variance a long terme |
| `sigma` | Volatilite de la volatilite (courbure du smile) |
| `rho` | Correlation entre les deux mouvements browniens |

### Condition de Feller

Pour garantir que le processus de variance reste strictement positif :

```
2 * kappa * theta > sigma^2
```

---

## Structure du projet

```
Projet/
|-- Projet_Ribordy_Rougeron.ipynb   # Notebook principal
|-- cac40_hist.csv                   # Donnees historiques (optionnel)
|-- output.png                       # Exemple de sortie
|-- 1747921103319.pdf                # Documentation additionnelle
|-- README.md
```

---

## Methodologie

### 1. Generation de donnees synthetiques

Les donnees de marche historiques etant inaccessibles, nous generons un jeu de donnees synthetiques :

- **16 000 echantillons** couvrant plusieurs regimes de marche
- **4 regimes distincts** : core (45%), stress (20%), high_skew (20%), fast_mean_reversion (15%)
- **Filtrage rigoureux** : condition de Feller, controles ATM, amplitude du smile

### 2. Grille de calibration

| Element | Valeur |
|---------|--------|
| Moneyness (K/S0) | 11 niveaux de 0.5 a 1.5 |
| Maturites | 8 niveaux de 0.1 a 2.0 ans |
| Points de surface | 88 (11 x 8) |
| Taux sans risque | 2% |
| Spot de reference | 1.0 (normalise) |

### 3. Pricer semi-analytique de Heston

Le prix d'un Call europeen dans le modele de Heston s'ecrit :

```
C(S0, K, T) = S0 * P1 - K * exp(-rT) * P2
```

Les probabilites P1 et P2 sont obtenues par inversion de la fonction caracteristique via l'integrale de Gil-Pelaez, discretisee par **quadrature de Gauss-Legendre** (48 points).

### 4. Conversion en volatilite implicite

Plutot que de travailler sur les prix, nous utilisons les **volatilites implicites Black-Scholes** :
- Normalisation des donnees (echelle comparable)
- Meilleure convergence du reseau
- Evite la concentration sur les options ITM

L'inversion est effectuee par **Newton-Raphson vectorise**.

### 5. Pretraitement

| Transformation | Application |
|----------------|-------------|
| Standardisation | Surfaces de volatilite (mean=0, std=1) |
| Rescaling [-1, 1] | Parametres de Heston (compatible avec tanh) |

---

## Architecture des reseaux

### Calibrateur DNN (Surface -> Parametres)

```
Input: Surface IV (88 points)
         |
         v
+----------------------------------+
|  Dense(192) + BatchNorm + ELU    |
|  Dropout(0.15)                   |
+----------------------------------+
         |
         v
+----------------------------------+
|  Bloc Residuel (128 neurones)    |
|  Bloc Residuel (128 neurones)    |
|  Bloc Residuel (64 neurones)     |
+----------------------------------+
         |
         v
+----------------------------------+
|  Dense(64) + BatchNorm + ELU     |
+----------------------------------+
         |
         v
+----------------------------------+
|  Dense(5, activation=tanh)       |
|  -> [v0, rho, sigma, theta, kappa]|
+----------------------------------+
```

### Generateur DNN (Parametres -> Surface)

Architecture symetrique pour validation bidirectionnelle :
- Entree : 5 parametres
- Sortie : 88 points de surface
- Role : diagnostic interne (boucle surface -> params -> surface)

### Blocs residuels

Chaque bloc residuel inclut :
- Dense + BatchNormalization
- Activation ELU
- Dropout (regularisation)
- Connexion shortcut (facilite la propagation du gradient)

---

## Fonction de perte

La fonction `advanced_calibration_loss` combine :

### 1. RMSE ponderee

Poids differencies selon l'impact sur la surface :

| Parametre | Poids |
|-----------|-------|
| v0 | 20.0 |
| theta | 15.0 |
| sigma | 10.0 |
| rho | 5.0 |
| kappa | 5.0 |

### 2. Penalites physiques

| Contrainte | Penalite |
|------------|----------|
| Violation Feller (2*kappa*theta <= sigma^2) | lambda = 10.0 |
| Correlation extreme (\|rho\| >= 0.999) | lambda = 5.0 |
| Variance negative (v0 <= 0) | lambda = 5.0 |

---

## Parametres

### Bornes des parametres Heston

| Parametre | Min | Max |
|-----------|-----|-----|
| v0 | 0.004 | 0.16 |
| rho | -0.98 | -0.15 |
| sigma | 0.04 | 0.95 |
| theta | 0.008 | 0.16 |
| kappa | 0.35 | 9.0 |

### Configuration d'entrainement

| Parametre | Valeur |
|-----------|--------|
| Learning rate | 7e-4 |
| Batch size | 256 |
| Epochs | 600 |
| Validation split | 20% |
| LR patience | 12 |
| Early stopping patience | 25 |

---


## References

1. Heston, S.L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*. The Review of Financial Studies.

2. Albrecher, H., et al. (2007). *The little Heston trap*. Wilmott Magazine.

3. He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.

4. Horvath, B., et al. (2021). *Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models*. Quantitative Finance.


# Réseaux de neurones

> Ce chapitre explique comment fonctionne un réseau de neurones, du neurone isolé
> jusqu'aux architectures modernes (CNN, RNN, GAN, diffusion). C'est la brique de base
> sur laquelle reposent tous les grands modèles actuels : sans comprendre les poids,
> le gradient et la rétropropagation, on ne peut pas lire la suite du guide.

Avant de commencer : une **tension** (tenseur) est un tableau de nombres à n dimensions.
Un vecteur est une tension de rang 1, une matrice de rang 2. Un réseau de neurones enchaîne
des multiplications de matrices : c'est du calcul numérique pur. Les rappels utiles
(produit scalaire, dérivée, règle de dérivation en chaîne) se trouvent dans
[le chapitre 02](../02-mathematiques/README.md).

## 1. Le neurone artificiel

Un neurone artificiel est une petite fonction mathématique. Il prend plusieurs nombres
en entrée, applique des poids à chacun, fait la somme, ajoute une constante, puis
passe le résultat dans une fonction non linéaire.

On note `x` le vecteur d'entrée, `w` le vecteur des **poids** (poids = importance
apprise de chaque entrée), et `b` le **biais** (biais = constante qui décale la sortie).
La **pré-activation** `z` est le produit scalaire des deux, plus le biais :

```
z = w . x + b = w1*x1 + w2*x2 + ... + wn*xn + b
a = f(z)
```

`f` est la **fonction d'activation** : c'est elle qui introduit la non-linéarité.
Le produit scalaire mesure à quel point l'entrée ressemble à ce que le neurone
recherche : si `w` et `x` pointent dans la même direction, `z` est grand et positif.

Le biais compte vraiment : sans lui, un neurone dont toutes les entrées sont nulles
produit toujours zéro, quelle que soit la question. Le biais permet de décaler le seuil
de déclenchement.

Ordre de grandeur : un « petit » réseau de vision a des couches de 64 à 512 neurones.
Un modèle de langage de l'ordre de 10 milliards de paramètres empile des milliers de
dimensions par activation. Le neurone est le même dans les deux cas.

## 2. Perceptron et la limite du XOR

Le **perceptron** (Rosenblatt, 1958) est le plus simple des réseaux : un seul neurone
avec une activation en échelon (sortie 0 ou 1). Il trace une frontière de décision
linéaire : une droite en 2D, un plan en 3D.

Il sait apprendre `AND` et `OR`. Il échoue sur `XOR` (ou exclusif) : la sortie vaut 1
quand les deux entrées sont différentes, 0 quand elles sont égales.

| x1 | x2 | AND | OR | XOR |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 |

Pour XOR, les points de classe 1 (haut-gauche et bas-droite) ne sont pas séparables
des points de classe 0 (bas-gauche et haut-droite) par une seule droite. Un perceptron
unique ne peut pas le faire, quel que soit l'entraînement.

La solution : empiler une **couche cachée** de neurones. Avec deux neurones cachés
(un qui détecte `OR`, un qui détecte `NAND`) et une sortie qui combine les deux,
un XOR devient résoluble. C'est l'idée fondatrice du perceptron multicouche.

> Idée clé : une seule couche linéaire ne peut pas résoudre un problème non
> linéairement séparable. Empiler des couches avec des activations non linéaires le peut.

## 3. MLP, couches cachées et approximation universelle

Un **MLP** (Multi-Layer Perceptron, perceptron multicouche) est une suite de couches
entièrement connectées (chaque neurone d'une couche est relié à tous ceux de la
suivante). Par convention, toutes les couches entre l'entrée et la sortie sont dites
**cachées**.

Schéma ASCII d'un MLP à deux couches cachées :

```
entrée      couche 1        couche 2        sortie
 x1 --\   |-- n1 --\     |-- m1 --\
 x2 ---+--|-- n2 ---+----|-- m2 ---+----- y
 x3 --/   |-- n3 --/     |-- m3 --/
          (activation)   (activation)   (activation)
```

Le **théorème d'approximation universelle** (Cybenko 1989, Hornik 1991) dit : un réseau
à une seule couche cachée, avec suffisamment de neurones et une activation non linéaire
raisonnable, peut approximer n'importe quelle fonction continue sur un domaine borné,
à une précision arbitraire.

Deux nuances importantes que ce théorème ne dit PAS :
- il ne dit rien sur le nombre de neurones nécessaires (il peut être astronomique) ;
- il ne dit rien sur la capacité à apprendre ces poids par gradient (l'entraînement
  n'est pas garanti de trouver la bonne configuration).

En pratique, les réseaux **profonds** (beaucoup de couches) sont plus efficaces que
les réseaux larges (une couche énorme) pour une même capacité, car ils réutilisent
des représentations hiérarchiques.

## 4. Fonctions d'activation

Une fonction d'activation transforme la pré-activation `z` en sortie `a`. Sans
non-linéarité, empiler des couches revient à une seule transformation linéaire :
`W2*(W1*x) = (W2*W1)*x`. Le réseau profond n'apporterait alors rien.

| Nom | Formule | Sortie | Usage typique |
|---|---|---|---|
| Sigmoïde | σ(z) = 1/(1+e^-z) | (0,1) | sorties binaires, portes (ancien) |
| Tanh | (e^z - e^-z)/(e^z + e^-z) | (-1,1) | couches cachées (ancien) |
| ReLU | max(0, z) | [0,∞) | défaut CNN/MLP |
| LeakyReLU | max(0.01z, z) | (-∞,∞) | quand ReLU meurt |
| GELU | z * Φ(z) | ≈(-0.17,∞) | transformers (BERT, GPT) |
| SiLU (Swish) | z * σ(z) | ≈(-0.28,∞) | modèles modernes (Llama) |
| Softmax | e^zi / Σ e^zj | (0,1), somme=1 | sortie multi-classes |

**Sigmoïde** écrase les valeurs extrêmes vers 0 ou 1. Sa dérivée est petite (max 0.25),
donc sur des réseaux profonds elle provoque la **disparition du gradient** : l'erreur
s'atténue à chaque couche et les premières couches n'apprennent plus. **Tanh** a le
même souci, mais centrée en zéro.

**ReLU** (Rectified Linear Unit) est devenue le défaut car : (1) sa dérivée vaut
exactement 1 pour z>0, donc le gradient passe bien ; (2) elle est très rapide à calculer.
Son défaut est le « neurone mort » : si un neurone reste toujours en z<0, son gradient
est nul et il ne s'en sortira jamais. **LeakyReLU** laisse passer une petite pente
pour les valeurs négatives et évite ce blocage.

**GELU** et **SiLU** sont des ReLU lissées, non nulles pour z légèrement négatif.
Elles coûtent un peu plus cher mais donnent de meilleurs résultats sur les modèles
de langage ; SiLU (via la variante SwiGLU) est très répandue dans Llama, Mistral,
Qwen.

**Softmax** transforme un vecteur de scores en distribution de probabilité. Elle est
utilisée uniquement en sortie de classification, ou comme normalisation dans l'attention
(voir [chapitre 04](../04-transformers/README.md)).

## 5. Initialisation des poids

On ne peut pas initialiser tous les poids à zéro : tous les neurones d'une couche
seraient identiques et le resteraient (symétrie). Mais une mauvaise initialisation
aléatoire casse aussi l'entraînement : trop grande, les activations explosent ;
trop petite, elles s'éteignent.

**Initialisation de Xavier (Glorot)** : conçue pour tanh/sigmoïde, elle tire les poids
avec une variance de `2 / (n_in + n_out)`, où `n_in` et `n_out` sont le nombre
d'entrées et de sorties de la couche. Cela garde la variance des activations stable
d'une couche à l'autre.

**Initialisation de He (Kaiming)** : adaptée à ReLU. Comme ReLU annule la moitié des
entrées en moyenne, la variance est doublée : `2 / n_in`.

Règle pratique : ReLU -> He. Tanh/sigmoïde -> Xavier. Sur un gros modèle, une mauvaise
initialisation se voit immédiatement : la perte reste bloquée ou part à NaN dès les
premières centaines d'itérations.

## 6. Rétropropagation pas à pas

La **rétropropagation** (backpropagation) calcule la dérivée de la perte par rapport
à chaque poids. Elle repose sur la règle de dérivation en chaîne : la dérivée d'une
composition est le produit des dérivées.

Prenons le plus petit réseau possible : une entrée `x`, un neurone caché, un neurone
de sortie, activations sigmoïdes, perte quadratique.

```
x --[w1,b1]--> z1 --σ--> a1 --[w2,b2]--> z2 --σ--> y_hat
                                                     |
                                          L = 0.5*(y - y_hat)^2
```

Étapes de la passe avant :
```
z1 = w1*x + b1
a1 = σ(z1)
z2 = w2*a1 + b2
y_hat = σ(z2)
L = 0.5 * (y - y_hat)^2
```

Passe arrière (on propage l'erreur de la sortie vers l'entrée) :
```
dL/dy_hat = -(y - y_hat)
dL/dz2 = dL/dy_hat * σ'(z2)          = -(y - y_hat) * y_hat*(1-y_hat)
dL/dw2 = dL/dz2 * a1
dL/db2 = dL/dz2
dL/da1 = dL/dz2 * w2
dL/dz1 = dL/da1 * σ'(z1)             = dL/da1 * a1*(1-a1)
dL/dw1 = dL/dz1 * x
dL/db1 = dL/dz1
```

**Mini exemple numérique** (valeurs volontairement simples) :
```
x=1.0  y=1.0  w1=0.5  b1=0.0  w2=0.3  b2=0.0

passe avant :
  z1 = 0.5*1 + 0 = 0.5
  a1 = σ(0.5)      ≈ 0.6225
  z2 = 0.3*0.6225  ≈ 0.1868
  y_hat = σ(0.1868) ≈ 0.5466
  L = 0.5*(1-0.5466)^2 ≈ 0.1028

passe arrière :
  dL/dy_hat = -0.4534
  σ'(z2) = 0.5466*0.4534 ≈ 0.2478
  dL/dz2 ≈ -0.1124
  dL/dw2 = -0.1124 * 0.6225 ≈ -0.0700
  dL/da1 = -0.1124 * 0.3    ≈ -0.0337
  σ'(z1) = 0.6225*0.3775    ≈ 0.2350
  dL/dz1 ≈ -0.0079
  dL/dw1 = -0.0079 * 1.0    ≈ -0.0079

mise à jour (η = 0.1) :
  w2 <- 0.3 - 0.1*(-0.0700) = 0.3070
  w1 <- 0.5 - 0.1*(-0.0079) = 0.5008
```

Point notable : le gradient de `w1` (-0.0079) est bien plus petit que celui de `w2`
(-0.0700). C'est la disparition du gradient en action : plus on est loin de la sortie,
plus le signal est faible. Sur de nombreuses couches, ce facteur devient négligeable ;
d'où les techniques des sections suivantes.

## 7. Optimiseurs et taux d'apprentissage

Le **gradient** indique la direction de la pente. L'optimiseur décide comment s'en
servir pour modifier les poids. Le **taux d'apprentissage** (learning rate, `η`)
est la taille du pas.

| Optimiseur | Idée | Quand l'utiliser |
|---|---|---|
| SGD | pas proportionnel au gradient | baseline, parfois meilleure généralisation |
| Momentum | ajoute de l'inertie au pas | quand les gradients oscillent |
| RMSProp | pas adapté par paramètre | gradients d'échelles très différentes |
| Adam | momentum + RMSProp, corrigé du biais | défaut robuste, convergence rapide |
| AdamW | Adam avec weight decay découplé | défaut des transformers |

**SGD** : `θ <- θ - η * g`. Simple mais sensible au choix de `η` et peut zigzaguer.

**Momentum** : on accumule une vitesse `v <- γ*v + η*g`, puis `θ <- θ - v`. Comme une
bille qui roule, cela traverse les petits trous et lisse les oscillations.

**RMSProp** : on divise le pas par la racine de la moyenne des carrés récents des
gradients. Les paramètres à gros gradient sont freinés, les petits accélérés.

**Adam** : combine momentum (premier moment) et RMSProp (second moment), avec une
correction de biais au démarrage. C'est le couteau suisse, souvent le réglage par
défaut.

**AdamW** : dans Adam, la régularisation L2 est noyée dans le gradient et se comporte
mal avec l'adaptation du pas. AdamW applique la décroissance des poids directement,
hors du calcul du gradient. C'est le standard pour entraîner les transformers.

### Schedulers et warmup

Un taux d'apprentissage fixe est rarement optimal. Un **scheduler** le fait varier au
cours de l'entraînement :
- **Warmup** : commencer très bas et monter linéairement sur les premières centaines
  ou milliers d'itérations, pour ne pas casser le modèle avec des gradients instables.
- **Cosine decay** : décroître en suivant un cosinus jusqu'à un minimum. Très courant.
- **Step decay** : diviser le taux par 10 tous les N epochs.
- **ReduceLROnPlateau** : réduire quand la perte de validation stagne.

Arbitrage : un scheduler complique le réglage et l'arrêt anticipé, mais améliore
presque toujours la qualité finale. Sur les gros modèles, warmup + cosine est le
standard de fait.

## 8. Normalisation et connexions résiduelles

Deux inventions ont rendu possible l'entraînement de réseaux très profonds.

**Normalisation** : on recentre et remet à l'échelle les activations pour qu'elles
gardent une distribution raisonnable d'une couche à l'autre.
- **BatchNorm** : normalise sur le lot (moyenne/variance calculées sur les exemples
  du batch, par canal). Très efficace en vision, mais dépend de la taille du lot
  et pose problème aux séquences de longueurs variables.
- **LayerNorm** : normalise sur les caractéristiques d'un seul exemple, indépendamment
  du lot. C'est la norme pour les transformers.
- **RMSNorm** : comme LayerNorm mais ne soustrait pas la moyenne, normalise seulement
  par la moyenne quadratique. Plus rapide, utilise dans Llama, Mistral, T5.

**Connexions résiduelles** (skip connections), popularisées par ResNet (2015) :
on ajoute l'entrée d'un bloc à sa sortie : `y = F(x) + x`. Le gradient peut alors
traverser le bloc sans être multiplié par `F`. Un réseau résiduel peut avoir des
centaines de couches sans que le gradient s'éteigne.

> Sans résidus + normalisation, les réseaux au-delà d'une vingtaine de couches
> deviennent en pratique impossibles à entraîner correctement.

## 9. Dropout et régularisation

Le **dropout** (Srivastava et al., 2014) désactive aléatoirement une fraction des
neurones à chaque passe d'entraînement (souvent 10 à 50 pour cent). Le réseau ne peut
plus compter sur un neurone précis et doit apprendre des représentations redondantes.
C'est une régularisation : moins d'overfitting, mais un peu plus lent à converger.
À l'inférence, le dropout est désactivé.

Autres régularisations courantes :
- **L2 / weight decay** : pénalise les poids trop grands, lisse le modèle.
- **L1** : pousse certains poids exactement à zéro (parcimonie).
- **Data augmentation** : transformer les exemples d'entraînement pour enrichir le jeu
  de données (voir [chapitre 12](../12-donnees/README.md)).
- **Early stopping** : arrêter quand la perte de validation remonte.

Arbitrage classique : trop de régularisation -> underfitting ; pas assez -> overfitting.
La bonne valeur se trouve par validation, pas par règle absolue.

## 10. Architecture CNN (vision)

Les réseaux convolutifs (CNN) sont conçus pour les images (et plus largement les
données organisées spatialement). Au lieu de relier chaque pixel à chaque neurone,
on applique un petit filtre qui glisse sur l'image.

- **Convolution** : un noyau (par ex. 3x3) multiplie localement une fenêtre de l'image
  et somme les produits. Le résultat est une carte d'activation.
- **Partage des poids** : le même noyau est réutilisé à toutes les positions. Cela
  réduit énormément le nombre de paramètres.
- **Pooling** : on sous-échantillonne en gardant le maximum (max-pooling) ou la moyenne
  d'une petite fenêtre. Réduit la résolution et donne une certaine invariance.
- **Champ réceptif** : zone de l'image d'origine qui influence une activation donnée.
  Il grandit avec la profondeur et les couches de pooling.

**Invariance par translation** : un motif (un bord, un œil) reconnu en haut à gauche
sera reconnu partout ailleurs, car le même noyau s'applique partout. C'est exactement
ce qu'on veut pour la vision.

Ordre de grandeur : LeNet (1989) avait de l'ordre de 60 000 paramètres ; ResNet-50
(2015) environ 25 millions ; un backbone de vision moderne peut dépasser le milliard.

## 11. Architecture RNN, LSTM, GRU (séquences)

Les **RNN** traitent une séquence élément par élément, en gardant un **état caché**
qui résume le passé.

```
h_t = f(W_h * h_{t-1} + W_x * x_t + b)
```

La même matrice `W_h` est réutilisée à chaque pas de temps. Le problème : à la
rétropropagation, on multiplie ce même terme des centaines de fois, ce qui fait
disparaître (ou exploser) le gradient. Un RNN simple oublie donc vite le début d'une
longue phrase.

**LSTM** (Long Short-Term Memory) et **GRU** (Gated Recurrent Unit) résolvent cela
avec des **portes** qui décident quoi écrire, garder ou oublier. Le LSTM a trois portes
(entrée, oubli, sortie) et un état de cellule ; le GRU a deux portes et est plus léger.
Ces mécanismes permettent de retenir des dépendances sur des centaines de pas.

Malgré cela, les RNN restent **séquentiels** : on ne peut pas paralléliser le calcul
sur le temps, ce qui les rend lents à entraîner sur GPU. C'est la raison principale
de leur remplacement par les transformers, voir [chapitre 04](../04-transformers/README.md).

## 12. Autoencodeurs, GAN, VAE, diffusion

Ces architectures servent à apprendre des représentations ou à générer des données.

**Autoencodeur** : un encodeur comprime l'entrée en un **espace latent** (représentation
condensée), un décodeur la reconstruit. La perte est l'erreur de reconstruction.
Utile pour la compression et le débruitage.

```
x --> [encodeur] --> z (latent) --> [décodeur] --> x_hat
                                            perte = ||x - x_hat||^2
```

**GAN** (Generative Adversarial Network) : deux réseaux s'affrontent. Le **générateur**
fabrique des données à partir de bruit ; le **discriminateur** essaie de distinguer
le vrai du faux. Le générateur apprend à tromper. Puissant pour l'image, mais
l'entraînement est instable (le jeu peut diverger).

**VAE** (Variational Autoencoder) : au lieu d'un point, l'encodeur produit une
distribution (moyenne et variance) dans l'espace latent. On échantillonne dedans, ce
qui rend la génération lisse et contrôlable. On ajoute une pénalité (divergence KL)
qui garde l'espace latent bien ordonné. Contrepartie : les images générées sont
souvent plus floues qu'avec un GAN.

**Modèles de diffusion** : on ajoute progressivement du bruit à une image jusqu'à
obtenir du bruit pur (processus direct), puis on apprend à inverser ce processus
(dénaturation étape par étape). Le réseau apprend à prédire le bruit ajouté à chaque
étape, et génère en le retirant. DDPM a formalisé cette approche ; les **modèles de
diffusion latente** (comme Stable Diffusion) font le processus dans l'espace latent
d'un autoencodeur, ce qui réduit énormément le coût de calcul.

## 13. GNN et transfer learning

**GNN** (Graph Neural Networks) opèrent sur des graphes : chaque nœud agrège les
informations de ses voisins, couche après couche, pour construire sa représentation.
Utilisés en chimie (molécules), réseaux sociaux, recommandation. Voir aussi
[chapitre 17](../17-au-dela/README.md).

**Transfer learning** : on part d'un modèle pré-entraîné sur une grande tâche et on
l'adapte à une tâche proche. Pourquoi ça marche : les premières couches apprennent des
caractéristiques génériques (contours, textures, structures syntaxiques) réutilisables,
et seules les dernières couches sont spécifiques à la tâche cible. On gèle souvent
les premières couches et on n'entraîne que la fin, ce qui demande beaucoup moins de
données et de calcul. Le transfert complet côté langage (fine-tuning, LoRA) est traité
dans [chapitre 06](../06-fine-tuning/README.md).

Cas extrême : avec un modèle pré-entraîné, quelques centaines d'exemples suffisent
souvent pour une tâche de classification correcte, là où un réseau partant de zéro
en demanderait des dizaines de milliers.

## 14. Ce qu'il faut retenir

- Un neurone calcule un produit scalaire pondéré plus un biais, puis applique une
  fonction d'activation non linéaire.
- Sans non-linéarité, empiler des couches ne fait rien de plus qu'une seule couche
  linéaire.
- Le perceptron seul ne résout pas XOR ; une couche cachée suffit pour le faire.
- Le théorème d'approximation universelle garantit la capacité d'exprimer, pas la
  capacité d'apprendre : le réglage et les données comptent autant que l'architecture.
- ReLU évite la saturation du gradient et est devenue l'activation par défaut ; GELU
  et SiLU dominent dans les transformers.
- L'initialisation (Xavier pour tanh, He pour ReLU) conditionne la stabilité de
  l'entraînement.
- La rétropropagation applique la règle de dérivation en chaîne ; le gradient
  s'atténue en s'éloignant de la sortie, ce que résidu et normalisation corrigent.
- AdamW avec warmup et décroissance cosinus est le réglage de référence pour les gros
  modèles.
- Les résidus et la normalisation sont indispensables pour dépasser quelques dizaines
  de couches.
- CNN, RNN, GAN, VAE et diffusion répondent chacun à un type de données ou d'objectif,
  mais les transformers ont largement supplanté les RNN pour le texte.

## 15. Erreurs fréquentes / idées reçues

- "Un réseau plus profond est toujours meilleur" -> sans résidus et normalisation, la
  profondeur dégrade l'entraînement ; et au-delà d'un certain point, on surajuste.
- "Le réseau fonctionne comme un cerveau" -> c'est une inspiration, pas un modèle
  biologique : l'apprentissage par gradient n'a rien à voir avec la plasticité
  synaptique réelle.
- "Il faut toujours d'énormes jeux de données" -> le transfer learning permet
  d'atteindre de bonnes performances avec peu d'exemples, si le pré-entraînement
  est bon.
- "Un modèle plus gros est forcément meilleur" -> au même budget de calcul, un modèle
  trop gros sous-entraîné est battu par un modèle plus petit bien entraîné.
- "Le taux d'apprentissage doit être tout petit" -> trop petit, la convergence est
  interminable ; warmup et scheduler font souvent mieux qu'un taux fixe minuscule.
- "Dropout améliore toujours" -> sur les grands modèles modernes, il est souvent
  retiré ou très faible ; il peut nuire à la convergence.
- "La rétropropagation calcule des dérivées exactes globales" -> elle calcule des
  gradients locaux par composition ; le résultat n'est exact que pour le point courant.
- "Softmax est une activation comme les autres" -> elle est surtout une normalisation
  de vecteur, utilisée en sortie ou dans l'attention.

## 16. Pour aller plus loin

Dans l'encyclopédie :
- [Chapitre 02 : mathématiques](../02-mathematiques/README.md) pour le produit scalaire,
  les dérivées et la règle de dérivation en chaîne.
- [Chapitre 04 : transformers](../04-transformers/README.md), la suite directe de ce
  chapitre : l'attention remplace les RNN pour le texte.
- [Chapitre 06 : fine-tuning](../06-fine-tuning/README.md) pour le transfer learning
  et l'adaptation de modèles pré-entraînés.
- [Chapitre 10 : infrastructure](../10-infrastructure/README.md) pour le matériel
  (GPU, VRAM) qui rend ces calculs possibles.

Ressources externes réelles :
- "Deep Learning", Ian Goodfellow, Yoshua Bengio, Aaron Courville (MIT Press, 2016).
- Cours CS231n, Stanford, sur les réseaux convolutifs pour la vision.
- Article "Learning représentations by back-propagating errors", Rumelhart, Hinton,
  Williams, Nature, 1986.
- Article "Deep Residual Learning for Image Recognition", He et al., 2015 (ResNet).
- Article "Adam: A Method for Stochastic Optimization", Kingma et Ba, 2014.

# Les maths de l'IA, sans douleur

> On n'a pas besoin d'un bagage universitaire pour comprendre les réseaux de
> neurones : il faut cinq idées, expliquées une fois, proprement. Ce chapitre les
> donne avec des intuitions, des ordres de grandeur et les quatre formules qui
> reviennent partout (produit scalaire, softmax, entropie croisée, divergence KL).

## 1. Vecteurs : la géométrie qui fait tourner l'IA

Un **vecteur** est une liste ordonnée de nombres. Un embedding de 768 dimensions est
un vecteur de 768 nombres. Un poids de couche est un scalaire, une couche est une
matrice.

```
vecteur a = [0.2, -1.4, 3.0]        (3 dimensions)
matrice W = [[0.1, 0.5],            (2 lignes, 2 colonnes)
             [-0.3, 0.8]]
```

**Produit scalaire** : on multiplie terme à terme puis on additionne.

```
a . b = a1*b1 + a2*b2 + ... + an*bn
```

Interprétation : plus deux vecteurs pointent dans la même direction, plus le produit
scalaire est grand. C'est exactement ce que fait un neurone : il calcule le produit
scalaire entre ses entrées et ses poids, puis ajoute un biais.

**Norme** (longueur) :

```
||a|| = racine(a1^2 + a2^2 + ... + an^2)
```

**Similarité cosinus** : produit scalaire divise par le produit des normes. Le
résultat est dans [-1, 1] et ne dépend plus de la longueur des vecteurs.

```
cos(a, b) = (a . b) / (||a|| * ||b||)
```

C'est la mesure la plus utilisée pour comparer des embeddings : 0, 9 en cosinus
signifie « très proche », 0, 2 signifie « sans rapport ». Quantifier la longueur
(normaliser) permet de remplacer le cosinus par un simple produit scalaire, donc
d'utiliser des index optimises pour le produit scalaire. Voir
[chapitre 07](../07-rag/README.md).

## 2. Multiplier des matrices, c'est transformer un espace

Une multiplication de matrices ne fait pas qu'aligner des nombres : elle applique
une transformation géométrique à tout un ensemble de vecteurs à la fois. Une
rotation, un étirement, une projection.

```
y = W x + b

W de forme (sortie, entrée)
x de forme (entrée,)
y de forme (sortie,)
```

Empiler des transformations linéaires sans fonction d'activation ne sert à rien :
composer deux matrices donne une seule matrice. C'est pour cela que les non-linéarités
(ReLU, GELU) sont indispensables : elles rendent la composition réellement expressive.

Ordre de grandeur : la multiplication matricielle est le calcul dominant d'un
réseau de neurones, et c'est précisément celui que les GPU accélèrent. Une couche
de 4 096 x 4 096 sur un batch de 1 024 tokens représente environ 34 milliards
d'opérations : voilà pourquoi les GPU existent. Voir [chapitre 10](../10-infrastructure/README.md).

**Vecteurs propres et valeurs propres** : pour certaines matrices, il existe des
directions que la transformation ne fait qu'étirer sans les faire tourner. Ces
directions sont les vecteurs propres, et le facteur d'étirement la valeur propre.

**SVD (décomposition en valeurs singulières)** : toute matrice peut s'écrire comme
rotation x étirement x rotation. Utile pour deux choses très concrètes :
la compression de rang faible (base de LoRA, voir [chapitre 06](../06-fine-tuning/README.md))
et l'analyse en composantes principales (ACP/PCA), qui projette les données sur les
directions de plus grande variance pour les visualiser ou les compacter.

## 3. Dérivées et gradient : comment on apprend

Une **dérivée** mesure comment une sortie change quand une entrée change un tout
petit peu. Le **gradient** est la généralisation à plusieurs variables : le vecteur
des dérivées partielles.

```
f(x) = x^2          f'(x) = 2x
f(x, y) = x^2 + 3y  grad f = (2x, 3)
```

Le gradient pointe vers la direction de plus forte **augmentation**. Pour minimiser
une perte, on avance donc dans la direction opposée :

```
theta <- theta - lr * grad_theta(J)
```

| Symbole | Sens |
|---|---|
| theta | les paramètres (des millions à des milliards de nombres) |
| J | la fonction de coût (un seul nombre) |
| grad_theta(J) | comment J change si on bouge chaque paramètre |
| lr | le learning rate : la taille du pas |

**Règle de la chaîne** : pour une composition de fonctions, la dérivée se multiplie.

```
si y = g(f(x)) alors dy/dx = g'(f(x)) * f'(x)
```

C'est *toute* la rétropropagation : un réseau profond est une composition de fonctions,
et son gradient s'obtient en multipliant les dérivées locales couche après couche, en
partant de la fin. Voir [chapitre 03](../03-reseaux-de-neurones/README.md).

Un détail qui compte : multiplier beaucoup de nombres plus petits que 1 fait
tendre le produit vers zero (gradient qui disparaît), et multiplier des nombres
plus grands que 1 le fait exploser. D'où les normalisations, les connexions
residuelles et le clipping du gradient.

## 4. Probabilités : le langage de l'incertitude

| Notion | Définition | Exemple |
|---|---|---|
| Variable aléatoire | Un résultat dépendant du hasard | Le prochain token |
| Distribution | La loi qui donne la probabilité de chaque valeur | Uniforme, normale, catégorielle |
| Espérance E[X] | La moyenne théorique | Valeur moyenne d'un de |
| Variance | La dispersion autour de l'espérance | À quel point les résultats varient |
| Indépendance | P(A et B) = P(A) x P(B) | Deux des lances séparément |

**Théorème de Bayes** :

```
P(A | B) = P(B | A) * P(A) / P(B)
```

- P(A) : ce qu'on croyait avant (à priori).
- P(B | A) : la vraisemblance des données si A était vrai.
- P(B) : la probabilité des données observees.
- P(A | B) : ce qu'on croit après (à posteriori).

C'est la base du raisonnement probabiliste et des classifieurs bayésiens, mais
aussi de la lecture quotidienne des tests médicaux : un test très sensible sur une
maladie rare produit surtout des faux positifs.

**Vraisemblance et maximum de vraisemblance (MLE)** : la vraisemblance d'un
paramètre est la probabilité d'avoir observe les données sachant ce paramètre.
Maximiser la vraisemblance, c'est choisir le paramètre qui explique le mieux ce
qu'on a vu. Entraîner un modèle, c'est presque toujours résoudre un problème de ce
type.

## 5. Les trois formules qui reviennent partout

### Entropie

Mesure de l'incertitude d'une distribution, en bits ou en nats.

```
H(p) = - somme_i p_i * log(p_i)
```

- Distribution certaine (un seul cas a 1) : H = 0.
- Distribution uniforme sur n cas : H = log(n).

Un modèle de langue très sur du prochain token a une entropie faible.

### Entropie croisee

Compare deux distributions : la vraie distribution p et celle du modèle q.

```
H(p, q) = - somme_i p_i * log(q_i)
```

En classification, p vaut 1 sur la bonne classe et 0 ailleurs, donc la formule
se résumé a :

```
perte = - log( q[bonne_classe] )
```

Lecture concrète : si le modèle donne 0,9 de proba à la bonne réponse, la perte
vaut 0,105 ; s'il donne 0,1, la perte vaut 2,30. Le coût explose donc quand le
modèle est confiant et faux. C'est la fonction de coût par défaut de la
classification et de la génération de texte.

### Divergence KL

Mesure l'ecart entre deux distributions, mais de façon asymetrique :
`KL(p || q) != KL(q || p)`.

```
KL(p || q) = somme_i p_i * log(p_i / q_i)
```

Elle vaut 0 si les deux distributions sont identiques. Elle sert partout :

- **VAE** : pénaliser l'ecart de l'espace latent à une gaussienne standard.
- **Distillation** : le petit modèle apprend à coller aux probabilités du grand.
- **RLHF** : pénaliser la dérive du modèle aligne par rapport au modèle de départ,
  pour éviter qu'il ne s'éloigné complètement.

`H(p, q) = H(p) + KL(p || q)` : l'entropie croisee est l'entropie de la vérité plus
l'ecart entre les deux distributions. Minimiser l'entropie croisee revient donc a
minimiser la divergence KL quand p est fixe.

## 6. Softmax et logits

Le réseau produit des **logits** : des scores bruts, sans contrainte, par exemple
`[2.0, 1.0, 0.1]`. Le **softmax** les transforme en distribution de probabilités.

```
softmax(z)_i = exp(z_i) / somme_j exp(z_j)
```

```
z = [2.0, 1.0, 0.1]
exp(z) = [7.39, 2.72, 1.11]     somme = 11.22
softmax = [0.659, 0.242, 0.099]  (somme = 1)
```

Trois propriétés à retenir :

- Les différences de logits deviennent des rapports de probabilités : +1 de logit
  multiplie la probabilité par e (environ 2,72). D'où l'effet très fort de la
  température, qui divise les logits avant le softmax.
- Le max de probabilité sature : avec un ecart de 10 logits, la probabilité monte
  au-delà de 0,9999. C'est la saturation que l'on observe dans les runs
  d'entraînement adversarial.
- Numériquement, on calcule toujours `log_softmax` en soustrayant le maximum des
  logits pour éviter un dépassement de capacité.

## 7. Statistiques utiles au quotidien

| Outil | Usage |
|---|---|
| Moyenne et ecart-type | Résumer un jeu de mesures, typiquement une performance sur plusieurs graines |
| Échantillonnage | Estimer une quantité en n'observant qu'une partie (d'où les jeux de validation) |
| Intervalle de confiance | Donner une fourchette plutôt qu'un chiffre nu : 82 pour cent plus ou moins 3 |
| Test d'hypothèse | Décider si une différence est réelle ou due au hasard |
| Corrélation (r) | Association linéaire entre deux variables, entre -1 et 1 |

Le point à ne jamais oublier : **une différence de performance non accompagnée de
sa variabilité ne veut rien dire**. Si une recette donne 82,4 et une autre 78,3,
mais que l'ecart-type entre graines est de 3 points, la conclusion est fragile.
Répéter l'expérience avec plusieurs graines aléatoires est la seule réponse
honnête.

Et la phrase la plus utile du chapitre : **corrélation n'implique pas causalité**.
Deux variables peuvent bouger ensemble à cause d'une troisième, ou par pure
coincidence. Un modèle predit une association, jamais un mécanisme.

## 8. Pourquoi tout se calcule en log-espace

Les probabilités de longues séquences sont des produits de petits nombres.

```
P(phrase) = P(t1) * P(t2) * ... * P(t1000)
```

Avec des probabilités de l'ordre de 0, 01 chacune, le produit de mille termes
devient environ 10 puissance -2000 : un nombre irreprésentable en flottant standard
(le plus petit double normal est de l'ordre de 10 puissance -308).

On travaille donc avec des **log-probabilités** : le produit devient une somme, et
les très petits nombres deviennent de simples nombres négatifs modérés.

```
log(P(phrase)) = log P(t1) + log P(t2) + ... + log P(t1000)
```

Conséquences concrètes : la perte d'entraînement s'exprime en moyenne de log-proba
(perte en nats), la perplexité est son exponentielle, et toutes les bibliothèques
exposent `log_softmax` et `cross_entropy` plutôt que des produits de probabilités.

## Ce qu'il faut retenir

- Un neurone calcule un produit scalaire plus un biais : c'est de la géométrie, pas
  de la magie.
- Composer des transformations linéaires ne sert à rien sans non-linéarité ; les
  fonctions d'activation sont ce qui rend un réseau expressif.
- Le gradient indique la direction de plus forte augmentation ; on avance dans la
  direction opposée, avec un pas proportionnel au learning rate.
- La règle de la chaîne appliquée couche par couche, c'est exactement la
  rétropropagation.
- Entropie croisee et divergence KL sont les deux mesures d'ecart entre distributions
  qui apparaissent dans l'entraînement, la distillation et le RLHF.
- Le softmax transforme des logits en probabilités, et un ecart d'un logit multiplie
  la probabilité par environ 2,72.
- On calcule en log-espace parce que les produits de milliers de petites
  probabilités deviennent numériquement inutilisables.
- Corrélation n'est pas causalité, et une moyenne sans sa variance ne prouve rien.

## Erreurs fréquentes / idées reçues

- « Il faut etre bon en maths pour faire de l'IA » -> il faut etre à l'aise avec
  ces quelques objets ; le reste s'apprend en lisant le code et en experimentant.
- « Le gradient descend vers le minimum global » -> il descend vers un minimum
  local, ou un point selle. La surface de perte d'un réseau profond n'est pas convexe.
- « Le produit scalaire suffit » -> sans normalisation, il favorise les vecteurs
  longs ; d'où la normalisation des embeddings avant la recherche.
- « La divergence KL est une distance » -> non, elle est asymetrique et ne respecte
  pas l'inégalité triangulaire.
- « Une différence de 2 points est significative » -> sans répéter avec plusieurs
  graines, elle peut etre du bruit.

## Pour aller plus loin

- [Chapitre 01 : fondamentaux du machine learning](../01-fondamentaux/README.md)
- [Chapitre 03 : réseaux de neurones et rétropropagation](../03-reseaux-de-neurones/README.md)
- [Chapitre 04 : transformers et attention](../04-transformers/README.md)
- [Chapitre 07 : RAG, similarité et recherche vectorielle](../07-rag/README.md)
- Cours en ligne : « Mathematics for Machine Learning » (Deisenroth, Faisal, Ong),
  disponible gratuitement.
- Référence d'intuition : la série « Essence of Linear Algebra » de 3Blue1Brown.

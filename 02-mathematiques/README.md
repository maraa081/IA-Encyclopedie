# Les maths de l'IA, sans douleur

> On n'a pas besoin d'un bagage universitaire pour comprendre les reseaux de
> neurones : il faut cinq idees, expliquées une fois, proprement. Ce chapitre les
> donne avec des intuitions, des ordres de grandeur et les quatre formules qui
> reviennent partout (produit scalaire, softmax, entropie croisée, divergence KL).

## 1. Vecteurs : la geometrie qui fait tourner l'IA

Un **vecteur** est une liste ordonnee de nombres. Un embedding de 768 dimensions est
un vecteur de 768 nombres. Un poids de couche est un scalaire, une couche est une
matrice.

```
vecteur a = [0.2, -1.4, 3.0]        (3 dimensions)
matrice W = [[0.1, 0.5],            (2 lignes, 2 colonnes)
             [-0.3, 0.8]]
```

**Produit scalaire** : on multiplie terme a terme puis on additionne.

```
a . b = a1*b1 + a2*b2 + ... + an*bn
```

Interpretation : plus deux vecteurs pointent dans la meme direction, plus le produit
scalaire est grand. C'est exactement ce que fait un neurone : il calcule le produit
scalaire entre ses entrees et ses poids, puis ajoute un biais.

**Norme** (longueur) :

```
||a|| = racine(a1^2 + a2^2 + ... + an^2)
```

**Similarite cosinus** : produit scalaire divise par le produit des normes. Le
resultat est dans [-1, 1] et ne depend plus de la longueur des vecteurs.

```
cos(a, b) = (a . b) / (||a|| * ||b||)
```

C'est la mesure la plus utilisee pour comparer des embeddings : 0, 9 en cosinus
signifie « tres proche », 0, 2 signifie « sans rapport ». Quantifier la longueur
(normaliser) permet de remplacer le cosinus par un simple produit scalaire, donc
d'utiliser des index optimises pour le produit scalaire. Voir
[chapitre 07](../07-rag/README.md).

## 2. Multiplier des matrices, c'est transformer un espace

Une multiplication de matrices ne fait pas qu'aligner des nombres : elle applique
une transformation geometrique a tout un ensemble de vecteurs a la fois. Une
rotation, un etirement, une projection.

```
y = W x + b

W de forme (sortie, entree)
x de forme (entree,)
y de forme (sortie,)
```

Empiler des transformations lineaires sans fonction d'activation ne sert a rien :
composer deux matrices donne une seule matrice. C'est pour cela que les non-linearites
(ReLU, GELU) sont indispensables : elles rendent la composition reelement expressive.

Ordre de grandeur : la multiplication matricielle est le calcul dominant d'un
réseau de neurones, et c'est precisement celui que les GPU accelerent. Une couche
de 4 096 x 4 096 sur un batch de 1 024 tokens represente environ 34 milliards
d'operations : voila pourquoi les GPU existent. Voir [chapitre 10](../10-infrastructure/README.md).

**Vecteurs propres et valeurs propres** : pour certaines matrices, il existe des
directions que la transformation ne fait qu'etirer sans les faire tourner. Ces
directions sont les vecteurs propres, et le facteur d'etirement la valeur propre.

**SVD (decomposition en valeurs singulieres)** : toute matrice peut s'ecrire comme
rotation x etirement x rotation. Utile pour deux choses tres concretes :
la compression de rang faible (base de LoRA, voir [chapitre 06](../06-fine-tuning/README.md))
et l'analyse en composantes principales (ACP/PCA), qui projette les donnees sur les
directions de plus grande variance pour les visualiser ou les compacter.

## 3. Derivees et gradient : comment on apprend

Une **derivee** mesure comment une sortie change quand une entree change un tout
petit peu. Le **gradient** est la generalisation a plusieurs variables : le vecteur
des derivees partielles.

```
f(x) = x^2          f'(x) = 2x
f(x, y) = x^2 + 3y  grad f = (2x, 3)
```

Le gradient pointe vers la direction de plus forte **augmentation**. Pour minimiser
une perte, on avance donc dans la direction opposee :

```
theta <- theta - lr * grad_theta(J)
```

| Symbole | Sens |
|---|---|
| theta | les parametres (des millions a des milliards de nombres) |
| J | la fonction de cout (un seul nombre) |
| grad_theta(J) | comment J change si on bouge chaque parametre |
| lr | le learning rate : la taille du pas |

**Regle de la chaine** : pour une composition de fonctions, la derivee se multiplie.

```
si y = g(f(x)) alors dy/dx = g'(f(x)) * f'(x)
```

C'est *toute* la retropropagation : un reseau profond est une composition de fonctions,
et son gradient s'obtient en multipliant les derivees locales couche apres couche, en
partant de la fin. Voir [chapitre 03](../03-reseaux-de-neurones/README.md).

Un detail qui compte : multiplier beaucoup de nombres plus petits que 1 fait
tendre le produit vers zero (gradient qui disparait), et multiplier des nombres
plus grands que 1 le fait exploser. D'ou les normalisations, les connexions
residuelles et le clipping du gradient.

## 4. Probabilites : le langage de l'incertitude

| Notion | Definition | Exemple |
|---|---|---|
| Variable aleatoire | Un resultat dependant du hasard | Le prochain token |
| Distribution | La loi qui donne la probabilite de chaque valeur | Uniforme, normale, categoriale |
| Esperance E[X] | La moyenne theorique | Valeur moyenne d'un de |
| Variance | La dispersion autour de l'esperance | A quel point les resultats varient |
| Independance | P(A et B) = P(A) x P(B) | Deux des lances separement |

**Theoreme de Bayes** :

```
P(A | B) = P(B | A) * P(A) / P(B)
```

- P(A) : ce qu'on croyait avant (a priori).
- P(B | A) : la vraisemblance des donnees si A etait vrai.
- P(B) : la probabilite des donnees observees.
- P(A | B) : ce qu'on croit apres (a posteriori).

C'est la base du raisonnement probabiliste et des classifieurs bayesiens, mais
aussi de la lecture quotidienne des tests medicaux : un test tres sensible sur une
maladie rare produit surtout des faux positifs.

**Vraisemblance et maximum de vraisemblance (MLE)** : la vraisemblance d'un
parametre est la probabilite d'avoir observe les donnees sachant ce parametre.
Maximiser la vraisemblance, c'est choisir le parametre qui explique le mieux ce
qu'on a vu. Entrainer un modele, c'est presque toujours resoudre un probleme de ce
type.

## 5. Les trois formules qui reviennent partout

### Entropie

Mesure de l'incertitude d'une distribution, en bits ou en nats.

```
H(p) = - somme_i p_i * log(p_i)
```

- Distribution certaine (un seul cas a 1) : H = 0.
- Distribution uniforme sur n cas : H = log(n).

Un modele de langue tres sur du prochain token a une entropie faible.

### Entropie croisee

Compare deux distributions : la vraie distribution p et celle du modele q.

```
H(p, q) = - somme_i p_i * log(q_i)
```

En classification, p vaut 1 sur la bonne classe et 0 ailleurs, donc la formule
se resume a :

```
perte = - log( q[bonne_classe] )
```

Lecture concrete : si le modele donne 0,9 de proba a la bonne reponse, la perte
vaut 0,105 ; s'il donne 0,1, la perte vaut 2,30. Le cout explose donc quand le
modele est confiant et faux. C'est la fonction de cout par defaut de la
classification et de la generation de texte.

### Divergence KL

Mesure l'ecart entre deux distributions, mais de façon asymetrique :
`KL(p || q) != KL(q || p)`.

```
KL(p || q) = somme_i p_i * log(p_i / q_i)
```

Elle vaut 0 si les deux distributions sont identiques. Elle sert partout :

- **VAE** : penaliser l'ecart de l'espace latent a une gaussienne standard.
- **Distillation** : le petit modele apprend a coller aux probabilites du grand.
- **RLHF** : penaliser la derive du modele aligne par rapport au modele de depart,
  pour eviter qu'il ne s'eloigne completement.

`H(p, q) = H(p) + KL(p || q)` : l'entropie croisee est l'entropie de la verite plus
l'ecart entre les deux distributions. Minimiser l'entropie croisee revient donc a
minimiser la divergence KL quand p est fixe.

## 6. Softmax et logits

Le reseau produit des **logits** : des scores bruts, sans contrainte, par exemple
`[2.0, 1.0, 0.1]`. Le **softmax** les transforme en distribution de probabilites.

```
softmax(z)_i = exp(z_i) / somme_j exp(z_j)
```

```
z = [2.0, 1.0, 0.1]
exp(z) = [7.39, 2.72, 1.11]     somme = 11.22
softmax = [0.659, 0.242, 0.099]  (somme = 1)
```

Trois proprietes a retenir :

- Les differences de logits deviennent des rapports de probabilites : +1 de logit
  multiplie la probabilite par e (environ 2,72). D'ou l'effet tres fort de la
  temperature, qui divise les logits avant le softmax.
- Le max de probabilite sature : avec un ecart de 10 logits, la probabilite monte
  au-dela de 0,9999. C'est la saturation que l'on observe dans les runs
  d'entrainement adversarial.
- Numeriquement, on calcule toujours `log_softmax` en soustrayant le maximum des
  logits pour eviter un depassement de capacite.

## 7. Statistiques utiles au quotidien

| Outil | Usage |
|---|---|
| Moyenne et ecart-type | Resumer un jeu de mesures, typiquement une performance sur plusieurs graines |
| Echantillonnage | Estimer une quantite en n'observant qu'une partie (d'ou les jeux de validation) |
| Intervalle de confiance | Donner une fourchette plutot qu'un chiffre nu : 82 pour cent plus ou moins 3 |
| Test d'hypothese | Decider si une difference est reelle ou due au hasard |
| Correlation (r) | Association lineaire entre deux variables, entre -1 et 1 |

Le point a ne jamais oublier : **une difference de performance non accompagnee de
sa variabilite ne veut rien dire**. Si une recette donne 82,4 et une autre 78,3,
mais que l'ecart-type entre graines est de 3 points, la conclusion est fragile.
Repeter l'experience avec plusieurs graines aleatoires est la seule reponse
honnete.

Et la phrase la plus utile du chapitre : **correlation n'implique pas causalite**.
Deux variables peuvent bouger ensemble a cause d'une troisieme, ou par pure
coincidence. Un modele predit une association, jamais un mecanisme.

## 8. Pourquoi tout se calcule en log-espace

Les probabilites de longues sequences sont des produits de petits nombres.

```
P(phrase) = P(t1) * P(t2) * ... * P(t1000)
```

Avec des probabilites de l'ordre de 0, 01 chacune, le produit de mille termes
devient environ 10 puissance -2000 : un nombre inrepresentable en flottant standard
(le plus petit double normal est de l'ordre de 10 puissance -308).

On travaille donc avec des **log-probabilites** : le produit devient une somme, et
les tres petits nombres deviennent de simples nombres negatifs moderes.

```
log(P(phrase)) = log P(t1) + log P(t2) + ... + log P(t1000)
```

Consequences concretes : la perte d'entrainement s'exprime en moyenne de log-proba
(perte en nats), la perplexite est son exponentielle, et toutes les bibliotheques
exposent `log_softmax` et `cross_entropy` plutot que des produits de probabilites.

## Ce qu'il faut retenir

- Un neurone calcule un produit scalaire plus un biais : c'est de la geometrie, pas
  de la magie.
- Composer des transformations lineaires ne sert a rien sans non-linearite ; les
  fonctions d'activation sont ce qui rend un reseau expressif.
- Le gradient indique la direction de plus forte augmentation ; on avance dans la
  direction opposee, avec un pas proportionnel au learning rate.
- La regle de la chaine appliquee couche par couche, c'est exactement la
  retropropagation.
- Entropie croisee et divergence KL sont les deux mesures d'ecart entre distributions
  qui apparaissent dans l'entrainement, la distillation et le RLHF.
- Le softmax transforme des logits en probabilites, et un ecart d'un logit multiplie
  la probabilite par environ 2,72.
- On calcule en log-espace parce que les produits de milliers de petites
  probabilites deviennent numeriquement inutilisables.
- Correlation n'est pas causalite, et une moyenne sans sa variance ne prouve rien.

## Erreurs frequentes / idees recues

- « Il faut etre bon en maths pour faire de l'IA » -> il faut etre a l'aise avec
  ces quelques objets ; le reste s'apprend en lisant le code et en experimentant.
- « Le gradient descend vers le minimum global » -> il descend vers un minimum
  local, ou un point selle. La surface de perte d'un reseau profond n'est pas convexe.
- « Le produit scalaire suffit » -> sans normalisation, il favorise les vecteurs
  longs ; d'ou la normalisation des embeddings avant la recherche.
- « La divergence KL est une distance » -> non, elle est asymetrique et ne respecte
  pas l'inegalite triangulaire.
- « Une difference de 2 points est significative » -> sans repeter avec plusieurs
  graines, elle peut etre du bruit.

## Pour aller plus loin

- [Chapitre 01 : fondamentaux du machine learning](../01-fondamentaux/README.md)
- [Chapitre 03 : reseaux de neurones et retropropagation](../03-reseaux-de-neurones/README.md)
- [Chapitre 04 : transformers et attention](../04-transformers/README.md)
- [Chapitre 07 : RAG, similarite et recherche vectorielle](../07-rag/README.md)
- Cours en ligne : « Mathematics for Machine Learning » (Deisenroth, Faisal, Ong),
  disponible gratuitement.
- Reference d'intuition : la serie « Essence of Linear Algebra » de 3Blue1Brown.

# Transformers

> Ce chapitre explique l'architecture Transformer, introduite en 2017, qui est la base
> de tous les grands modèles de langage actuels (GPT, Claude, Llama, Mistral, Qwen,
> DeepSeek). On y démonte le mécanisme d'attention, la tokenisation, l'encodage
> positionnel et les variantes modernes, sans supposer que tu connais déjà le paysage.

Avant ce chapitre, il est utile d'avoir lu [le chapitre 03](../03-reseaux-de-neurones/README.md)
(réseaux de neurones, rétropropagation, normalisation) et d'avoir les bases d'algèbre
linéaire du [chapitre 02](../02-mathematiques/README.md). L'idée centrale de tout ce
chapitre : plutôt que de lire une phrase mot après mot, on regarde tous les mots en
même temps et on laisse le modèle décider quels mots s'influencent.

## 1. Le problème avant les transformers

Les modèles précédents pour le texte étaient les RNN, LSTM et GRU (voir
[chapitre 03](../03-reseaux-de-neurones/README.md), section 11). Ils traitent la
séquence **élément par élément**, en transportant un état caché de mot en mot.

Deux limites structurelles en découlent :

1. **Séquentialité** : impossible de calculer le mot 10 000 avant d'avoir fini les
   9 999 précédents. Sur un GPU conçu pour faire des milliers d'opérations en parallèle,
   c'est un gaspillage massif. L'entraînement est lent.
2. **Longue portée** : l'information doit se propager à travers chaque pas de temps.
   Même les LSTM oublient le début d'un long texte : le signal s'atténue.

Ordre de grandeur : un LSTM sur une phrase de 100 mots fait 100 étapes séquentielles.
Un transformer, lui, traite les 100 mots en une seule opération matricielle parallèle.
C'est cette parallélisation qui a rendu possible l'entraînement de modèles à des
milliards de paramètres.

## 2. Attention Query / Key / Value

L'**attention** (attention) est un mécanisme qui calcule une moyenne pondérée des
informations disponibles, en fonction de leur pertinence pour la requête courante.

L'analogie classique : on cherche une réponse dans une bibliothèque.
- La **Query** (requête) est ce que tu cherches.
- Les **Keys** (clés) sont les étiquettes des livres, que tu compares à ta requête.
- Les **Values** (valeurs) sont le contenu des livres.

Pour chaque token, on calcule sa Query, sa Key et sa Value par projection linéaire de
son embedding. Les matrices de projection `W_Q`, `W_K`, `W_V` sont apprises.

La **formule d'attention scalaire** (scaled dot-product attention) est :

```
Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
```

Décomposée terme à terme :
- `Q * K^T` : produit scalaire entre chaque requête et chaque clé. Grand quand la
  requête et la clé sont similaires. Résultat : une matrice de scores de taille
  `n x n` pour `n` tokens.
- `/ sqrt(d_k)` : **mise à l'échelle** par la racine de la dimension `d_k` des clés.
  Sans ce facteur, les produits scalaires deviennent très grands quand `d_k` est grand,
  ce qui pousse le softmax vers des valeurs extrêmes où le gradient s'éteint. Le
  facteur garde les scores dans une plage saine.
- `softmax(...)` : transforme chaque ligne de scores en poids positifs qui somment à 1.
- `* V` : pondère les valeurs par ces poids et les additionne. Chaque token sort avec
  un mélange des valeurs des tokens qu'il juge pertinents.

### Multi-head

Une seule attention ne capture qu'un type de relation. La **multi-head attention**
(MHA) répète le calcul en parallèle sur `h` sous-espaces (par ex. 8, 16 ou 32 têtes),
chacun avec ses propres `W_Q`, `W_K`, `W_V`. On concatène les sorties et on les
projette. Une tête peut suivre la syntaxe, une autre les coréférences, une autre la
sémantique. C'est le même principe qu'un ensemble de modèles : plusieurs vues valent
mieux qu'une seule.

## 3. Self-attention et cross-attention

- **Self-attention** : Query, Key et Value viennent de la **même** séquence. Chaque
  token regarde tous les autres tokens de sa propre phrase. C'est le mécanisme
  principal des modèles de langage.
- **Cross-attention** : les Query viennent d'une séquence et les Key/Value d'une
  **autre**. Par exemple, en traduction, le décodeur (qui génère la phrase cible)
  interroge l'encodeur (qui a lu la phrase source). Utilisée dans les architectures
  encodeur-décodeur.

## 4. Masques : causal et padding

Un masque est une modification des scores d'attention avant le softmax, pour interdire
certaines positions.

**Masque causal** : essentiel pour la génération de texte. Quand le modèle prédit le
token `t`, il ne doit pas voir les tokens `t+1`, `t+2`, etc. (sinon il tricherait).
On met les scores des positions futures à `-infini`, ce qui les rend nuls après le
softmax. La matrice de masque est triangulaire.

```
scores (avant softmax), masque causal :
   t1  t2  t3
t1  .   -inf -inf
t2  .   .    -inf
t3  .   .    .
```

**Masque de padding** : dans un lot, les phrases n'ont pas toutes la même longueur.
On ajoute des tokens de remplissage (`<pad>`) jusqu'à la longueur maximale. Le masque
de padding empêche le modèle de prêter attention à ces tokens inutiles, sinon ils
polluent le calcul.

## 5. Encodeur, décodeur, et les trois grandes familles

L'architecture d'origine (Vaswani et al., 2017) comporte un encodeur et un décodeur.
Aujourd'hui, on distingue trois familles selon ce qu'on garde.

| Famille | Exemples | Attention | Idéal pour |
|---|---|---|---|
| Encodeur seul | BERT, RoBERTa, DeBERTa, E5 | bidirectionnelle | classification, extraction, embeddings |
| Décodeur seul | GPT-4o, Claude, Llama 3.1, Mistral, Qwen 2.5, DeepSeek | causale | génération, chat, raisonnement |
| Encodeur-décodeur | T5, BART, flan-T5 | bidir. + causale + cross | traduction, résumé, seq2seq |

- **Encodeur seul** : chaque token voit toute la phrase (attention bidirectionnelle).
  On l'entraîne souvent par masquage (prédire des mots cachés). Excellent pour la
  compréhension, incapable de générer nativement du texte libre.
- **Décodeur seul** : attention causale, un token à la fois. C'est ce qu'on entraîne
  à prédire le token suivant (voir [chapitre 05](../05-llm/README.md)). C'est la
  famille dominante aujourd'hui.
- **Encodeur-décodeur** : l'encodeur lit la source, le décodeur génère la cible en
  s'appuyant sur la cross-attention. Adapté aux tâches où l'entrée et la sortie sont
  deux séquences distinctes.

## 6. Encodage positionnel

Le mécanisme d'attention ne connaît pas l'ordre des tokens : mélanger la phrase ne
change rien au calcul. Il faut donc injecter l'information de position.

**Sinusoïdal** (original) : on ajoute à chaque embedding des valeurs de sinus et
cosinus à des fréquences différentes. Déterministe, ne dépend pas de la longueur.

**Appris** : on traite la position comme un embedding classique appris (BERT fait cela).
Limité à la longueur maximale vue à l'entraînement.

**RoPE** (Rotary Positional Embeddings) : on applique une rotation aux vecteurs Query
et Key en fonction de leur position, ce qui encode les positions relatives. Devenu le
standard des modèles modernes (Llama, Mistral, Qwen). Il brasse mieux les longues
séquences.

**ALiBi** (Attention with Linear Biases) : on pénalise linéairement les scores
d'attention en fonction de la distance entre tokens, sans encodage explicite. Permet
une bonne extrapolation à des longueurs non vues.

Arbitrage : RoPE est le choix par défaut aujourd'hui ; l'extension de fenêtre de
contexte (voir section 9) se fait en ajustant RoPE (scaling des fréquences), avec
parfois un peu de réentraînement.

## 7. Tokenisation

Un modèle ne lit pas des caractères : il lit des **tokens** (unités discrètes) et
travaille avec leurs identifiants numériques. La **tokenisation** est donc la toute
première étape.

- **BPE** (Byte Pair Encoding) : part des caractères et fusionne itérativement les
  paires les plus fréquentes. Les mots fréquents deviennent un token, les rares sont
  découpés en sous-mots. Algorithme de base de GPT.
- **WordPiece** : variante de BPE avec un critère probabiliste, utilisée par BERT.
- **SentencePiece** : traite l'espace comme un caractère normal, donc indépendant de
  la langue (utile pour les langues sans espaces, comme le chinois ou le japonais).
  Utilisé par T5, Llama.
- **Byte-level** : tokenise au niveau des octets, ce qui couvre n'importe quel texte
  sans symbole « inconnu » (GPT-2 et suivants).

**Tokens spéciaux** : on ajoute des symboles réservés comme `<|endoftext|>` (fin de
document), `<|im_start|>` (rôle de message), ou `[MASK]` (position à prédire).

**Vocabulaire** : ensemble de tous les tokens connus. Ordre de grandeur : de 30 000
à 50 000 pour les anciens modèles ; les modèles multilingues modernes montent souvent
à 100 000 à 150 000 tokens. Un vocabulaire plus grand réduit le nombre de tokens par
phrase (donc le coût), mais augmente la taille de la couche d'embedding.

Point pratique : en français, la tokenisation BPE est moins efficace qu'en anglais
(plus de tokens par mot), ce qui rend les modèles plus chers à l'usage sur notre
langue. Les gros vocabulaires modernes corrigent partiellement ce déséquilibre.

## 8. Embeddings

Un **embedding** est une représentation vectorielle dense d'un token. La couche
d'embedding est une grande table : on lit la ligne correspondant à l'identifiant du
token.

- **Embedding de mot** : un vecteur fixe par mot (approche historique, type word2vec).
  Simple mais ne distingue pas les sens multiples (« avocat » juriste et « avocat »
  fruit).
- **Embedding de sous-mot** : un mot rare ou inconnu est découpé en sous-tokens, chacun
  avec son vecteur. Plus robuste aux mots nouveaux.
- **Embedding contextuel** : la représentation finale d'un token dépend du contexte
  où il apparaît (grâce à l'attention). « Avocat » dans une phrase juridique et dans
  une phrase culinaire donnent des vecteurs différents. C'est la force des transformers.

**Dimension** : typiquement de 768 (petits modèles) à 4096 ou 8192 (gros modèles),
voire plus, dépend de la largeur `d_model`.

**Similarité** : deux embeddings proches dans l'espace vectoriel ont un sens proche.
On mesure cela par la similarité cosinus (voir [chapitre 07](../07-rag/README.md)
pour l'usage en recherche d'information).

## 9. Les briques d'une couche Transformer

Un bloc transformer empile toujours les mêmes composants :

```
x
 |--> [ LayerNorm ] --> [ Multi-Head Attention ] --> + (résidu)
 |                                                   |
 +---------------------------------------------------+
 |
 |--> [ LayerNorm ] --> [ Feed-Forward FFN ] --> + (résidu)
 |                                               |
 +-----------------------------------------------+
 |
 v  (bloc suivant)
```

- **FFN** (Feed-Forward Network) : deux couches linéaires avec une activation au
  milieu (ReLU, GELU, ou SwiGLU). Elle s'applique indépendamment à chaque position et
  mélange les canaux de représentation. Sa dimension interne est souvent 4 fois plus
  grande que `d_model`.
- **Résidus** : chaque sous-bloc ajoute son entrée à sa sortie, ce qui laisse filer le
  gradient (voir [chapitre 03](../03-reseaux-de-neurones/README.md)).
- **Placement du LayerNorm** : deux choix. En **Post-LN** (normalisation après chaque
  sous-bloc), c'est l'architecture d'origine mais l'entraînement est instable sur les
  gros modèles. En **Pre-LN** (normalisation avant chaque sous-bloc), c'est plus stable
  et c'est devenu le standard. RMSNorm remplace souvent LayerNorm.

Un modèle moderne empile des dizaines de ces blocs (typiquement de 12 pour un petit
modèle à plus de 100 pour les très gros), plus une couche d'embedding au début et une
tête de sortie à la fin.

## 10. Complexité O(n^2), fenêtre de contexte, FlashAttention

La matrice de scores d'attention a une taille `n x n` pour une séquence de `n` tokens.
La complexité en temps et en **mémoire** est donc quadratique : `O(n^2)`.

Conséquences concrètes :
- Doubler la longueur du contexte multiplie par 4 la mémoire d'attention (et l'usage
  du cache à l'inférence, voir [chapitre 09](../09-inference-optimisation/README.md)).
- Une fenêtre de contexte de 128 000 tokens (ordre de grandeur des modèles longs
  actuels) génère une matrice d'attention énorme ; c'est souvent la VRAM qui limite,
  pas le nombre de paramètres.

**Fenêtre de contexte** : nombre maximum de tokens que le modèle peut traiter d'un coup.
Ordre de grandeur : 4 000 tokens (anciens modèles), 32 000 à 128 000 (modèles
récents), parfois 1 million annoncé. Dépasser la fenêtre demande de découper ou de
compresser le contexte.

**FlashAttention** (Dao et al., 2022) : au lieu de matérialiser toute la matrice
d'attention en mémoire, on la calcule par blocs et on recalcule certaines valeurs à
la volée plutôt que de les stocker. Résultat : beaucoup moins de mémoire et un calcul
plus rapide, sans changer le résultat mathématique. C'est aujourd'hui utilisé partout.

Autres pistes : **attention sparse** (ne calculer que certains couples
requête-clé au lieu de tous), **sliding window** (chaque token ne regarde que ses
voisins proches, comme dans Mistral), **linéarisation** de l'attention.

## 11. Variantes modernes

Les architectures récentes modifient le transformer standard sur plusieurs axes.

**MoE** (Mixture of Experts) : au lieu d'une seule grande FFN par bloc, on en place
plusieurs (les « experts »), et un routeur choisit quelques experts à activer pour
chaque token. Conséquence : un modèle peut avoir beaucoup plus de paramètres sans
augmenter proportionnellement le calcul par token. Exemples : Mixtral, DeepSeek-V3,
certains modèles Qwen. Contrepartie : plus complexe à entraîner et à servir, et la
mémoire totale reste élevée.

**GQA / MQA** (Grouped-Query / Multi-Query Attention) : dans la multi-head attention
classique, chaque tête a ses propres Key et Value, ce qui gonfle la mémoire du cache
KV à l'inférence. GQA fait partager Key et Value par groupes de têtes ; MQA les fait
partager par toutes les têtes. Gain de mémoire et de vitesse à l'inférence, avec une
perte de qualité faible. Standard dans les modèles récents (Llama, Mistral).

**Mamba / SSM** (State Space Models) : famille alternative de modèles séquentiels dont
le coût est linéaire en la longueur (`O(n)`) au lieu de quadratique. Ils concurrencent
les transformers sur certaines tâches de long contexte, mais ne les ont pas remplacés.
Voir [chapitre 17](../17-au-dela/README.md).

**Attention sparse / windowed** et **hybrides** mélangeant attention locale et globale :
cherchent à réduire le coût des très longs contextes.

Point d'honnêteté : le domaine bouge vite et les compromis entre ces variantes ne sont
pas tranchés définitivement. Le transformer dense reste la référence.

## 12. Récapitulatif d'un modèle moderne (type Llama / DeepSeek)

Pour fixer les idées, voici les briques typiques d'un grand modèle de langage décodeur
seul récent :

| Composant | Choix courant |
|---|---|
| Type | décodeur seul, attention causale |
| Normalisation | RMSNorm en Pre-LN |
| Activation FFN | SwiGLU |
| Position | RoPE (+ scaling pour l'extension de contexte) |
| Attention | GQA (grouped-query) |
| Attention efficace | FlashAttention |
| Architecture conditionnelle | parfois MoE (routeur top-k) |
| Tokenisation | BPE ou SentencePiece, vocabulaire 100k+ |
| Tête de sortie | projection vers le vocabulaire, softmax |

On retrouve ces briques chez Meta (Llama), Mistral, Alibaba (Qwen) et DeepSeek, avec
des variations. La suite logique de ce chapitre est l'entraînement de ces modèles :
voir [chapitre 05 : LLM](../05-llm/README.md).

## 13. Ce qu'il faut retenir

- Le transformer remplace la récurrence par l'attention, ce qui permet de traiter
  toute une séquence en parallèle.
- L'attention Query/Key/Value calcule une moyenne pondérée des valeurs selon la
  similarité requête-clé, mise à l'échelle par la racine de `d_k` pour stabiliser
  le softmax.
- La multi-head attention apprend plusieurs types de relations simultanément.
- La self-attention relie les tokens d'une même séquence ; la cross-attention relie
  deux séquences.
- Le masque causal empêche de voir le futur ; le masque de padding ignore le remplissage.
- Il existe trois familles : encodeur seul (BERT), décodeur seul (GPT, Llama),
  encodeur-décodeur (T5), chacune adaptée à des tâches différentes.
- L'encodage positionnel est indispensable ; RoPE est le standard moderne.
- La tokenisation découpe le texte en sous-mots, ce qui gère les mots rares et le
  multilingue ; le vocabulaire et la qualité de la tokenisation influencent directement
  le coût et la performance en français.
- La complexité de l'attention est quadratique en la longueur ; FlashAttention et les
  variantes comme GQA réduisent fortement le coût mémoire.
- Le MoE permet de faire grossir les modèles sans augmenter linéairement le calcul par
  token.

## 14. Erreurs fréquentes / idées reçues

- "Le transformer a supprimé le besoin de données et de calcul" -> au contraire, il
  en faut énormément pour aboutir ; c'est son coût, pas son avantage.
- "L'attention comprend la grammaire comme un humain" -> c'est une pondération
  statistique, sans compréhension ni intention.
- "Le transformer est toujours supérieur au RNN" -> sur de très longues séquences ou
  avec des contraintes de latence, les SSM comme Mamba sont compétitifs.
- "Toutes les architectures transformer utilisent un masque causal" -> c'est spécifique
  aux décodeurs ; les encodeurs comme BERT regardent toute la séquence.
- "La fenêtre de contexte est illimitée" -> elle est bornée par la mémoire et la
  complexité quadratique ; annoncer un million de tokens ne veut pas dire que la
  qualité suit sur cette longueur.
- "Un grand vocabulaire est toujours mieux" -> il réduit le nombre de tokens mais
  alourdit l'embedding et la couche de sortie.
- "Le token est équivalent à un mot" -> non, un mot peut être découpé en plusieurs
  tokens, surtout en français ou dans les langues peu dotées.
- "La mise à l'échelle par racine(d_k) est un détail" -> sans elle, l'entraînement
  devient instable dès que `d_k` est grand.

## 15. Pour aller plus loin

Dans l'encyclopédie :
- [Chapitre 03 : réseaux de neurones](../03-reseaux-de-neurones/README.md), pour la
  rétropropagation, la normalisation et les résidus dont dépend le transformer.
- [Chapitre 05 : LLM](../05-llm/README.md), pour le pré-entraînement, les scaling laws
  et le post-training de ces architectures.
- [Chapitre 09 : optimisation d'inférence](../09-inference-optimisation/README.md),
  pour le cache KV et les techniques de service rapide des transformers.
- [Chapitre 10 : infrastructure](../10-infrastructure/README.md), pour la VRAM, les
  GPU et le parallélisme nécessaires à l'entraînement.

Ressources externes réelles :
- Article "Attention Is All You Need", Vaswani et al., 2017 (NIPS).
- Article "BERT: Pre-training of Deep Bidirectional Transformers", Devlin et al., 2018.
- Article "FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness", Dao et al., 2022.
- Blog "The Illustrated Transformer", Jay Alammar (explication visuelle de référence).
- Cours CS224n "Natural Language Processing with Deep Learning", Stanford.

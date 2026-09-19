# Grands modèles de langage (LLM)

> Ce chapitre explique ce qu'est un modèle de langage de grande taille (LLM), comment
> il est entraîné, comment il génère du texte, et comment on l'évalue. C'est le coeur
> technique de l'IA générative textuelle : comprendre ce chapitre conditionne tout le
> reste (RAG, agents, fine-tuning, optimisation d'inférence).

## 1. Qu'est-ce qu'un LLM : la prédiction du prochain token

Un LLM (Large Language Model, grand modèle de langage) est un réseau de neurones
profond, presque toujours de type Transformer, dont l'unique tâche d'entraînement
est de prédire le **prochain token** d'une séquence.

> Token : unité élémentaire de texte manipulée par le modèle. Ce n'est pas un mot mais
> un morceau de mot (sous-mot). Le découpage est fait par un algorithme de tokenisation,
> typiquement BPE (Byte Pair Encoding, encodage par paires d'octets), détaillé au
> [chapitre 04](../04-transformers/README.md).

Le modèle ne voit jamais "des mots" : il voit des identifiants numériques. Il calcule
une distribution de probabilité sur tout son vocabulaire (typiquement 32 000 à 200 000
tokens) et échantillonne le suivant. On répète l'opération, token après token : c'est
la génération **autorégressive**.

```
Entrée : "La capitale de la France est"
Sortie du modèle (probabilités simplifiées) :
  " Paris"   0.71
  " une"     0.06
  " située"  0.04
  ... (le reste du vocabulaire)
On tire " Paris", on l'ajoute à l'entrée, et on recommence.
```

Formellement, un LLM modélise P(t_n | t_1, ..., t_{n-1}) : la probabilité du token
suivant conditionnée par tous les tokens précédents. L'entraînement minimise la
**perte d'entropie croisée** (cross-entropy loss) entre la distribution prédite et le
token réellement présent dans le corpus. Autrement dit : on punit le modèle quand il
n'accorde pas assez de probabilité au mot qui suivait vraiment.

Point crucial : un LLM de base n'a **pas d'objectif de vérité**. Il optimise la
plausibilité statistique, pas la factualité. C'est l'origine directe des hallucinations
(section 11).

## 2. Le décodage : comment on choisit le token

Les probabilités de sortie ne sont pas directement le texte : il faut une stratégie de
**décodage** (decoding) pour transformer ces probabilités en choix concrets. C'est
ici qu'interviennent les paramètres que tu vois dans toutes les API.

| Paramètre | Ce qu'il fait | Effet concret |
|---|---|---|
| Greedy | Prend toujours le token le plus probable | Déterministe, répétitif, adapté au code |
| Temperature | Divise les logits par T avant softmax | T < 1 : plus sage ; T > 1 : plus créatif/chaotique |
| Top-k | Ne garde que les k tokens les plus probables | Élimine la queue improbable du vocabulaire |
| Top-p (nucleus) | Garde le plus petit ensemble dont la somme des probs >= p | S'adapte : peu de candidats si le modèle est sûr |
| Repetition penalty | Pénalise les tokens déjà générés | Réduit les boucles, risque de dégrader la cohérence |
| Beam search | Explore plusieurs hypothèses en parallèle | Séquences plus probables globalement, coût plus élevé |

Relation entre température et softmax, terme à terme :

```
logits  : scores bruts sortis par la dernière couche (non normalisés)
T       : temperature
softmax : normalise en probabilités qui somment à 1

P(token_i) = exp(logit_i / T) / somme_j exp(logit_j / T)
```

Quand T tend vers 0, on converge vers le greedy (argmax). Quand T vaut 1, on utilise la
distribution brute du modèle. Au-delà de 1, on "aplatit" la distribution et on rend
possible des tokens peu probables.

Le **beam search** maintient plusieurs séquences partielles (les beams) et choisit à la
fin celle de probabilité globale maximale. C'est utile en traduction mais souvent
décevant en conversation : les sorties sont plates et peu naturelles. Gemini, GPT-4o,
Claude ou Mistral utilisent par défaut un échantillonnage type top-p avec température
modérée.

```python
# Illustration du décodage top-p, sans dépendance externe
import numpy as np

def sample_next(logits, temperature=0.8, top_p=0.9):
    logits = np.asarray(logits) / temperature
    logits -= logits.max()               # stabilité numérique
    probs = np.exp(logits)
    probs /= probs.sum()
    order = np.argsort(probs)[::-1]      # probas décroissantes
    cum = np.cumsum(probs[order])
    k = np.searchsorted(cum, top_p) + 1  # plus petit ensemble >= top_p
    keep = order[:k]
    keep_p = probs[keep] / probs[keep].sum()
    return np.random.choice(keep, p=keep_p)
```

## 3. Tailles de modèles et architectures Mixture of Experts

La "taille" d'un modèle se mesure en nombre de paramètres (les poids appris). L'ordre
de grandeur détermine le matériel nécessaire et les capacités.

| Classe | Exemples | Paramètres | VRAM typique (4-bit) |
|---|---|---|---|
| Petit | Llama 3.2 3B, Qwen 2.5 1.5B | 1 à 4B | 1 à 3 Go |
| Moyen | Llama 3.1 8B, Mistral 7B | 7 à 14B | 5 à 9 Go |
| Grand | Qwen 2.5 72B, Llama 3.1 70B | 30 à 80B | 20 à 45 Go |
| Très grand | DeepSeek-V3, Llama 3.1 405B | 100 à 700B+ | multi-GPU |
| Propriétaire | GPT-4o, Claude Sonnet, Gemini | non publié | via API |

L'augmentation de la taille suit des **scaling laws** (lois d'échelle, section 5) : plus
de paramètres, plus de données, plus de calcul améliorent la performance de manière
prévisible. Mais le coût d'inférence croît aussi.

### Mixture of Experts (MoE)

> MoE (Mixture of Experts) : architecture où le réseau contient plusieurs "experts"
> (sous-réseaux) et un routeur qui, pour chaque token, n'active que quelques experts.
> Le modèle a beaucoup de paramètres au total mais n'en utilise qu'une fraction par token.

Mixtral 8x7B en est l'exemple canonique : environ 47 milliards de paramètres au total,
mais seulement environ 13 milliards activés par token (2 experts sur 8). On obtient donc
la capacité d'un grand modèle avec le coût de calcul d'un modèle moyen. DeepSeek-V3
pousse le principe très loin avec une architecture MoE à grande échelle.

Compromis : la mémoire (VRAM) doit contenir tous les experts, même inactifs. On gagne
du calcul, pas de la RAM.

## 4. Données de pré-entraînement

Un LLM est le reflet de ses données. Le pipeline typique :

1. **Crawl web** : récupération massive de pages (Common Crawl en est la source de
   base, des variantes filtrées existent comme C4 ou FineWeb).
2. **Filtrage** : suppression du spam, de la pornographie, des pages vides, des
   boilerplates (menus, cookies), détection de qualité par heuristiques.
3. **Déduplication** : retirer les textes répétés. Un document présent 50 fois sera
   sur-appris et dégradera la généralisation. La déduplication se fait par exact match
   ou par quasi-duplication (MinHash, LSH).
4. **Mélanges (mixing)** : pondération des sources. Typiquement beaucoup de web,
   beaucoup de code, plus du Wikipedia et des livres. Le ratio est un choix
   d'ingénierie influent : trop de code peut nuire au style naturel, trop peu nuit
   au raisonnement.
5. **Multilingue** : inclusion de langues multiples. L'anglais domine souvent, ce qui
   crée un déséquilibre de performance (le français est généralement bien servi par les
   modèles "multilingues", mais des langues rares le sont beaucoup moins).
6. **Code** : très valorisé car il encode des raisonnements structurés (logique,
   dépendances, mathématiques implicites).

> Point d'arbitrage : la qualité des données compte davantage que la quantité brute.
> Beaucoup d'équipes ont montré qu'un corpus mieux filtré et plus petit surpasse un
> corpus plus gros mais bruité à budget de calcul égal.

## 5. Le pré-entraînement et les scaling laws

Le **pré-entraînement** est la phase longue et coûteuse : le modèle parcourt des
milliards ou milliers de milliards de tokens et ajuste ses poids pour mieux prédire.

### Objectif
Apprentissage auto-supervisé (self-supervised) : l'étiquette est fournie par le texte
lui-même (le token suivant réel). Aucune annotation humaine n'est nécessaire, ce qui
permet d'exploiter le web entier.

### Compute (calcul) en FLOPs

> FLOPs : Floating Point Operations, opérations en virgule flottante. C'est l'unité de
> mesure du travail de calcul d'un entraînement.

Estimation classique pour un Transformer : le coût total d'entraînement vaut environ

```
FLOPs ~= 6 x N x D

N = nombre de paramètres
D = nombre de tokens d'entraînement
6 = facteur empirique (forward + backward, ordres de grandeur)
```

Exemple : un modèle de 7B entraîné sur 2 000 milliards de tokens consomme de l'ordre de
6 x 7e9 x 2e12, soit environ 8,4e22 FLOPs. Des estimations similaires circulent pour
les grands modèles frontières, de l'ordre de 1e24 à 1e25 FLOPs.

### Scaling laws et Chinchilla

Les scaling laws (Kaplan et al., OpenAI) montrent que la perte décroît comme une loi de
puissance quand N, D ou le compute augmentent. Le travail **Chinchilla** (DeepMind,
2022) affine : pour un budget de calcul donné, il faut équilibrer paramètres et tokens.
Le ratio recommandé est d'environ **20 tokens par paramètre**. Un modèle de 70B devrait
donc voir de l'ordre de 1 400 milliards de tokens pour être "compute-optimal".

| Modèle | Tokens/paramètre | Note |
|---|---|---|
| Ratios d'époque (pré-2022) | 1 à 3 | Modèles sous-entraînés selon Chinchilla |
| Chinchilla optimal | environ 20 | Équilibre calcul/performance |
| Modèles modernes (Llama 3) | 100+ | "Sur-entraînés" pour réduire le coût d'inférence |

Modèle **frontière** : expression désignant les modèles les plus avancés à un instant
donné (GPT-4o, Claude Sonnet, Gemini, DeepSeek-V3). Non défini de façon rigoureuse.

## 6. Modèle de base vs modèle instruct

Un modèle qui sort tout juste du pré-entraînement est un **modèle de base** (base
model). Il complète du texte mais ne suit pas d'instructions. Si tu lui demandes
"Comment cuisiner des pâtes ?", il peut répondre par une autre question dans le même
style, car il modélise la distribution d'un document web, pas d'un assistant.

Un **modèle instruct** (ou "chat") a subi un post-training (section 7) pour :

- suivre des instructions,
- répondre en format assistant,
- refuser les demandes nuisibles.

Le modèle de base reste utile pour du fine-tuning créatif ou de la recherche, mais pour
un usage applicatif on part presque toujours d'un modèle instruct.

## 7. Post-training : SFT, RLHF, DPO et variantes

> Post-training : ensemble des entraînements après le pré-entraînement, visant à
> aligner le modèle sur les instructions, le format et les préférences humaines.

### SFT (Supervised Fine-Tuning)

On entraîne le modèle sur des paires instruction/réponse rédigées ou validées par des
humains (parfois générées par un modèle fort et filtrées). Le modèle apprend le format
du dialogue, l'art de suivre une consigne et un ton standard.

### RLHF (Reinforcement Learning from Human Feedback)

Le RLHF comporte typiquement trois étapes :

1. Le modèle SFT génère plusieurs réponses à un prompt.
2. Des annotateurs les classent par préférence.
3. On entraîne un **reward model** (modèle de récompense) à prédire ce classement.
4. On optimise le LLM par RL (souvent PPO) pour maximiser la récompense, avec une
   pénalité de divergence KL qui empêche de trop s'éloigner du modèle SFT.

> PPO (Proximal Policy Optimization) : algorithme de RL qui met à jour la politique par
> petits pas contrôlés pour rester stable.

### DPO, ORPO, KTO

Le RLHF avec PPO est complexe, instable et coûteux (plusieurs modèles en mémoire).

- **DPO (Direct Preference Optimization)** : réécrit l'objectif RLHF comme une simple
  classification sur des paires (réponse préférée vs rejetée). Pas de reward model
  séparé, beaucoup plus simple et stable.
- **ORPO** : combine SFT et alignement en une seule étape (pas de phase séparée).
- **KTO (Kahneman-Tversky Optimization)** : n'a besoin que d'un signal "bon/mauvais"
  par exemple, sans paires appariées, ce qui simplifie la collecte de données.

### Constitutional AI et rejection sampling

- **Constitutional AI** (Anthropic) : le modèle critique et révise ses propres réponses
  selon un ensemble de principes écrits (une "constitution"), réduisant la dépendance
  aux annotations humaines de sûreté.
- **Rejection sampling** : générer beaucoup de réponses, garder celles validées par un
  vérificateur ou un reward model, et refaire du SFT dessus. Simple et efficace.

## 8. Modèles de raisonnement et test-time compute

Une génération de modèles (OpenAI o1/o3, DeepSeek-R1) mise sur le calcul à l'inférence
plutôt que sur la taille. Le principe : laisser le modèle "réfléchir" longtemps avant
la réponse finale.

- **Chain of thought** (CoT, chaîne de pensée) : le modèle explicite des étapes
  intermédiaires.
- **Test-time compute** : on autorise beaucoup plus de tokens de raisonnement, parfois
  cachés à l'utilisateur, avant d'émettre la réponse.
- **RL à récompenses vérifiables** : la récompense provient d'un vérificateur
  automatique (compilateur, solveur mathématique, tests unitaires) plutôt que d'humains.
  Cela permet un RL à grande échelle sur les maths et le code.

| Approche | Coût à l'inférence | Force | Limite |
|---|---|---|---|
| Modèle standard | Faible | Réponses rapides, dialogues | Maths/raisonnement longs |
| Modèle de raisonnement | Élevé (tokens cachés) | Problèmes complexes multi-étapes | Latence et coût élevés |

## 9. Capacités émergentes et in-context learning

L'in-context learning (apprentissage en contexte) est la capacité du modèle à
"apprendre" une tâche simplement à partir d'exemples dans le prompt, sans modifier ses
poids.

- **Zero-shot** : aucune démonstration, on décrit la tâche.
- **One-shot** : un exemple.
- **Few-shot** : quelques exemples. Pour une classification, cela suffit souvent à
  obtenir des résultats corrects sans aucun entraînement.

Certaines capacités dites **émergentes** (apparaissant à partir d'une certaine échelle)
sont débattues : leur existence dépend beaucoup de la métrique choisie. On parle de
capacité émergente quand une performance passe brutalement de quasi nulle à bonne
au-delà d'un seuil de taille.

## 10. Fenêtre de contexte et Needle in a Haystack

La **fenêtre de contexte** est le nombre maximal de tokens que le modèle peut prendre
en entrée (plus la sortie). Elle est passée de quelques milliers (GPT-3) à 128k,
1M et au-delà (Gemini, modèles long-context).

Le test **Needle in a Haystack** (aiguille dans la botte de foin) consiste à placer une
information précise au milieu d'un long texte et à vérifier que le modèle la retrouve.
Il révèle que la performance n'est pas uniforme : les informations situées au milieu
d'un contexte très long sont parfois moins bien récupérées.

> Point pratique : une grande fenêtre de contexte ne remplace pas un bon RAG. Remplir
> 1M de tokens coûte cher, ralentit l'inférence et peut diluer l'attention. Voir
> [chapitre 07](../07-rag/README.md).

## 11. Hallucinations : pourquoi et comment atténuer

> Hallucination : affirmation fausse ou inventée, énoncée avec assurance.

Causes principales :

1. Objectif statistique : le modèle optimise la plausibilité, pas la vérité.
2. Données manquantes ou contradictoires sur le sujet.
3. Pression du prompt à répondre (le modèle préfère inventer que dire "je ne sais pas").
4. Fuite d'information : généralisation erronée à partir d'un cas proche.

Atténuation :

- **RAG** : fournir les sources dans le contexte ([chapitre 07](../07-rag/README.md)).
- **Prompt** : autoriser explicitement l'abstention ("si l'information manque, dis-le").
- **Vérification** : citations obligatoires, croisement de plusieurs modèles.
- **Fine-tuning** ciblé sur un domaine avec des données factuelles
  ([chapitre 06](../06-fine-tuning/README.md)).
- **Température basse** : réduit la variance mais ne supprime pas le problème.

Une hallucination n'est pas un bug isolable qu'on corrige une fois pour toutes : c'est
une propriété structurelle de la génération probabiliste.

## 12. Prompt engineering

Le prompt engineering est l'art de formuler l'entrée pour obtenir la sortie attendue.

| Technique | Description | Usage |
|---|---|---|
| Zero-shot | Consigne directe | Tâches simples |
| Few-shot | Exemples dans le prompt | Classification, formatage |
| Role prompting | Assigner un rôle | Ton et vocabulaire |
| Chain of thought | "Réfléchis étape par étape" | Maths, logique |
| Sortie structurée | "Réponds en JSON valide" | Intégration logicielle |
| Délimiteurs | Balises autour des données | Éviter la confusion instructions/données |

```
Instruction : tu es un extracteur d'informations.
Ne réponds QUE par un JSON valide, sans texte autour.

Texte à analyser (délimité) :
---
Marie Dupont, née en 1985, habite à Lyon.
---

Format attendu :
{"nom": "...", "annee_naissance": ..., "ville": "..."}
```

Attention : un prompt n'est pas une barrière de sécurité. Les délimiteurs aident à la
clarté mais ne protègent pas contre les injections (voir
[chapitre 13](../13-securite/README.md)).

## 13. Évaluation

Évaluer un LLM est difficile : le langage est ouvert et les métriques classiques
(BLEU, ROUGE) captent mal la qualité.

| Benchmark | Mesure | Remarque |
|---|---|---|
| MMLU | Connaissances générales (57 matières) | Référence, désormais saturé par les meilleurs |
| ARC | Raisonnement scientifique (questions type examen) | Variantes easy/challenge |
| HellaSwag | Complétion de phrase plausible | Sens commun |
| GSM8K | Problèmes mathématiques scolaires | Nécessite du raisonnement multi-étapes |
| MATH | Compétitions mathématiques | Plus dur que GSM8K |
| HumanEval | Génération de code Python | Mesuré par tests unitaires |

Problèmes structurels :

- **Contamination** : les questions d'un benchmark peuvent se retrouver dans les
  données d'entraînement, gonflant artificiellement les scores. D'où l'intérêt de
  benchmarks privés ou régulièrement renouvelés.
- **Saturation** : quand tous les bons modèles dépassent 85-90 pour cent sur un
  benchmark, il ne discrimine plus rien (cas de MMLU et HellaSwag).
- **LLM-as-judge** : utiliser un LLM fort pour noter les réponses d'un autre. Pratique,
  mais introduit des biais (préférence pour les réponses longues, pour son propre
  style).
- **Chatbot Arena** : classement Elo basé sur des votes humains en aveugle entre deux
  modèles. Considéré comme l'un des signaux les plus fiables, mais sensible au style.

## 14. Ouvert vs fermé, et coûts

| Aspect | Modèles ouverts | Modèles fermés (API) |
|---|---|---|
| Poids | Téléchargeables | Non disponibles |
| Exemples | Llama 3, Mistral, Qwen, DeepSeek | GPT-4o, Claude, Gemini |
| Coût | Matériel + électricité | Prix par token |
| Contrôle | Total (local, fine-tuning) | Limité aux paramètres exposés |
| Conformité | Tu gères les données | Dépend du fournisseur |

Ordres de grandeur de coût API (à titre indicatif, les prix bougent vite) : de l'ordre
de quelques dixièmes de centime à quelques centimes pour 1 000 tokens, les modèles
frontières étant bien plus chers que les petits modèles. Une requête conversationnelle
courante coûte typiquement une fraction de centime à quelques centimes.

> Règle pratique : prompt engineering et RAG d'abord, fine-tuning ensuite, entraînement
> complet en dernier recours. Chaque palier coûte beaucoup plus que le précédent.

## 15. Anthropomorphisme et limites réelles

Le modèle produit du texte fluide et peut donner l'impression de comprendre. C'est un
effet de surface. Il n'a ni intentions, ni croyances, ni mémoire persistante entre
appels (sauf mécanisme externe ajouté). Il ne raisonne pas "comme" un humain : il
calcule.

Limites concrètes :

- Pas de mémoire durable sans architecture externe.
- Faible fiabilité sur les calculs exacts et les faits récents (sans outils).
- Sensible à la formulation du prompt.
- Reproduit les biais présents dans ses données ([chapitre 14](../14-ethique-societe/README.md)).
- Fenêtre de contexte finie et coûteuse.

## Ce qu'il faut retenir
- Un LLM est un modèle autorégressif qui prédit le prochain token ; rien de plus, rien de moins.
- Le décodage (température, top-p, top-k, beam search) contrôle le compromis cohérence/créativité.
- Les Mixture of Experts offrent beaucoup de paramètres au total mais n'en activent qu'une fraction par token.
- Les scaling laws et le ratio Chinchilla (environ 20 tokens/paramètre) guident le dimensionnement de l'entraînement.
- Le post-training (SFT, RLHF, DPO) transforme un modèle de base en assistant utilisable.
- Les modèles de raisonnement dépensent du calcul à l'inférence plutôt qu'en paramètres.
- Le RAG est la parade principale aux hallucinations factuelles.
- Les benchmarks sont menacés par la contamination et la saturation ; l'arène humaine reste un signal utile.
- Un LLM ne "comprend" pas : l'anthropomorphisme mène à de mauvaises décisions d'ingénierie.

## Erreurs frequentes / idees recues
- "Un LLM plus gros est toujours meilleur" -> faux : les scaling laws disent qu'il faut équilibrer taille, données et calcul.
- "Le modèle apprend pendant la conversation" -> faux : les poids sont gelés après l'entraînement ; seul le contexte change.
- "Une hallucination est un bug qu'on peut corriger" -> faux : c'est une conséquence de l'objectif probabiliste.
- "Il suffit d'une grande fenêtre de contexte pour tout gérer" -> faux : coût, latence et dilution de l'attention forcent une sélection de contexte.
- "Le modèle comprend ce qu'il écrit" -> faux : il prédit des tokens plausibles.
- "Les benchmarks suffisent à juger un modèle" -> faux : contamination et saturation faussent les scores.

## Pour aller plus loin
- [Chapitre 04-transformers](../04-transformers/README.md) : l'attention, la tokenisation, les briques du LLM.
- [Chapitre 06-fine-tuning](../06-fine-tuning/README.md) : adapter un LLM à un domaine.
- [Chapitre 07-rag](../07-rag/README.md) : ancrer les réponses dans des sources.
- [Chapitre 09-inference-optimisation](../09-inference-optimisation/README.md) : réduire latence et coût d'inférence.
- [Chapitre 13-securite](../13-securite/README.md) : prompt injection et garde-fous.
- [Chapitre 15-ecosysteme](../15-ecosysteme/README.md) : panorama des modèles et fournisseurs.
- Ressources externes :
  - Article *Attention Is All You Need* (Vaswani et al., 2017).
  - Article *Training Compute-Optimal Large Language Models* (Chinchilla, DeepMind, 2022).
  - Documentation officielle des modèles Llama (Meta AI) et rapports techniques DeepSeek.

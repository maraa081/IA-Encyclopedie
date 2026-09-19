# RAG (Retrieval-Augmented Génération)

> Le RAG (génération augmentée par récupération) combine un moteur de recherche documentaire et un grand modèle de langage pour ancrer les réponses dans des sources externes vérifiables. Il permet d'utiliser des connaissances à jour ou privées, de citer ses sources et d'éviter de ré-entraîner le modèle à chaque changement de données.

## 1. Pourquoi le RAG

Un LLM seul (voir le [chapitre 05](../05-llm/README.md)) connaît uniquement ce qu'il a
appris pendant son entraînement. Sa mémoire est figée à la date de coupure du corpus
(`knowledge cutoff` : date au-delà de laquelle le modèle n'a vu aucun document), il ne
connaît pas tes documents internes, et il peut inventer des faits avec assurance
(hallucination). Le RAG attaque ces trois problèmes en allant chercher des documents
réels et en les fournissant au modèle comme contexte au moment de la question.

L'analogie classique : c'est un examen à livre ouvert. Le modèle n'a pas besoin de
mémoriser par cœur les 400 pages d'un manuel, il a juste besoin de savoir lire la bonne
page qu'on lui tend et d'en tirer la réponse. Le travail difficile consiste à trouver la
bonne page parmi des millions.

Le RAG se distingue du fine-tuning (voir le [chapitre 06](../06-fine-tuning/README.md)),
qui modifie les poids du modèle pour lui apprendre un style ou un comportement. Le RAG,
lui, ne touche jamais aux poids : il change seulement ce qu'on met dans le contexte. On
peut donc mettre à jour la base documentaire en quelques minutes, sans GPU ni
ré-entraînement.

Les quatre raisons principales d'adopter le RAG :

- **Connaissance à jour** : on met à jour l'index, pas le modèle.
- **Données privées** : documents internes que le modèle n'a jamais vus.
- **Citations** : chaque réponse peut être reliée au document source, ce qui est
  essentiel pour la confiance et l'audit.
- **Coût** : ré-indexer des Documents coûte quelques centimes, ré-entraîner un modèle
  de 70 milliards de paramètres coûte de l'ordre de plusieurs milliers d'euros par run.

## 2. Pipeline complet

Le pipeline RAG se découpe en deux phases : une phase hors ligne d'ingestion (on prépare
l'index) et une phase en ligne de requête (on répond à l'utilisateur).

```
PHASE HORS LIGNE (ingestion, faite une fois puis mise à jour)

  [Documents]      [Découpage]     [Embeddings]     [Base vectorielle]
   PDF, HTML  -->   chunking  -->   encodeur   -->   index + métadonnées
   Markdown         500 tok         bi-encoder       (HNSW, filtres)
     |                 |               |                   |
     v                 v               v                   v
   nettoyage      chevauchement    vecteurs 1024-d      persistance

PHASE EN LIGNE (par requête, doit tenir en quelques centaines de ms)

  [Question]    [Transformation]   [Retrieval]      [Reranking]     [Génération]
   utilisateur     HyDE,            top-k 50         cross-encoder    LLM lit le
     |            multi-query        dens+sparse       garde top 5     contexte
     v                 |                 |                 |             |
   5 a 20 mots         v                 v                 v             v
                  plusieurs         scores de         scores de      réponse
                  formulations      similarité        pertinence     + citations
```

Chaque étage a un coût et un bénéfice. Le retrieval (récupération) ramène un large
ensemble de candidats ; le reranking trie finement ; la génération synthétise. Si le
retrieval rate un document pertinent, aucun étage ultérieur ne pourra le rattraper :
c'est le point de défaillance principal.

Le post-traitement final (facultatif mais recommandé en production) vérifie que la
réponse est bien ancrée dans les documents fournis, ajoute les références, et peut
refuser de répondre si le contexte est insuffisant.

## 3. Chunking

Le `chunking` est le découpage des documents en fragments (chunks) de taille adaptée à
la recherche. C'est l'étape la plus souvent bâclée et la plus déterminante. Un chunk
trop petit perd le contexte, un chunk trop gros dilue l'information pertinente et
consomme des tokens utiles.

Repères de taille :

| Type de contenu | Taille de chunk typique | Chevauchement |
|---|---|---|
| Texte courant (prose) | 300 a 500 tokens | 10 a 20 % |
| Documentation technique | 500 a 800 tokens | 10 % |
| Code source | 1 fonction ou 1 classe | 0 |
| Tableaux / fiches | 1 enregistrement logique | 0 |
| Dialogue / transcription | 200 a 400 tokens | tour de parole |

Le chevauchement (`overlap`) consiste à répéter la fin d'un chunk au début du suivant.
Il évite de couper une idée en deux à la frontière. Trop de chevauchement gonfle
l'index inutilement ; 10 à 20 % suffisent dans la plupart des cas.

Stratégies de découpage, de la plus simple à la plus fine :

- **Taille fixe** : on coupe tous les N tokens. Simple, mais coupe parfois au milieu
  d'une phrase ou d'une section.
- **Par structure** : on respecte la structure du document (titres Markdown, balises
  HTML, sections). Préserve la cohérence sémantique, dépend d'un parsage correct.
- **Sémantique** : on coupe quand la similarité entre phrases voisines chute, ce qui
  marque un changement de sujet. Plus coûteux car il faut encoder les phrases.
- **Parent-child** : on indexe de petits chunks (les enfants) pour la recherche, mais
  on récupère le grand chunk parent pour la génération. Bon compromis précision/contexte.
- **Sentence-window** : on indexe chaque phrase, mais on récupère aussi les phrases
  voisines autour de celle trouvée. Utile pour les questions très ciblées.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Découpage recursif : essaie de couper sur les paragraphes,
# puis les phrases, puis les mots, pour preserver la structure.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # en caracteres (approximation des tokens)
    chunk_overlap=80,      # environ 16 pourcent de chevauchement
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document)
for i, c in enumerate(chunks):
    print(f"chunk {i}: {len(c)} caracteres")
```

## 4. Embeddings

Un `embedding` (plongement) est une représentation d'un texte sous forme de vecteur de
nombres réels. Deux textes de sens proche ont des vecteurs proches dans l'espace. C'est
la brique qui permet la recherche sémantique : chercher "comment réparer une fuite" peut
retrouver un document qui parle de "joint défectueux" sans partager aucun mot.

Les modèles d'embedding se comparent sur des classements comme MTEB (Massive Text
Embedding Benchmark, un benchmark public). Ordres de grandeur :

| Modèle | Dimensions | Type | Remarque |
|---|---|---|---|
| OpenAI text-embedding-3-small | 1536 | propriétaire | bon rapport qualité/prix, redimensionnable |
| OpenAI text-embedding-3-large | 3072 | propriétaire | plus précis, plus cher |
| BGE (BAAI, versions v1.5 / M3) | 384 a 1024 | ouvert | forte communauté, multilingue en M3 |
| E5 (Microsoft) | 384 a 1024 | ouvert | très utilisé en recherche |
| Cohere Embed v3 | 1024 | propriétaire | multilingue, orienté entreprise |
| nomic-embed-text | 768 | ouvert | longue fenêtre de contexte |

La comparaison des vecteurs se fait par similarité cosinus ou produit scalaire.

- **Similarité cosinus** : mesure l'angle entre deux vecteurs, entre -1 et 1. Elle est
  insensible à la longueur du vecteur. Formule : `cos(a,b) = (a . b) / (||a|| * ||b||)`.
- **Produit scalaire** (`dot product`) : `a . b`. Plus rapide à calculer, mais sensible
  aux normes. Si on **normalise** tous les vecteurs à une norme de 1, produit scalaire
  et cosinus donnent exactement le même classement. Beaucoup de bases vectorielles
  exigent des vecteurs normalisés pour utiliser l'index rapide.

Ne pas normaliser est une source classique de bug : des vecteurs de grande norme
dominent le classement sans être plus pertinents.

Il faut distinguer deux familles de modèles :

- **Bi-encoder** : encode la question et le document séparément, en deux passes
  indépendantes. Rapide, car on peut pré-calculer les embeddings des documents. C'est
  ce qu'on utilise pour le retrieval initial.
- **Cross-encoder** : encode la question et le document ensemble, dans une seule passe.
  Beaucoup plus précis, mais il faut le faire pour chaque paire (question, document),
  donc impossible de pré-calculer. On l'utilise pour le reranking sur un petit nombre
  de candidats.

Ce compromis vitesse/précision est la raison d'être du pipeline en deux temps :
retrieval large et rapide avec un bi-encoder, puis reranking étroit et précis avec un
cross-encoder.

## 5. Bases vectorielles

Une base vectorielle stocke des millions de vecteurs et retrouve les plus proches d'une
requête en un temps très court. Le calcul exact (comparer la requête à tous les vecteurs)
est en O(n) et devient trop lent au-delà de quelques centaines de milliers de vecteurs ;
on utilise donc des index approximatifs (ANN, Approximate Nearest Neighbors).

Comparatif :

| Base | Type | Points forts | Points faibles |
|---|---|---|---|
| FAISS | bibliothèque | très rapide, GPU, gratuit | pas de serveur, persistance à gérer |
| Qdrant | serveur | filtrage riche, écrit en Rust | ressource à héberger |
| Chroma | embarqué | simple, idéal prototype | passage à l'échelle limité |
| Weaviate | serveur | modules intégrés, hybride natif | plus lourd à opérer |
| pgvector | extension Postgres | un seul magasin (SQL + vecteurs) | performance sous forte charge |
| Milvus | serveur distribué | échelle massive (milliards) | complexe à déployer |
| Pinecone | SaaS managé | zéro ops, scalable | propriétaire, coût récurrent |

Deux index ANN dominent :

- **HNSW** (Hierarchical Navigable Small World) : un graphe hiérarchique où la recherche
  navigue de voisins en voisins. Excellent rappel et faible latence, mais forte
  consommation de mémoire.
- **IVF** (Inverted File Index) : on partitionne l'espace en amas (clusters) et on ne
  parcourt que quelques amas proches. Moins gourmand en mémoire, rappel un peu inférieur.

La `quantization` (quantification) compresse les vecteurs pour réduire la mémoire.
Passer de float32 (32 bits par dimension) à int8 (8 bits) divise la mémoire par 4 avec
une perte de précision modérée ; la quantization binaire va plus loin encore mais
dégrade nettement le rappel. C'est le réglage classique mémoire contre qualité.

Le filtrage par métadonnées (`metadata filtering`) restreint la recherche selon des
champs (date, auteur, service, niveau de confidentialité). Attention : un filtrage très
sélectif peut casser l'efficacité de l'index HNSW si le filtre est appliqué après la
recherche ; certaines bases proposent un filtrage intégré qui gère ce cas.

```python
# Exemple Qdrant : creation d'une collection avec distance cosinus
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)
# recherche filtree : seulement les documents publics
hits = client.search(
    collection_name="docs",
    query_vector=embedding_question,
    limit=20,
    query_filter={"must": [{"key": "visibilite", "match": {"value": "public"}}]},
)
```

## 6. Recherche dense, sparse et hybride

La recherche **dense** utilise les embeddings : elle capte le sens, mais peut rater un
terme rare ou un identifiant exact (référence de produit, nom propre, code d'erreur).

La recherche **sparse** repose sur la correspondance de termes. Le standard est
**BM25** (Best Match 25), une évolution de TF-IDF (fréquence du terme pondérée par sa
rareté dans le corpus). BM25 excelle sur les mots-clés exacts et reste étonnamment
difficile à battre sur des requêtes factuelles.

On combine les deux en **recherche hybride**. Deux façons de fusionner :

- **Pondération des scores** : on additionne les scores normalisés des deux méthodes
  avec un poids (par exemple 0.7 dense, 0.3 sparse). Simple, mais les scores n'ont pas
  la même échelle, la normalisation est délicate.
- **RRF** (Reciprocal Rank Fusion) : on ignore les scores bruts et on combine les rangs.
  Pour chaque document, on somme `1 / (k + rang)` sur chaque liste, avec `k` souvent
  fixé à 60. Robuste, sans réglage, c'est devenu le défaut des systèmes modernes.

```
RRF : score(doc) = somme sur chaque liste de 1 / (k + rang_dans_la_liste)

   Liste dense (rang)   Liste BM25 (rang)    score RRF (k=60)
   doc A (1)            doc C (1)             A: 1/61 = 0.0164
   doc B (2)            doc A (2)             C: 1/61 = 0.0164
   doc C (3)            doc D (3)             B: 1/62 = 0.0161
                                             A: + 1/62 = 0.0325  <- gagnant
```

L'hybride est en général supérieur au dense seul de plusieurs points de rappel, au prix
d'un index supplémentaire à maintenir (le lexique BM25) et d'une logique de fusion.

## 7. Reranking

Le retrieval ramène typiquement 20 à 50 candidats. Tous ne sont pas pertinents : la
similarité d'embedding est grossière. Le `reranking` réordonne ces candidats avec un
cross-encoder, bien plus précis, et ne garde que les 3 à 7 meilleurs pour la génération.

C'est l'étage au meilleur rapport gain/coût du pipeline : sur beaucoup de systèmes, il
apporte plus d'amélioration que le passage à un meilleur modèle de génération.

| Modèle de reranking | Type | Remarque |
|---|---|---|
| BGE-reranker (BAAI, base/large) | ouvert | référence open source |
| Cohere Rerank | propriétaire | API simple, multilingue |
| Jina Reranker | ouvert/propriétaire | bonnes performances multilingues |
| cross-encoder/ms-marco | ouvert | historique, point de départ |

```python
# Pseudo-code : retrieval puis reranking
candidats = index.search(embed(question), top_k=50)        # bi-encoder, rapide
paires = [(question, c.texte) for c in candidats]
scores = cross_encoder.predict(paires)                      # cross-encoder, précis
top = [c for _, c in sorted(zip(scores, candidats), reverse=True)[:5]]
contexte = "\n\n".join(c.texte for c in top)
```

Compromis : le cross-encoder coûte un appel de modèle par paire. Reranker 50 candidats
avec un gros cross-encoder ajoute typiquement 100 à 300 ms ; sur plus de 100 candidats,
la latence devient perceptible. On règle donc le nombre de candidats selon le budget de
latence (voir le [chapitre 09](../09-inference-optimisation/README.md)).

## 8. Transformation de requête

La question brute de l'utilisateur est rarement la meilleure requête pour la recherche :
elle est courte, ambiguë, ou contient plusieurs intentions. On la transforme avant le
retrieval.

- **HyDE** (Hypothetical Document Embeddings) : on demande au LLM de générer une réponse
  hypothétique, puis on cherche les documents proches de cette réponse plutôt que de la
  question. Intuition : une réponse ressemble plus à un document qu'une question.
- **Multi-query** : on génère plusieurs reformulations de la question et on fusionne les
  résultats. Couvre les cas où la question est mal formulée.
- **Décomposition** : on coupe une question complexe ("compare A et B sur le critère C")
  en sous-questions traitées séparément.
- **Routage** : on classe la question pour choisir la bonne base ou le bon outil (base
  technique contre base juridique). C'est le pont vers les agents (voir le
  [chapitre 08](../08-agents/README.md)).

Ces techniques ajoutent un ou plusieurs appels LLM avant la recherche : elles améliorent
le rappel mais augmentent la latence et le coût. On les réserve aux cas où le retrieval
naïf échoue.

## 9. RAG avancé

Les variantes modernes ajoutent des boucles de contrôle autour du pipeline de base.

- **Self-RAG** : le modèle juge lui-même si un document est pertinent et si sa réponse
  est fondée, et peut redemander une recherche si ce n'est pas le cas.
- **Corrective RAG (CRAG)** : on évalue la qualité des documents récupérés ; si le
  score est trop faible, on déclenche une recherche web en secours.
- **Graph RAG** : on construit un graphe d'entités et de relations extraites des
  documents, ce qui permet de répondre à des questions globales ("quels sont les thèmes
  récurrents de ce corpus") que la recherche par similarité gère mal.
- **Agentic RAG** : le RAG devient un outil que l'agent appelle plusieurs fois, avec des
  requêtes qu'il raffine en boucle. Puissant mais coûteux en tokens.
- **RAG multimodal** : on indexe aussi des images ou des pages scannées, via des
  modèles d'embedding multimodaux (voir le [chapitre 11](../11-multimodal/README.md)).
- **Contextual retrieval** : avant d'indexer, on préfixe chaque chunk d'un court résumé
  du contexte du document d'origine. Réduit la perte de contexte aux frontières.
- **Late chunking** : on envoie le document entier dans l'encodeur puis on découpe les
  embeddings obtenus, ce qui conserve les dépendances longues entre morceaux.

Règle pratique : commence simple (chunking soigné, hybride, reranking), mesure, puis
n'ajoute une variante avancée que si l'évaluation montre un manque précis. L'empilement
de techniques sans mesure est un piège courant.

## 10. Assemblage du prompt et "lost in the middle"

Une fois les meilleurs chunks sélectionnés, on assemble le prompt final :

```
[instruction système]
Tu réponds uniquement à partir des documents fournis.
Si la réponse n'y figure pas, dis-le explicitement.

[contexte]
<document source="rapport-2025.pdf" page="12">
...extrait...
</document>
<document source="note-interne.md">
...extrait...
</document>

[question]
Quelle est la procédure de validation ?

Réponse en citant les sources entre crochets.
```

Le phénomène du **lost in the middle** (perte au milieu) : les LLM exploitent mieux
l'information placée au début et à la fin de leur contexte, et tendent à ignorer ce qui
est au milieu quand le contexte est long. Conséquence pratique : placer les chunks les
plus pertinents en tête et en queue du contexte, et limiter le nombre total de chunks
plutôt que d'en empiler vingt.

Ordre de grandeur : les instructions de suivi ("réponds uniquement à partir des
documents") sont mieux respectées lorsqu'elles sont répétées au début ET à la fin du
contexte, pas seulement au tout début.

## 11. Évaluation du RAG

On ne peut améliorer que ce qu'on mesure. L'évaluation d'un RAG se fait à deux niveaux :
la qualité du retrieval et la qualité de la réponse générée.

| Métrique | Niveau | Question posée |
|---|---|---|
| Context recall | retrieval | les documents utiles ont-ils été retrouvés ? |
| Context précision | retrieval | les documents retrouvés sont-ils tous utiles ? |
| Faithfulness | génération | la réponse est-elle fondée sur le contexte ? |
| Answer relevance | génération | la réponse répond-elle à la question ? |

**RAGAS** (Retrieval Augmented Generation Assessment) est un framework de référence qui
implémente ces métriques, souvent en s'appuyant sur un LLM juge. On construit un **jeu
de test doré** (`golden test set`) : un ensemble de questions avec les documents
attendus et, idéalement, la réponse attendue, rédigé manuellement ou semi-automatiquement
puis vérifié. On rejoue ce jeu après chaque modification du pipeline.

Limite à connaître : un LLM juge est lui-même faillible et peut avoir des biais
(longueur, position). Il faut le calibrer sur un échantillon annoté à la main avant de
lui faire confiance à grande échelle.

## 12. RAG contre fine-tuning contre contexte long

| Critère | RAG | Fine-tuning | Contexte long |
|---|---|---|---|
| Mise à jour des connaissances | index, minutes | ré-entraînement, heures/jours | recoller les docs, par requête |
| Coût de mise à jour | faible | élevé | nul |
| Coût par requête | modéré (retrieval) | faible | élevé (contexte énorme) |
| Citations | natives | impossibles | possibles si prompt structuré |
| Comportement / style | inchangé | modifiable | inchangé |
| Latence | plus élevée | minimale | élevée (attention quadratique) |
| Risque d'hallucination | réduit si contexte bon | inchangé | réduit si contexte bon |

Ces approches ne s'excluent pas. Un système typique combine : un modèle fine-tuné pour
le ton et le format professionnel, un RAG pour les faits à jour, et éventuellement un
contexte long pour quelques documents volumineux ponctuels. On choisit en fonction de ce
qui doit changer : le comportement (fine-tuning) ou les faits (RAG).

## 13. Pièges courants

- **Ignorer l'ingestion** : la majorité des RAG décevants échouent au parsing ou au
  chunking, pas au modèle. Un PDF mal extrait (colonnes mélangées, tableaux cassés)
  ruine tout le reste.
- **Confondre rappel et précision** : ramener beaucoup de chunks augmente le rappel mais
  baisse la précision et noie le modèle. Ajuste le `top-k` par mesure.
- **Pas de reranking** : c'est souvent le gain le plus facile et le plus rentable.
- **Mélanger les langues** : un encodeur monolingue sur un corpus multilingue dégrade
  fortement le retrieval.
- **Oublier la fraîcheur** : un index jamais mis à jour renvoie d'anciennes procédures.
- **Pas d'évaluation** : sans jeu de test doré, on modifie le pipeline à l'aveugle et on
  croit les impressions au lieu de mesurer.
- **Sécurité** : les documents récupérés peuvent contenir une injection de prompt
  indirecte. Il faut traiter le contenu de la base comme non fiable (voir le
  [chapitre 13](../13-securite/README.md)).

## Ce qu'il faut retenir
- Le RAG fournit au LLM des documents externes au moment de la requête, sans toucher aux
  poids du modèle.
- Il apporte fraîcheur, accès aux données privées, citations et coût de mise à jour bas.
- Le pipeline comprend ingestion, chunking, embeddings, indexation, retrieval,
  reranking, génération et post-traitement.
- Le chunking (taille et chevauchement) est l'étape la plus déterminante et la plus
  souvent négligée.
- Un bi-encoder sert au retrieval rapide, un cross-encoder au reranking précis.
- La recherche hybride (dense plus BM25, fusionnée par RRF) bat généralement le dense
  seul.
- Le reranking est souvent le meilleur gain par unité de coût.
- Les variantes avancées (Self-RAG, CRAG, Graph RAG) ajoutent du contrôle mais ne se
  justifient que par une mesure.
- L'évaluation repose sur des métriques comme RAGAS et un jeu de test doré.

## Erreurs fréquentes / idées reçues
- Idée reçue : le RAG rend le fine-tuning inutile -> ce sont des outils complémentaires ;
  le RAG gère les faits, le fine-tuning le comportement.
- Idée reçue : un contexte de 1 million de tokens remplace le RAG -> le coût et la
  latence par requête restent prohibitifs et la perte au milieu persiste.
- Idée reçue : il suffit de prendre le meilleur modèle d'embedding du classement -> le
  chunking et le reranking pèsent souvent plus lourd que le choix du modèle.
- Idée reçue : plus on récupère de chunks, mieux c'est -> au-delà de quelques chunks
  pertinents, le bruit dégrade la réponse.
- Idée reçue : la similarité cosinus et le produit scalaire sont interchangeables -> ils
  ne le sont que si les vecteurs sont normalisés.
- Idée reçue : un RAG est forcément exact -> si le retrieval rate le bon document, le
  modèle répond faux avec assurance.

## Pour aller plus loin
- [Chapitre 04 - Transformers](../04-transformers/README.md) : embeddings et attention.
- [Chapitre 05 - LLM](../05-llm/README.md) : contexte, hallucination, connaissance figée.
- [Chapitre 06 - Fine-Tuning](../06-fine-tuning/README.md) : quand adapter les poids.
- [Chapitre 08 - Agents](../08-agents/README.md) : le RAG comme outil d'agent.
- [Chapitre 13 - Sécurité](../13-securite/README.md) : injection indirecte via documents.
- Documentation officielle : Lewis et al., "Retrieval-Augmented Génération for
  Knowledge-Intensive NLP Tasks" (Facebook AI Research, 2020, arXiv:2005.11401).
- Documentation RAGAS : framework d'évaluation RAG (docs.ragas.io).
- Documentation Qdrant et FAISS : guides d'indexation vectorielle.

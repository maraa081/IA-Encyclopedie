# Glossaire de l'IA

> Plus de 340 termes, classés de A à Z. Chaque entrée tient en une à trois lignes et
> donne, quand c'est pertinent, un ordre de grandeur ou un repère concret.

## Comment lire ce glossaire

- Les **termes anglais** sont conservés quand c'est l'usage courant dans la profession,
  avec la traduction entre parenthèses à la première mention utile.
- La mention `[ch. NN]` renvoie au chapitre qui développe le sujet.
- Les ordres de grandeur sont donnés en unités explicites (Go, ms, tokens/s) et sont
  des valeurs typiques, pas des garanties.
- Pour les sigles, voir la section [Acronymes](#acronymes) en fin de document.

---

## A

**Activation (fonction d')** : Fonction non linéaire appliquée à la sortie d'un neurone (ReLU, GELU, GELU, softmax). Sans elle, empiler des couches revient à une seule transformation linéaire. [ch. 03](03-reseaux-de-neurones/README.md)

**Activation checkpointing** : Technique d'entraînement qui ne stocke pas toutes les activations intermédiaires et les recalcule pendant la rétropropagation. Échange du temps de calcul contre de la VRAM (gain typique : plusieurs Go sur un modèle de 7B). [ch. 10](10-infrastructure/README.md)

**Adam** : Optimiseur combinant moment du gradient et moment du carré du gradient, avec un pas adaptatif par paramètre. Valeur par défaut de fait pour entraîner un réseau de neurones. [ch. 01](01-fondamentaux/README.md), [ch. 03](03-reseaux-de-neurones/README.md)

**AdamW** : Variante d'Adam où la régularisation L2 est appliquée directement aux poids (weight decay découplé) au lieu d'être ajoutée au gradient. Standard actuel pour les transformers. [ch. 03](03-reseaux-de-neurones/README.md)

**Adapter** : Petite couche supplémentaire insérée dans un modèle gelé pour l'adapter à une tâche. Chaque adaptateur pèse quelques Mo contre plusieurs Go pour le modèle complet. [ch. 06](06-fine-tuning/README.md)

**Adversarial example (exemple adversarial)** : Entrée légèrement modifiée, souvent de façon imperceptible pour un humain, qui fait changer la décision d'un modèle. La perturbation est calculée par gradient, pas choisie au hasard. [ch. 13](13-securite/README.md)

**Agent** : Système qui boucle : il observe, décide, agit au moyen d'outils, puis observe le résultat. Un chatbot qui répond en un tour n'est pas un agent. [ch. 08](08-agents/README.md)

**Agentic RAG** : RAG dans lequel le modèle décide lui-même des requêtes à lancer, des sources à interroger et du moment où il a assez d'information. Plus souple, plus lent, plus cher qu'un RAG linéaire. [ch. 07](07-rag/README.md)

**AGI (Artificial General Intelligence / IA générale)** : IA hypothétique capable de réaliser n'importe quelle tâche intellectuelle humaine. Aucune définition faisant consensus, donc aucune évaluation fiable de sa présence. [ch. 17](17-au-dela/README.md)

**AI Act (règlement européen sur l'IA)** : Règlement de l'UE qui classe les systèmes d'IA par niveau de risque et impose des obligations proportionnées. Entré en application par étapes à partir de 2025. [ch. 14](14-ethique-societe/README.md)

**AI safety** : Ensemble des travaux visant à rendre les systèmes d'IA fiables, contrôlables et non nuisibles : alignement, interprétabilité, évaluation de risques, garde-fous. [ch. 14](14-ethique-societe/README.md), [ch. 17](17-au-dela/README.md)

**ALiBi (Attention with Linear Biases)** : Méthode d'encodage de la position qui ajoute un biais linéaire décroissant selon la distance, permettant une extrapolation à des contextes plus longs que ceux vus à l'entraînement. [ch. 04](04-transformers/README.md)

**Alignement** : Faire en sorte qu'un modèle poursuive ce que son concepteur veut réellement, et pas seulement ce qu'il optimise formellement. Familles : RLHF, DPO, constitutional AI. [ch. 05](05-llm/README.md)

**AlphaFold** : Système de DeepMind qui prédit la structure 3D des protéines à partir de leur séquence d'acides aminés, cas d'usage emblématique de l'IA scientifique. [ch. 17](17-au-dela/README.md)

**Annotation** : Attribution manuelle ou assistée d'étiquettes à des données (classe, note de qualité, réponse préférée). C'est souvent le poste de coût dominant d'un projet ML. [ch. 12](12-donnees/README.md)

**Anonymisation** : Suppression ou transformation irréversible des données personnelles. Attention : un LLM peut réidentifier partiellement des données mal anonymisées. [ch. 14](14-ethique-societe/README.md)

**API (Application Programming Interface)** : Interface par laquelle on consomme un modèle distant. On envoie du texte et on reçoit du texte, facturé au token. [ch. 15](15-ecosysteme/README.md)

**Apprentissage auto-supervisé (self-supervised learning)** : Le modèle fabrique lui-même ses étiquettes à partir des données brutes (par exemple : prédire le mot masqué). C'est le régime qui a rendu possible le pré-entraînement des LLM. [ch. 01](01-fondamentaux/README.md), [ch. 05](05-llm/README.md)

**Apprentissage continu (continual learning)** : Capacité à apprendre de nouvelles données sans réapprendre de zéro et sans oublier les anciennes. Problème ouvert, souvent conflictuel avec l'oubli catastrophique. [ch. 06](06-fine-tuning/README.md)

**Apprentissage fédéré (federated learning)** : Entraînement réparti sur de nombreux appareils qui n'échangent que des mises à jour de paramètres, jamais les données brutes. [ch. 10](10-infrastructure/README.md), [ch. 14](14-ethique-societe/README.md)

**Apprentissage non supervisé** : Apprentissage sans étiquette : on cherche une structure dans les données (clusters, dimensions principales, densités). [ch. 01](01-fondamentaux/README.md)

**Apprentissage par renforcement (RL)** : Un agent agit dans un environnement, reçoit des récompenses, et apprend une politique qui maximise la récompense cumulée. [ch. 01](01-fondamentaux/README.md)

**Apprentissage semi-supervisé** : Combinaison d'un petit jeu étiqueté et d'un grand jeu non étiqueté. Souvent le meilleur rapport qualité/coût quand l'annotation est chère. [ch. 01](01-fondamentaux/README.md)

**Apprentissage supervisé** : Apprentissage à partir de couples entrée/sortie attendue. Le modèle apprend une fonction qui généralise à de nouvelles entrées. [ch. 01](01-fondamentaux/README.md)

**ASI (Artificial Superintelligence / superintelligence)** : IA hypothétique surpassant les humains dans tous les domaines. Notion spéculative, objet de débats non tranchés. [ch. 17](17-au-dela/README.md)

**ASR (Automatic Speech Recognition / reconnaissance automatique de la parole)** : Transcription de la parole en texte. Modèle de référence : Whisper. [ch. 11](11-multimodal/README.md)

**Attention** : Mécanisme qui calcule, pour chaque élément d'une séquence, une moyenne pondérée des autres éléments, les poids dépendant de la similarité entre requête et clés. C'est la brique centrale du transformer. [ch. 04](04-transformers/README.md)

**Attention multi-têtes (multi-head attention)** : Plusieurs calculs d'attention parallèles sur des projections différentes, dont les sorties sont concaténées. Permet de capter plusieurs types de relations à la fois. [ch. 04](04-transformers/README.md)

**Attention sparse** : Attention qui ne considère qu'un sous-ensemble des positions (fenêtre locale, motifs creux) pour réduire le coût quadratique en longueur de séquence. [ch. 04](04-transformers/README.md)

**Auto-régressif** : Se dit d'un modèle qui génère une séquence élément par élément, chaque nouvel élément étant conditionné par les précédents. Un LLM est auto-régressif. [ch. 05](05-llm/README.md)

**Autoencodeur** : Réseau entraîné à reconstruire son entrée en passant par une représentation de dimension réduite. Base des architectures de compression et de débruitage. [ch. 03](03-reseaux-de-neurones/README.md)

**Augmentation de données** : Création d'exemples supplémentaires par transformations (rotations, recadrages, paraphrases). Réduit le sur-apprentissage sans collecter de nouvelles données. [ch. 12](12-donnees/README.md)

**AWQ (Activation-aware Weight Quantization)** : Méthode de quantification 4 bits qui protège les poids les plus importants pour les activations. Populaire pour faire tenir un modèle 7B sur une carte 8 Go. [ch. 09](09-inference-optimisation/README.md)

## B

**Backdoor (porte dérobée)** : Comportement malveillant déclenché par un motif précis (mot-clé, phrase particulière) injecté volontairement dans un modèle pendant l'entraînement. Reste invisible sur les tests normaux. [ch. 13](13-securite/README.md)

**Backpropagation (rétropropagation du gradient)** : Algorithme qui applique la règle de la chaîne pour calculer le gradient de la fonction de coût par rapport à chaque poids, couche par couche, en partant de la sortie. [ch. 03](03-reseaux-de-neurones/README.md)

**Bande passante mémoire (memory bandwidth)** : Débit auquel un processeur lit la mémoire. Souvent le facteur limitant de l'inférence d'un LLM : une carte grand public récente est autour de 500 à 1 000 Go/s, une carte de datacenter autour de 2 à 8 To/s. [ch. 10](10-infrastructure/README.md)

**Base de données vectorielle (vector database)** : Système qui stocke des vecteurs et répond vite à la question « quels vecteurs sont les plus proches de celui-ci ? ». Exemples : FAISS, Qdrant, Chroma, pgvector, Milvus, Weaviate, Pinecone. [ch. 07](07-rag/README.md)

**Baseline (référence)** : Modèle ou méthode simple servant de point de comparaison obligatoire. Un résultat sans baseline ne veut rien dire. [ch. 01](01-fondamentaux/README.md)

**Batch (lot)** : Groupe d'exemples traités ensemble avant une mise à jour des paramètres. Le batch size influence la stabilité, la vitesse et l'usage mémoire. [ch. 01](01-fondamentaux/README.md)

**Batch size** : Nombre d'exemples par batch. Typiquement de quelques unités pour un grand modèle à quelques milliers pour un petit modèle. Trop grand sans ajuster le learning rate dégrade la convergence. [ch. 01](01-fondamentaux/README.md)

**BatchNorm (normalisation par batch)** : Normalisation des activations sur la dimension du batch, très efficace pour les CNN. Mal adaptée aux séquences de longueur variable, où LayerNorm lui est préférée. [ch. 03](03-reseaux-de-neurones/README.md)

**Bayes (théorème de)** : Relation entre probabilité à priori, vraisemblance et probabilité à posteriori : P(A|B) = P(B|A) P(A) / P(B). Fondement des classifieurs bayésiens et du raisonnement probabiliste. [ch. 02](02-mathematiques/README.md)

**Beam search** : Décodage qui maintient plusieurs hypothèses en parallèle et garde les k meilleures à chaque étape. Améliore la cohérence globale, mais plus coûteux et moins varié que l'échantillonnage. [ch. 05](05-llm/README.md)

**Benchmark** : Jeu de tests standardisé servant à comparer des modèles (MMLU, GSM8K, HumanEval...). Un modèle peut être bon au benchmark et décevant en usage réel. [ch. 05](05-llm/README.md)

**BERT (Bidirectional Encoder Représentations from Transformers)** : Famille de modèles encodeurs, entraînés à prédire des mots masqués, utilisés pour la compréhension et l'embedding plutôt que pour la génération. [ch. 04](04-transformers/README.md)

**BF16 (bfloat16)** : Format flottant sur 16 bits, avec la même plage d'exposant que le FP32 mais une précision réduite. Préféré au FP16 pour l'entraînement car il n'a pas besoin de mise à l'échelle des gradients. [ch. 10](10-infrastructure/README.md)

**Biais (statistique)** : Propriété d'un estimateur qui ne converge pas vers la vraie valeur. L'erreur d'un modèle se décompose en biais (trop simple) et variance (trop sensible aux données). [ch. 01](01-fondamentaux/README.md)

**Biais (éthique)** : Traitement systématiquement défavorable d'un groupe, hérité des données, de l'objectif choisi ou du contexte de déploiement. Se mesure par des indicateurs comme la parité démographique. [ch. 14](14-ethique-societe/README.md)

**Biais-variance (compromis)** : Arbitrage entre un modèle trop rigide (biais élevé, sous-apprentissage) et un modèle trop flexible (variance élevée, sur-apprentissage). [ch. 01](01-fondamentaux/README.md)

**Bi-encoder** : Modèle qui encode requête et document séparément en vecteurs comparés ensuite par similarité. Rapide, donc utilisé pour la recherche ; moins précis qu'un cross-encoder. [ch. 07](07-rag/README.md)

**bitsandbytes** : Bibliothèque qui implémente les couches quantifiées 8 et 4 bits (dont NF4 utilisé par QLoRA) pour PyTorch. [ch. 06](06-fine-tuning/README.md)

**Black box (boîte noire)** : Se dit d'un modèle dont on ne peut expliquer le comportement interne de façon fiable. La plupart des LLM sont dans ce cas, d'où le champ de l'interprétabilité. [ch. 17](17-au-dela/README.md)

**BM25** : Fonction de classement de documents par mots-clés (famille TF-IDF), très solide en recherche lexicale et souvent combinée à la recherche vectorielle. [ch. 07](07-rag/README.md)

**BPE (Byte Pair Encoding)** : Algorithme de tokenisation qui fusionne itérativement les paires de symboles les plus fréquentes. Utilisé par la plupart des LLM modernes. [ch. 04](04-transformers/README.md)

## C

**C2PA (Coalition for Content Provenance and Authenticity)** : Standard de métadonnées signées permettant de tracer l'origine d'un contenu (appareil, retouches, IA). Utile mais retirable : ce n'est pas une preuve absolue. [ch. 14](14-ethique-societe/README.md)

**Cas pratique (évaluation)** : Exercice d'application consistant à résoudre une situation concrète avec les règles applicables. Analogue, en ML, au test de généralisation hors distribution. [ch. 05](05-llm/README.md)

**Causalité** : Relation de cause à effet. Un modèle prédictif corrèle sans comprendre le mécanisme, ce qui casse dès que la distribution change. Voir « corrélation n'est pas causalité ». [ch. 02](02-mathematiques/README.md)

**Chain of Thought (chaîne de pensée)** : Faire produire au modèle des étapes intermédiaires avant la réponse finale. Améliore nettement les tâches de raisonnement et de calcul. [ch. 05](05-llm/README.md)

**Chat template** : Format exact des messages (rôles système, utilisateur, assistant, tokens spéciaux) attendu par un modèle instruct. Un template erroné dégrade fortement la qualité perçue. [ch. 05](05-llm/README.md)

**Chatbot Arena (LMArena)** : Classement par comparaisons à l'aveugle où des votants humains choisissent la meilleure réponse entre deux modèles anonymes, agrégé par un score Elo. [ch. 05](05-llm/README.md)

**Checkpoint** : Sauvegarde de l'état d'un modèle pendant l'entraînement (poids, état de l'optimiseur, itération). Permet de reprendre un entraînement interrompu. [ch. 01](01-fondamentaux/README.md)

**Chinchilla (lois d'échelle)** : Résultat empirique selon lequel, à budget de calcul fixé, il faut entraîner un modèle plus petit sur beaucoup plus de tokens que ce qui se faisait avant, typiquement de l'ordre de 20 tokens par paramètre. [ch. 05](05-llm/README.md)

**Chunking (découpage)** : Découpage des documents en fragments indexables. La taille, le chevauchement et les frontières (titres, paragraphes, fonctions) déterminent en grande partie la qualité d'un RAG. [ch. 07](07-rag/README.md)

**Classe (classification)** : Catégorie discrète à prédire. Pour plus de deux classes on parle de classification multiclasse ; pour plusieurs étiquettes simultanées, de classification multi-étiquettes. [ch. 01](01-fondamentaux/README.md)

**Classifier-free guidance (CFG)** : Technique de génération d'image qui combine une prédiction conditionnée par le prompt et une prédiction non conditionnée, avec un poids de guidage. Un poids élevé augmente la fidélité au prompt mais appauvrit la diversité. [ch. 11](11-multimodal/README.md)

**Clé (key) / Requête (query) / Valeur (value)** : Les trois projections utilisées par l'attention : la requête décrit ce qu'on cherche, la clé ce qu'un élément propose, la valeur ce qu'il transmet effectivement. [ch. 04](04-transformers/README.md)

**CLIP** : Modèle entraîné de façon contrastive à rapprocher images et textes correspondants dans un même espace vectoriel. Permet la recherche d'images par description. [ch. 11](11-multimodal/README.md)

**Clustering (partitionnement)** : Regroupement non supervisé d'objets similaires. K-means reste la baseline à battre. [ch. 01](01-fondamentaux/README.md)

**CNN (Convolutional Neural Network / réseau convolutif)** : Réseau utilisant des filtres glissants partagés pour exploiter la localité spatiale. Longtemps l'architecture dominante en vision. [ch. 03](03-reseaux-de-neurones/README.md)

**Cold start** : Situation où le service doit charger un modèle en mémoire avant la première requête (plusieurs secondes à dizaines de secondes). Rendu invisible par le keep-alive ou le préchargement. [ch. 09](09-inference-optimisation/README.md)

**Computer use** : Capacité d'un agent à piloter une interface graphique (clics, frappes, lecture d'écran) plutôt qu'une API. Puissant et fragile : une erreur d'un pixel casse le plan. [ch. 08](08-agents/README.md)

**Compute (calcul)** : Ressource de calcul consommée, souvent exprimée en FLOPs ou en GPU-heures. Le compute, les données et les paramètres forment les trois leviers d'un modèle. [ch. 10](10-infrastructure/README.md)

**Compute-optimal** : Répartition optimale d'un budget de calcul entre taille du modèle et volume de données. Voir Chinchilla. [ch. 05](05-llm/README.md)

**Contexte (context)** : Tout ce que le modèle voit au moment de générer : instructions système, historique, documents fournis, sorties d'outils. Le contexte est la seule mémoire de travail d'un LLM. [ch. 05](05-llm/README.md)

**Contexte long (long context)** : Modèles capables de traiter de très grandes entrées (de 128 000 à plus d'un million de tokens). Attention : capacité annoncée n'égale pas récupération fiable au milieu du contexte. [ch. 04](04-transformers/README.md)

**Continual pré-entraînement** : Poursuite du pré-entraînement sur un corpus spécialisé (code, langue, domaine médical) pour ajuster un modèle de base. [ch. 06](06-fine-tuning/README.md)

**Contrastif (apprentissage)** : Objectif qui rapproche les représentations des paires associées et éloigne les paires non associées. Base de CLIP et de la plupart des modèles d'embedding. [ch. 11](11-multimodal/README.md)

**ControlNet** : Extension qui ajoute un conditionnement spatial (pose, contours, profondeur) à un modèle de diffusion, pour contrôler la composition de l'image générée. [ch. 11](11-multimodal/README.md)

**Convergence** : Moment où la fonction de coût se stabilise et où les améliorations deviennent négligeables. Le learning rate et la taille du batch gouvernent la vitesse de convergence. [ch. 01](01-fondamentaux/README.md)

**Convolution** : Opération qui applique un petit filtre sur une fenêtre glissante de l'entrée, avec des poids partagés en tous points. Détecte le même motif où qu'il apparaisse. [ch. 03](03-reseaux-de-neurones/README.md)

**Corrélation** : Mesure d'association linéaire entre deux variables, entre -1 et 1. Ne prouve aucune causalité. [ch. 02](02-mathematiques/README.md)

**Coût (loss), fonction de** : Fonction scalaire mesurant l'écart entre prédiction et cible, dont on minimise la moyenne par descente de gradient. [ch. 01](01-fondamentaux/README.md)

**CPU (Central Processing Unit)** : Processeur généraliste. Excellent pour le séquentiel et le contrôle, faible pour la multiplication matricielle massive : il est courant qu'un GPU soit 10 à 100 fois plus rapide sur les calculs d'un LLM. [ch. 10](10-infrastructure/README.md)

**Cross-encoder** : Modèle qui lit requête et document ensemble et produit directement un score de pertinence. Plus précis qu'un bi-encoder, mais on ne peut pas précalculer les scores : on l'utilise en réordonnancement. [ch. 07](07-rag/README.md)

**Cross-validation (validation croisée)** : Découpage répété des données en k blocs, chacun servant une fois de validation. Estimation plus robuste de la performance, au prix de k entraînements. [ch. 01](01-fondamentaux/README.md)

**CUDA** : Plateforme de calcul GPU de NVIDIA, avec son langage et ses bibliothèques (cuBLAS, cuDNN, FlashAttention). C'est l'écosystème qui verrouille de fait le marché de l'entraînement. [ch. 10](10-infrastructure/README.md)

**Curriculum learning** : Entraîner d'abord sur des exemples faciles puis de plus en plus difficiles. En entraînement adversarial, l'ordre des budgets de perturbation change radicalement le résultat final. [ch. 13](13-securite/README.md)

## D

**Data leakage (fuite de données)** : Information du jeu de test qui contamine l'entraînement ou la validation, gonflant artificiellement les scores. Erreur très fréquente et rarement détectée. [ch. 01](01-fondamentaux/README.md)

**Data parallel (parallélisme de données)** : Répliquer le modèle sur plusieurs GPU et répartir les batchs, en synchronisant les gradients. Méthode la plus simple et la plus répandue. [ch. 10](10-infrastructure/README.md)

**Data poisoning (empoisonnement de données)** : Injection volontaire d'exemples corrompus dans un jeu d'entraînement pour modifier le comportement futur du modèle. [ch. 13](13-securite/README.md)

**Dataset (jeu de données)** : Ensemble structuré d'exemples. Sa qualité, sa couverture et sa licence comptent souvent plus que l'architecture choisie. [ch. 12](12-donnees/README.md)

**Dataset card (datasheet)** : Documentation décrivant la composition, la collecte, les biais connus et la licence d'un jeu de données. [ch. 12](12-donnees/README.md)

**Décodage (decoding)** : Processus de choix du prochain token à partir des probabilités produites par le modèle : greedy, température, top-k, top-p, beam search. [ch. 05](05-llm/README.md)

**Deep learning (apprentissage profond)** : Machine learning à base de réseaux de neurones profonds, qui apprennent eux-mêmes leurs représentations. [ch. 01](01-fondamentaux/README.md), [ch. 03](03-reseaux-de-neurones/README.md)

**Deepfake** : Contenu audio ou vidéo synthétique imitant une personne réelle. Le coût de production est devenu quasi nul, la détection reste difficile. [ch. 14](14-ethique-societe/README.md)

**Dense (vecteur dense)** : Représentation où presque toutes les dimensions sont non nulles (par opposition aux représentations creuses type sac de mots). [ch. 02](02-mathematiques/README.md)

**Descente de gradient (gradient descent)** : Algorithme d'optimisation qui déplace les paramètres dans la direction opposée au gradient. Par mini-batchs en pratique, d'où le nom SGD. [ch. 01](01-fondamentaux/README.md)

**Diffusion (modèle de)** : Modèle génératif entraîné à débruiter progressivement du bruit, puis capable de générer une image en partant de bruit pur et en inversant le processus. [ch. 03](03-reseaux-de-neurones/README.md), [ch. 11](11-multimodal/README.md)

**Diffusion latente (latent diffusion)** : Diffusion appliquée dans l'espace compressé d'un autoencodeur plutôt que dans l'espace des pixels, ce qui réduit fortement le coût. Base de Stable Diffusion. [ch. 11](11-multimodal/README.md)

**Distillation** : Entraîner un petit modèle (élève) à imiter les sorties d'un grand modèle (professeur). Souvent le meilleur moyen d'obtenir un modèle spécialisé et pas cher. [ch. 06](06-fine-tuning/README.md)

**Distributed training (entraînement distribué)** : Entraînement réparti sur plusieurs GPU ou plusieurs machines, par parallélisme de données, de tenseurs ou de pipeline. [ch. 10](10-infrastructure/README.md)

**Divergence KL (Kullback-Leibler)** : Mesure asymétrique de l'écart entre deux distributions, en nats ou en bits. Pilier de la distillation, du RLHF (pénalité de dérive) et du VAE. [ch. 02](02-mathematiques/README.md)

**DoRA (Weight-Decomposed Low-Rank Adaptation)** : Variante de LoRA qui sépare le poids en une composante de magnitude et une composante de direction. Un peu plus coûteuse, légèrement plus performante sur certaines tâches. [ch. 06](06-fine-tuning/README.md)

**DPO (Direct Preference Optimization)** : Méthode d'alignement qui optimise directement le modèle sur des paires de réponses préférées/rejetées, sans entraîner de modèle de récompense ni faire de RL. Plus simple et plus stable que le RLHF classique. [ch. 05](05-llm/README.md)

**Dropout** : Désactivation aléatoire d'une fraction des neurones pendant l'entraînement, ce qui force le réseau à ne pas dépendre d'un seul chemin. Régularisation classique. [ch. 03](03-reseaux-de-neurones/README.md)

**DVC (Data Version Control)** : Outil de versionnement de données et de modèles, qui stocke des pointeurs dans Git et les fichiers lourds ailleurs. [ch. 12](12-donnees/README.md)

## E

**Échantillonnage (sampling)** : Tirage aléatoire du prochain token selon la distribution de probabilités, éventuellement filtrée (top-k, top-p). Produit des réponses variées, là où le décodage glouton est déterministe. [ch. 05](05-llm/README.md)

**Edge AI (IA embarquée)** : Exécution de modèles directement sur l'appareil (téléphone, caméra, voiture) plutôt que sur un serveur. Avantages : latence, vie privée, mode hors ligne. Contrainte : mémoire et énergie. [ch. 10](10-infrastructure/README.md)

**Elo (score)** : Système de classement par comparaisons successives, utilisé par la Chatbot Arena pour ordonner des modèles jugés par des humains. [ch. 05](05-llm/README.md)

**Embedding (plongement)** : Représentation d'un objet (mot, phrase, image) par un vecteur de nombres réels, tel que les objets proches dans le sens soient proches dans l'espace. Dimensions typiques : 384 à 4 096. [ch. 04](04-transformers/README.md), [ch. 07](07-rag/README.md)

**Emergence (capacités émergentes)** : Apparition, au-delà d'une certaine taille de modèle, de capacités peu visibles sur les petits modèles. Effet réel mais partiellement expliqué par le choix des métriques. [ch. 05](05-llm/README.md)

**Encoder / Decoder** : Encodeur = transforme l'entrée en représentations intermédiaires. Décodeur = produit la sortie. Trois familles : encodeur seul (BERT), décodeur seul (GPT), les deux (T5). [ch. 04](04-transformers/README.md)

**Entraînement (training)** : Phase où les paramètres sont ajustés par descente de gradient sur un jeu d'entraînement. À distinguer de l'inférence. [ch. 01](01-fondamentaux/README.md)

**Entropie (de Shannon)** : Mesure de l'incertitude d'une distribution, en bits ou en nats. Un modèle très sûr de lui a une entropie faible sur sa distribution de sortie. [ch. 02](02-mathematiques/README.md)

**Entropie croisée (cross-entropy)** : Fonction de coût classique en classification et en génération de texte : moins la probabilité donnée à la bonne réponse est élevée, plus la pénalité est forte. [ch. 02](02-mathematiques/README.md)

**Epoch** : Un passage complet sur le jeu d'entraînement. Un LLM fait typiquement moins d'une epoch ; un petit modèle supervisé en fait des dizaines. [ch. 01](01-fondamentaux/README.md)

**Erreur (métrique)** : Ecart moyen entre prédiction et valeur attendue. En régression : RMSE, MAE, MAPE. En classification : taux d'erreur, log-loss. [ch. 01](01-fondamentaux/README.md)

**Étiquetage (labeling)** : Attribution de la réponse attendue. Les étiquettes peuvent être exactes (supervision classique), préférentielles (RLHF) ou faibles (weak supervision). [ch. 12](12-donnees/README.md)

**Eval (évaluation)** : Ensemble des mesures qui déterminent si un système fonctionne : métriques automatiques, tests ciblés, revue humaine, tests adversariaux. Indispensable et presque toujours sous-dimensionnée. [ch. 05](05-llm/README.md)

**Exact match** : Métrique de comparaison stricte entre réponse produite et réponse attendue. Utile pour l'extraction, trompeuse pour la génération libre. [ch. 05](05-llm/README.md)

**Exemple adversarial** : Voir « Adversarial example ». [ch. 13](13-securite/README.md)

**Exfiltration** : Extraction non autorisée de données depuis un système, souvent par un canal détourné (lien Markdown, image distante, appel d'outil). Risque structurel des agents. [ch. 13](13-securite/README.md)

**Explanation (explicabilité)** : Capacité à justifier une décision. Attention à ne pas confondre la plausibilité d'une explication produite par un LLM et son exactitude causale. [ch. 14](14-ethique-societe/README.md)

## F

**FAISS** : Bibliothèque de recherche de similarité sur vecteurs, très rapide, développée par Meta. Pas un serveur : elle s'intègre dans un programme. [ch. 07](07-rag/README.md)

**Faithfulness (fidélité)** : Dans un RAG, proportion des affirmations de la réponse qui sont réellement soutenues par les documents fournis. Métrique centrale pour limiter l'hallucination. [ch. 07](07-rag/README.md)

**Feature (variable)** : Attribut d'entrée utilisé par le modèle. En ML classique, la qualité des features décide du résultat ; en deep learning, le modèle les apprend. [ch. 01](01-fondamentaux/README.md)

**Feature engineering** : Construction manuelle de variables pertinentes à partir des données brutes. Toujours utile, mais remplacée en partie par l'apprentissage de représentations. [ch. 01](01-fondamentaux/README.md)

**Fenêtre de contexte (context window)** : Nombre maximal de tokens qu'un modèle peut prendre en entrée et en sortie combinés. De 8 000 sur les anciens modèles à plus d'un million aujourd'hui. [ch. 05](05-llm/README.md)

**Few-shot** : Fournir quelques exemples dans le prompt pour guider la réponse, sans modifier les poids. Voir aussi zero-shot et many-shot. [ch. 05](05-llm/README.md)

**FFN (Feed-Forward Network)** : Sous-couche du transformer, appliquée indépendamment à chaque position, qui contient souvent la majorité des paramètres du modèle. [ch. 04](04-transformers/README.md)

**Fine-tuning** : Poursuite de l'entraînement d'un modèle pré-entraîné sur des données spécifiques. Peut être complet (tous les poids) ou paramétrique (LoRA). [ch. 06](06-fine-tuning/README.md)

**FlashAttention** : Implémentation d'attention économe en mémoire, qui évite de matérialiser la grande matrice d'attention et limite les transferts mémoire. Gain typique : plusieurs fois moins de VRAM et un calcul plus rapide. [ch. 04](04-transformers/README.md)

**FLOPs (Floating Point Opérations)** : Nombre d'opérations en virgule flottante. Sert à estimer le coût d'un entraînement (de l'ordre de 6 x paramètres x tokens pour un entraînement complet) et d'une inférence (environ 2 x paramètres par token). [ch. 10](10-infrastructure/README.md)

**Fondation (modèle de)** : Modèle de très grande taille entraîné sur des données très générales, destiné à être adapté à de nombreuses tâches. [ch. 05](05-llm/README.md)

**FP8** : Format flottant sur 8 bits, utilisé en entraînement et en inférence sur les GPU récents pour gagner en débit et en mémoire, au prix d'une perte de précision à surveiller. [ch. 10](10-infrastructure/README.md)

**FP16 (demi-précision)** : Flottant sur 16 bits avec 10 bits de mantisse. Divise par deux la mémoire des poids par rapport au FP32, mais nécessite une mise à l'échelle des gradients. [ch. 10](10-infrastructure/README.md)

**FP32 (simple précision)** : Flottant sur 32 bits, référence de précision. Coûteux en mémoire et plus lent : on ne l'utilise plus guère que pour les calculs sensibles et les maîtres-poids. [ch. 10](10-infrastructure/README.md)

**FGSM (Fast Gradient Sign Method)** : Attaque adverse en un seul pas, qui ajoute le signe du gradient multiplié par epsilon. Base pédagogique de l'entraînement adversarial. [ch. 13](13-securite/README.md)

**FSDP (Fully Sharded Data Parallel)** : Stratégie d'entraînement distribué de PyTorch qui découpe paramètres, gradients et états d'optimiseur entre les GPU. Cousin de ZeRO côté NVIDIA. [ch. 10](10-infrastructure/README.md)

**Function calling** : Voir « Tool calling ». [ch. 08](08-agents/README.md)

## G

**GAN (Generative Adversarial Network)** : Deux réseaux en compétition : un générateur produit des échantillons, un discriminateur apprend à les distinguer du réel. Puissant mais instable à entraîner. [ch. 03](03-reseaux-de-neurones/README.md)

**Gate (modèle sous accès contrôlé)** : Modèle dont les poids ne sont téléchargeables qu'après acceptation d'une licence et validation du compte. Fréquent sur les dépôts Hugging Face. [ch. 15](15-ecosysteme/README.md)

**GELU** : Fonction d'activation lisse, très utilisée dans les transformers à la place de ReLU. [ch. 03](03-reseaux-de-neurones/README.md)

**Gemma** : Famille de modèles ouverts publiés par Google, de tailles modestes à moyennes. [ch. 15](15-ecosysteme/README.md)

**Génération augmentée par récupération** : Voir « RAG ». [ch. 07](07-rag/README.md)

**GGUF** : Format de fichier de modèles quantifiés, utilisé par llama.cpp et l'écosystème local (Ollama, LM Studio). Un seul fichier, portable CPU/GPU. [ch. 09](09-inference-optimisation/README.md)

**GNN (Graph Neural Network / réseau de neurones sur graphes)** : Réseau qui propage de l'information le long des arêtes d'un graphe (molécules, réseaux sociaux, dépendances). [ch. 03](03-reseaux-de-neurones/README.md)

**GPT** : Famille de modèles décodeurs autorégressifs de OpenAI. Par extension, désigne souvent tout LLM décodeur. [ch. 04](04-transformers/README.md)

**GPTQ** : Méthode de quantification post-entraînement en 4 bits, calibrée sur un petit échantillon de données. Très répandue pour l'inférence locale. [ch. 09](09-inference-optimisation/README.md)

**GQA (Grouped Query Attention)** : Attention où plusieurs têtes de requête partagent une même tête de clé/valeur, ce qui réduit fortement la taille du KV cache. [ch. 04](04-transformers/README.md)

**Gradient** : Vecteur des dérivées partielles de la fonction de coût par rapport aux paramètres. Il indique la direction de plus forte augmentation : on va donc dans la direction opposée. [ch. 02](02-mathematiques/README.md)

**Gradient checkpointing** : Voir « Activation checkpointing ». [ch. 10](10-infrastructure/README.md)

**Grand modèle de langage** : Voir « LLM ». [ch. 05](05-llm/README.md)

**Graphe de connaissances (knowledge graph)** : Base de faits structurés en entités et relations, utilisée par certains systèmes RAG pour combiner recherche vectorielle et raisonnement relationnel. [ch. 07](07-rag/README.md)

**Grounding (ancrage)** : Faire reposer la réponse du modèle sur des sources vérifiables (documents, résultats d'outils) plutôt que sur sa seule mémoire paramétrique. [ch. 07](07-rag/README.md)

**Guardrail** : Barrière technique qui filtre ou bloque des entrées/sorties dangereuses, ou restreint les actions d'un agent (liste blanche d'outils, sandbox, budget). Utile mais jamais suffisant seul. [ch. 13](13-securite/README.md)

## H

**Hallucination** : Production d'une affirmation fausse présentée avec assurance. Cause structurelle : le modèle optimise la plausibilité du texte suivant, pas la vérité. Se réduit par l'ancrage documentaire, les outils et la vérification, jamais par la seule bonne volonté du prompt. [ch. 05](05-llm/README.md)

**HBM (High Bandwidth Memory)** : Mémoire des accélérateurs de datacenter, empilée et très large en bande passante (plusieurs To/s). C'est elle qui détermine en grande partie la vitesse d'inférence. [ch. 10](10-infrastructure/README.md)

**HellaSwag** : Benchmark de complétion de scénarios courts, longtemps utilisé pour évaluer le bon sens des modèles. Aujourd'hui largement saturé. [ch. 05](05-llm/README.md)

**HNSW (Hierarchical Navigable Small World)** : Structure d'index de graphe permettant une recherche approximative de plus proches voisins en temps quasi logarithmique. Choix par défaut de la plupart des bases vectorielles. [ch. 07](07-rag/README.md)

**Hugging Face** : Plateforme centrale de l'écosystème IA ouverte : dépôt de modèles, de jeux de données, d'espaces de démonstration, et bibliothèques Python associées. [ch. 15](15-ecosysteme/README.md)

**Human-in-the-loop (humain dans la boucle)** : Faire valider une action sensible par une personne avant exécution. Mesure de sécurité la plus efficace pour un agent qui agit sur des systèmes réels. [ch. 08](08-agents/README.md)

**HyDE (Hypothetical Document Embeddings)** : Technique RAG qui demande d'abord au modèle d'écrire un document hypothétique répondant à la question, puis cherche les vrais documents proches de ce texte fictif. [ch. 07](07-rag/README.md)

**Hyperparamètre** : Réglage fixé avant l'entraînement et non appris : learning rate, taille de batch, nombre de couches, dimension des embeddings, rang LoRA. [ch. 01](01-fondamentaux/README.md)

## I

**IA (Intelligence artificielle)** : Ensemble des techniques permettant à une machine de réaliser des tâches qui requièrent habituellement de l'intelligence humaine. Terme parapluie qui couvre le ML, le DL et bien d'autres approches. [ch. 00](00-introduction/README.md)

**IA générative** : Sous-ensemble de l'IA qui produit du contenu nouveau (texte, image, audio, code). Repose aujourd'hui principalement sur les transformers et les modèles de diffusion. [ch. 00](00-introduction/README.md)

**IA symbolique** : Approche historique à base de règles et de logique explicites. Redevient utile pour vérifier ou contraindre les sorties d'un modèle neuronal. [ch. 17](17-au-dela/README.md)

**In-context learning (apprentissage en contexte)** : Capacité d'un modèle à accomplir une tâche à partir d'exemples donnés dans le prompt, sans mise à jour de ses poids. [ch. 05](05-llm/README.md)

**InfiniBand** : Réseau très haut débit et faible latence utilisé pour interconnecter des milliers de GPU dans un cluster d'entraînement. [ch. 10](10-infrastructure/README.md)

**Inférence (inférence)** : Utilisation d'un modèle entraîné pour produire une prédiction ou une génération. Deux phases : prefill (lecture du prompt) puis decode (génération token par token). [ch. 09](09-inference-optimisation/README.md)

**Index (vectoriel)** : Structure de données qui accélère la recherche des vecteurs les plus proches (HNSW, IVF). Son choix se fait sur le compromis entre rappel, latence et mémoire. [ch. 07](07-rag/README.md)

**Init poids (initialisation)** : Valeurs de départ des poids. Une mauvaise initialisation empêche l'apprentissage (activations qui s'éteignent ou explosent) : Xavier pour tanh, He pour ReLU. [ch. 03](03-reseaux-de-neurones/README.md)

**Instruction tuning (ajustement par instructions)** : Fine-tuning supervisé sur des paires instruction/réponse, qui transforme un modèle de base en assistant capable de suivre des consignes. [ch. 05](05-llm/README.md)

**INT4 / INT8** : Formats entiers sur 4 et 8 bits pour stocker poids ou activations quantifiés. Un modèle 7B passe d'environ 14 Go en FP16 à environ 3,5 Go en 4 bits. [ch. 09](09-inference-optimisation/README.md)

**Interprétabilité** : Discipline qui cherche à comprendre le fonctionnement interne d'un modèle : quels neurones, quels circuits, quelles représentations. [ch. 17](17-au-dela/README.md)

**Interprétabilité mécaniste (mechanistic interpretability)** : Branche qui tente de reconstituer les algorithmes implémentés par un réseau, souvent par analyse de circuits et de directions dans l'espace des activations. [ch. 17](17-au-dela/README.md)

**Inversion de modèle (model inversion)** : Attaque qui reconstruit des informations d'entraînement à partir des sorties du modèle, y compris des données personnelles. [ch. 13](13-securite/README.md)

**Isolation (sandbox)** : Exécution d'un agent ou de code généré dans un environnement restreint, sans accès au réseau ni au reste du système. [ch. 13](13-securite/README.md)

**IVF (Inverted File Index)** : Index de recherche approximative qui partitionne l'espace vectoriel en cellules et ne visite que les plus proches. Se combine souvent avec la quantification des vecteurs. [ch. 07](07-rag/README.md)

## J

**Jailbreak** : Contournement des garde-fous d'un modèle par manipulation du prompt (jeu de rôle, encodage, découpage, multilingue). L'alignement étant statistique, aucune parade n'est définitive. [ch. 13](13-securite/README.md)

**JAX** : Bibliothèque de calcul numérique différentiable de Google, très appréciée pour la recherche et la compilation sur TPU. [ch. 15](15-ecosysteme/README.md)

**JSONL (JSON Lines)** : Format d'une ligne JSON par exemple, standard de fait pour les jeux de données d'entraînement et de fine-tuning. [ch. 12](12-donnees/README.md)

**Juge LLM (LLM-as-judge)** : Utiliser un modèle pour noter les réponses d'un autre. Rapide et scalable, mais sensible à la position, à la verbosité et aux biais de préférence. [ch. 05](05-llm/README.md)

## K

**Kappa (coefficient de Cohen)** : Mesure d'accord entre annotateurs au-delà du hasard. En dessous d'environ 0,6, les étiquettes sont trop bruitées pour entraîner un modèle fiable. [ch. 12](12-donnees/README.md)

**Kernel (noyau GPU)** : Fonction exécutée en parallèle sur des milliers de fils. La fusion de kernels est une technique classique d'optimisation : moins de lectures mémoire, plus de débit. [ch. 10](10-infrastructure/README.md)

**Knowledge distillation** : Voir « Distillation ». [ch. 06](06-fine-tuning/README.md)

**KV cache** : Mémoire qui stocke les clés et valeurs déjà calculées pour ne pas les recalculer à chaque nouveau token. Sa taille croît linéairement avec le contexte et le batch : c'est le principal consommateur de VRAM en inférence longue. [ch. 09](09-inference-optimisation/README.md)

## L

**L1 / L2 (régularisation)** : Pénalités ajoutées à la fonction de coût : L1 pousse des poids exactement à zéro (sélection de variables), L2 les rend petits (lissage). [ch. 01](01-fondamentaux/README.md)

**Label (étiquette)** : Réponse attendue associée à une entrée. Voir « Étiquetage ». [ch. 01](01-fondamentaux/README.md)

**LangChain / LangGraph** : Frameworks Python pour enchaîner prompts, outils et états. LangGraph modélise explicitement une boucle d'agent sous forme de graphe. [ch. 08](08-agents/README.md)

**Latence** : Temps de réponse. En chat, deux mesures comptent : le temps jusqu'au premier token et le débit de génération ensuite. [ch. 09](09-inference-optimisation/README.md)

**LayerNorm (normalisation par couche)** : Normalisation des activations sur les dimensions d'une même position. Standard dans les transformers, insensible à la taille du batch. [ch. 03](03-reseaux-de-neurones/README.md)

**Learning rate (taux d'apprentissage)** : Taille du pas de la descente de gradient. Trop grand : divergence. Trop petit : convergence interminable. C'est l'hyperparamètre le plus important. [ch. 01](01-fondamentaux/README.md)

**Loi d'échelle (scaling law)** : Relation empirique entre taille du modèle, volume de données, compute et performance. Sert à planifier un entraînement avant de le lancer. [ch. 05](05-llm/README.md)

**LLaMA** : Famille de modèles ouverts de Meta, très largement déclinée par la communauté. Sa licence autorise l'usage commercial sous conditions. [ch. 15](15-ecosysteme/README.md)

**LLM (Large Language Model / grand modèle de langage)** : Modèle de langage de très grande taille, entraîné à prédire le token suivant sur d'énormes corpus, puis aligné pour suivre des instructions. [ch. 05](05-llm/README.md)

**LLM-as-judge** : Voir « Juge LLM ». [ch. 05](05-llm/README.md)

**LM Studio** : Application de bureau pour télécharger et faire tourner des modèles quantifiés en local, avec une interface graphique. [ch. 15](15-ecosysteme/README.md)

**lm_head (tête de langage)** : Dernière couche qui projette l'état caché du modèle sur le vocabulaire, produisant les logits du prochain token. [ch. 04](04-transformers/README.md)

**Logits** : Scores bruts produits avant normalisation. On les passe dans un softmax pour obtenir des probabilités, ou on les utilise tels quels avec une température. [ch. 02](02-mathematiques/README.md)

**LoRA (Low-Rank Adaptation)** : Fine-tuning paramétrique qui apprend deux petites matrices de rang faible au lieu de modifier les poids d'origine. Coût mémoire divisé par plusieurs ordres de grandeur. [ch. 06](06-fine-tuning/README.md)

**Lost in the middle** : Tendance d'un modèle à mieux exploiter l'information située au début ou à la fin d'un long contexte qu'au milieu. Argument majeur pour un RAG bien trié. [ch. 07](07-rag/README.md)

**LSTM (Long Short-Term Memory)** : Réseau récurrent à portes, conçu pour retenir l'information sur de longues séquences. Supplanté par les transformers pour la plupart des usages. [ch. 03](03-reseaux-de-neurones/README.md)

## M

**MAE (Mean Absolute Error / erreur absolue moyenne)** : Erreur moyenne en valeur absolue, dans l'unité de la cible, robuste aux valeurs extrêmes. [ch. 01](01-fondamentaux/README.md)

**Mamba / SSM (State Space Model)** : Familles d'architectures séquentielles à coût linéaire en longueur, concurrentes des transformers sur les très longs contextes. [ch. 04](04-transformers/README.md)

**Many-shot** : Mettre des dizaines voire des centaines d'exemples dans le prompt. Améliore l'adhérence au format, mais peut aussi faciliter certains contournements de garde-fous. [ch. 05](05-llm/README.md)

**MATH** : Benchmark de problèmes mathématiques de compétition, utilisé pour mesurer le raisonnement des modèles. [ch. 05](05-llm/README.md)

**Matrice de confusion** : Tableau croisant prédictions et vérités pour chaque classe (vrais positifs, faux positifs, faux négatifs, vrais négatifs). Source de toutes les métriques de classification. [ch. 01](01-fondamentaux/README.md)

**MCP (Model Context Protocol)** : Protocole ouvert qui standardisé la façon dont un modèle découvre et appelle des outils et des sources de données, via des serveurs dédiés. [ch. 08](08-agents/README.md)

**Mémoire (d'un agent)** : Stockage d'information au-delà du contexte : résumé de conversation, notes persistantes, base vectorielle de faits. Sans elle, l'agent repart de zéro à chaque tour. [ch. 08](08-agents/README.md)

**Métadonnées** : Attributs attachés à un document ou à un vecteur (date, source, langue, droits) servant à filtrer les recherches. Un filtrage correct vaut souvent mieux qu'un meilleur modèle d'embedding. [ch. 07](07-rag/README.md)

**Métrique** : Mesure chiffrée de performance. Le choix de la métrique définit ce qu'on considère comme « bon » : accuracy sur un jeu déséquilibré récompense le modèle qui prédit toujours la classe majoritaire. [ch. 01](01-fondamentaux/README.md)

**Milvus** : Base de données vectorielle open source orientée très grands volumes, avec index HNSW et IVF. [ch. 07](07-rag/README.md)

**Mini-batch** : Sous-ensemble du batch utilisé pour une étape d'optimisation. Terme employé quand on insiste sur le fait que le batch n'est pas le jeu complet. [ch. 01](01-fondamentaux/README.md)

**Mixture of Experts (MoE)** : Architecture où chaque token n'active qu'une partie des paramètres, via un routeur. Permet un modèle très large à coût d'inférence réduit, au prix d'une VRAM totale élevée. [ch. 04](04-transformers/README.md)

**MLM (Masked Language Modeling)** : Objectif d'entraînement qui consiste à prédire des tokens masqués dans un texte. Utilisé par BERT. [ch. 04](04-transformers/README.md)

**MLOps** : Pratiques d'industrialisation du ML : versionnement, tests, déploiement, surveillance, retour arrière. [ch. 15](15-ecosysteme/README.md)

**MLP (Multi-Layer Perceptron)** : Réseau entièrement connecté à au moins une couche cachée. Brique de base et sous-couche des transformers. [ch. 03](03-reseaux-de-neurones/README.md)

**MMLU (Massive Multitask Language Understanding)** : Benchmark de connaissances générales à choix multiples couvrant des dizaines de disciplines. Très utilisé, aujourd'hui largement saturé par les modèles frontières. [ch. 05](05-llm/README.md)

**Model card (fiche modèle)** : Document décrivant un modèle : données d'entraînement, performances, limites, usages prévus, licence. À lire avant tout déploiement. [ch. 15](15-ecosysteme/README.md)

**Modèle de base (base model)** : Modèle issu du seul pré-entraînement, qui complète du texte mais ne suit pas d'instructions. Il faut le post-entraîner pour obtenir un assistant. [ch. 05](05-llm/README.md)

**Modèle instruct** : Modèle post-entraîné pour suivre des instructions et dialoguer, généralement attendu par les applications grand public. [ch. 05](05-llm/README.md)

**MoE routing** : Mécanisme qui décide, pour chaque token, quels experts activer. Un routage déséquilibré gaspille de la capacité et déstabilise l'entraînement. [ch. 04](04-transformers/README.md)

**Momentum** : Terme d'inertie ajouté à la descente de gradient pour lisser les trajectoires et accélérer dans les directions cohérentes. [ch. 03](03-reseaux-de-neurones/README.md)

**MQA (Multi-Query Attention)** : Attention où toutes les têtes partagent une seule paire de clés/valeurs. Réduit fortement le KV cache, parfois au prix d'une légère baisse de qualité. [ch. 04](04-transformers/README.md)

## N

**Négatif (échantillon)** : Exemple ne correspondant pas à la cible. La qualité des négatifs (négatifs durs, négatifs en batch) gouverne la qualité des modèles contrastifs et des rerankers. [ch. 07](07-rag/README.md)

**Neuro-symbolique** : Approche hybride combinant réseaux de neurones et raisonnement formel, pour obtenir à la fois la souplesse statistique et la garantie logique. [ch. 17](17-au-dela/README.md)

**NF4 (NormalFloat 4 bits)** : Format de quantification 4 bits conçu pour des poids approximativement gaussiens, utilisé par QLoRA. [ch. 06](06-fine-tuning/README.md)

**NLP (Natural Language Processing / traitement automatique du langage)** : Champ qui traite le langage : classification, extraction, traduction, génération. Aujourd'hui largement dominé par les transformers. [ch. 04](04-transformers/README.md)

**NPU (Neural Processing Unit)** : Accélérateur spécialisé pour l'inférence de réseaux de neurones, embarqué dans téléphones et PC récents. Très efficace en énergie, peu polyvalent. [ch. 10](10-infrastructure/README.md)

**NVLink** : Interconnexion très haut débit entre GPU NVIDIA d'un même serveur, indispensable au parallélisme de tenseurs. [ch. 10](10-infrastructure/README.md)

## O

**Offloading** : Déplacer des poids ou un cache vers la RAM ou le disque quand la VRAM ne suffit pas. Permet d'exécuter un modèle trop gros, avec une chute de débit parfois massive. [ch. 09](09-inference-optimisation/README.md)

**Ollama** : Outil qui simplifie le téléchargement et l'exécution locale de modèles quantifiés, avec un serveur compatible API OpenAI. [ch. 15](15-ecosysteme/README.md)

**Oubli catastrophique (catastrophic forgetting)** : Perte des capacités acquises quand on entraîne un modèle sur de nouvelles données sans précaution. Classique en fine-tuning trop agressif. [ch. 06](06-fine-tuning/README.md)

**Out-of-distribution (hors distribution)** : Entrée qui ne ressemble pas à ce que le modèle a vu à l'entraînement. Les performances s'effondrent souvent sans avertissement. [ch. 01](01-fondamentaux/README.md)

**Overfitting** : Voir « Sur-apprentissage ». [ch. 01](01-fondamentaux/README.md)

**OWASP Top 10 for LLM Applications** : Liste de référence des dix risques majeurs des applications à base de LLM : injection de prompt, fuite de données sensibles, composants vulnérables, etc. [ch. 13](13-securite/README.md)

## P

**PagedAttention** : Technique qui gère le KV cache par pages, comme une mémoire virtuelle, réduisant la fragmentation et augmentant fortement le débit de service. Cœur de vLLM. [ch. 09](09-inference-optimisation/README.md)

**Parallélisme de pipeline** : Répartir les couches du modèle sur plusieurs GPU, chaque GPU traitant une étape. Attention aux bulles de pipeline qui réduisent l'occupation. [ch. 10](10-infrastructure/README.md)

**Parallélisme de tenseurs (tensor parallel)** : Découper une même matrice sur plusieurs GPU, qui doivent se synchroniser à chaque couche. Efficace mais gourmand en communication. [ch. 10](10-infrastructure/README.md)

**Parquet** : Format colonne compressé, courant pour stocker de grands jeux de données d'entraînement. [ch. 12](12-donnees/README.md)

**PEFT (Parameter-Efficient Fine-Tuning)** : Famille de méthodes de fine-tuning n'ajustant qu'une petite fraction des paramètres : LoRA, adapters, prefix tuning. [ch. 06](06-fine-tuning/README.md)

**Perceptron** : Plus ancien neurone artificiel : une somme pondérée suivie d'un seuil. Ne peut pas résoudre XOR, ce qui a motivé les réseaux multicouches. [ch. 03](03-reseaux-de-neurones/README.md)

**Perplexité** : Exponentielle de l'entropie croisée, interprétée comme le nombre moyen de choix également plausibles au token suivant. Plus bas vaut mieux, à jeu de test comparable. [ch. 05](05-llm/README.md)

**PGD (Projected Gradient Descent)** : Attaque adverse itérative et puissante, souvent utilisée comme référence pour évaluer la robustesse. Sert aussi d'entraînement adversarial. [ch. 13](13-securite/README.md)

**pgvector** : Extension PostgreSQL ajoutant des types et index vectoriels, pratique quand on veut éviter d'ajouter un service dédié. [ch. 07](07-rag/README.md)

**PII (Personally Identifiable Information / données personnelles)** : Toute donnée permettant d'identifier une personne. À détecter et retirer des corpus avant pré-entraînement, et à protéger au moment de l'inférence. [ch. 12](12-donnees/README.md)

**Pinecone** : Service managé de base vectorielle, sans infrastructure à gérer, facturé à l'usage. [ch. 07](07-rag/README.md)

**Plan-and-exécute** : Stratégie d'agent qui établit d'abord un plan complet, puis exécute chaque étape. Plus stable qu'une boucle réactive, moins adaptative. [ch. 08](08-agents/README.md)

**Poisoning** : Voir « Data poisoning ». [ch. 13](13-securite/README.md)

**Poids (weights)** : Paramètres appris du modèle. En inférence, leur taille en mémoire détermine à elle seule si le modèle tient sur la carte : environ 2 Go par milliard de paramètres en FP16. [ch. 10](10-infrastructure/README.md)

**Politique (policy, RL)** : Fonction qui associe à un état une distribution d'actions. En RLHF, la politique est le LLM lui-même. [ch. 01](01-fondamentaux/README.md)

**Positional encoding (encodage positionnel)** : Information injectée pour que le modèle sache où se trouve chaque token, l'attention seule étant invariante par permutation. [ch. 04](04-transformers/README.md)

**PPO (Proximal Policy Optimization)** : Algorithme de RL utilisé dans le RLHF classique, avec une contrainte de proximité entre l'ancienne et la nouvelle politique. [ch. 05](05-llm/README.md)

**Pré-entraînement (pré-entraînement)** : Entraînement initial, très coûteux, sur un corpus massif, avec l'objectif de prédire le token suivant ou masqué. [ch. 05](05-llm/README.md)

**Précision (précision, en classification)** : Part des positifs prédits qui sont réellement positifs. À distinguer du rappel. [ch. 01](01-fondamentaux/README.md)

**Prefill** : Phase d'inférence qui traite tout le prompt en une passe, parallélisable et limitée par le calcul. Détermine le temps jusqu'au premier token. [ch. 09](09-inference-optimisation/README.md)

**Prefix caching** : Réutilisation du KV cache d'un préfixe commun entre requêtes (system prompt, documents partagés). Gain majeur en production. [ch. 09](09-inference-optimisation/README.md)

**Prompt** : Texte envoyé au modèle. Qualité, structure et longueur du prompt ont un effet direct sur résultat, latence et coût. [ch. 05](05-llm/README.md)

**Prompt engineering** : Conception méthodique des instructions et exemples pour obtenir le comportement voulu, avec tests et itérations plutôt qu'improvisation. [ch. 05](05-llm/README.md)

**Prompt injection** : Insertion d'instructions malveillantes dans les données que le modèle lit (document, page web, email, sortie d'outil), détournant son comportement. Risque principal des systèmes à base d'agents. [ch. 13](13-securite/README.md)

**Pruning (élagage)** : Suppression de poids ou de têtes peu utiles pour réduire taille et coût. Souvent combiné à un réentraînement léger. [ch. 09](09-inference-optimisation/README.md)

**PUE (Power Usage Effectiveness)** : Rapport entre l'énergie totale d'un datacenter et celle consommée par les seuls serveurs. Proche de 1,1 à 1,2 chez les hyperscalers modernes. [ch. 10](10-infrastructure/README.md)

**PyTorch** : Framework d'apprentissage profond dominant en recherche comme en production. [ch. 15](15-ecosysteme/README.md)

## Q

**Q-learning** : Algorithme de RL qui apprend la valeur des couples état/action sans modèle de l'environnement. [ch. 01](01-fondamentaux/README.md)

**Qdrant** : Base de données vectorielle open source écrite en Rust, avec filtrage riche sur les métadonnées. [ch. 07](07-rag/README.md)

**QLoRA** : Fine-tuning LoRA sur un modèle dont les poids sont quantifiés en 4 bits (NF4), avec double quantification et optimiseurs paginés. Rend possible l'adaptation d'un modèle 7B sur une seule carte 16 Go. [ch. 06](06-fine-tuning/README.md)

**Quantization (quantification)** : Réduction du nombre de bits utilisés pour représenter poids et activations. Gagne en mémoire et en vitesse, perd en précision : à mesurer, pas à supposer. [ch. 09](09-inference-optimisation/README.md)

**Quantization-aware training (QAT)** : Entraînement qui simulé la quantification pour que le modèle apprenne à y résister. Meilleure qualité que la quantification post-entraînement, mais plus coûteux. [ch. 09](09-inference-optimisation/README.md)

**Query (requête)** : Voir « Clé / Requête / Valeur », et côté RAG, la formulation envoyée au moteur de recherche. [ch. 04](04-transformers/README.md)

**Qwen** : Famille de modèles ouverts d'Alibaba, déclinée en nombreuses tailles, y compris multimodales et MoE. [ch. 15](15-ecosysteme/README.md)

## R

**RAG (Retrieval-Augmented Génération / génération augmentée par récupération)** : Architecture qui recherche des documents pertinents, les insère dans le prompt, puis demande au modèle de répondre à partir d'eux. Réduit les hallucinations et permet de citer des sources. [ch. 07](07-rag/README.md)

**RAGAS** : Bibliothèque et cadre d'évaluation de RAG, mesurant notamment la fidélité au contexte et la pertinence des documents récupérés. [ch. 07](07-rag/README.md)

**Rappel (recall)** : Part des positifs réels effectivement détectés. Crucial en recherche d'information : un document non récupéré ne pourra jamais être utilisé par le modèle. [ch. 01](01-fondamentaux/README.md), [ch. 07](07-rag/README.md)

**ReAct (Reason + Act)** : Motif d'agent alternant une étape de raisonnement explicite et une action sur un outil, jusqu'à disposer de la réponse. [ch. 08](08-agents/README.md)

**Reasoning model (modèle de raisonnement)** : Modèle entraîné à produire une longue chaîne de raisonnement avant la réponse finale, avec un budget de calcul à l'inférence ajustable. Plus précis sur les tâches difficiles, plus lent et plus cher. [ch. 05](05-llm/README.md)

**Récupération (retrieval)** : Étape qui sélectionne les documents pertinents parmi un corpus indexé. Peut être lexicale, dense ou hybride. [ch. 07](07-rag/README.md)

**Red teaming** : Attaque volontaire et méthodique d'un système pour découvrir ses faiblesses avant qu'un tiers ne le fasse. Inclut la génération de prompts hostiles et l'audit des outils exposés. [ch. 13](13-securite/README.md)

**Régression** : Prédire une valeur continue (prix, température, durée) par opposition à la classification. [ch. 01](01-fondamentaux/README.md)

**Régularisation** : Ensemble des techniques qui pénalisent la complexité pour améliorer la généralisation : L1/L2, dropout, early stopping, augmentation, bruit. [ch. 01](01-fondamentaux/README.md)

**Réinitialisation (relu?)** : Voir « Init poids ». [ch. 03](03-reseaux-de-neurones/README.md)

**Reranking (réordonnancement)** : Réévaluation fine des documents récupérés par un modèle plus coûteux mais plus précis (cross-encoder), afin de ne garder que les meilleurs. [ch. 07](07-rag/README.md)

**Réponse préférée (chosen)** : Dans un jeu de préférences (RLHF, DPO), la réponse jugée meilleure entre deux propositions. [ch. 05](05-llm/README.md)

**Représentation (apprentissage de)** : Fait, pour un réseau profond, d'apprendre lui-même les bonnes variables intermédiaires au lieu de les construire à la main. [ch. 01](01-fondamentaux/README.md)

**Résidu (connexion)** : Ajout de l'entrée d'un bloc à sa sortie, ce qui crée un chemin direct pour le gradient et permet d'empiler des centaines de couches. [ch. 03](03-reseaux-de-neurones/README.md)

**Retrieval** : Voir « Récupération ». [ch. 07](07-rag/README.md)

**Reward model (modèle de récompense)** : Modèle entraîné à prédire la préférence humaine, utilisé comme signal de récompense dans le RLHF. Il est faillible, et un modèle peut apprendre à l'exploiter : c'est le reward hacking. [ch. 05](05-llm/README.md)

**Reward hacking** : Exploitation par le modèle d'une faille du signal de récompense, produisant des réponses qui maximisent la note sans satisfaire l'intention réelle. [ch. 05](05-llm/README.md)

**RGPD (Règlement général sur la protection des données)** : Cadre européen sur les données personnelles : base légale, minimisation, droits des personnes, obligation d'analyse d'impact. S'applique pleinement aux systèmes d'IA. [ch. 14](14-ethique-societe/README.md)

**RLHF (Reinforcement Learning from Human Feedback)** : Alignement en trois temps : collecte de préférences humaines, entraînement d'un modèle de récompense, optimisation du modèle par RL avec une pénalité de dérive. [ch. 05](05-llm/README.md)

**RLVR (RL with Verifiable Rewards)** : Apprentissage par renforcement où la récompense est vérifiable automatiquement (résultat juste, test qui passe). Base de nombreux modèles de raisonnement récents. [ch. 05](05-llm/README.md)

**RMSE (Root Mean Squared Error)** : Racine de l'erreur quadratique moyenne. Pénalise davantage les grandes erreurs que la MAE. [ch. 01](01-fondamentaux/README.md)

**RNN (Recurrent Neural Network)** : Réseau qui traite une séquence pas à pas en conservant un état caché. Coût séquentiel, gradient qui s'évanouit sur les longues dépendances. [ch. 03](03-reseaux-de-neurones/README.md)

**Robustesse (adversariale)** : Capacité d'un modèle à résister à des perturbations conçues pour le tromper. Se mesure par l'attaque la plus forte disponible, jamais par la seule précision propre. [ch. 13](13-securite/README.md)

**RoPE (Rotary Position Embedding)** : Encodage positionnel appliqué par rotation des vecteurs, qui gère bien l'extrapolation au-delà de la longueur vue à l'entraînement. Standard dans les LLM modernes. [ch. 04](04-transformers/README.md)

**Routage (routing)** : Voir « MoE routing ». [ch. 04](04-transformers/README.md)

**RRF (Reciprocal Rank Fusion)** : Méthode simple et robuste pour fusionner plusieurs classements de recherche : on additionne l'inverse du rang de chaque document. [ch. 07](07-rag/README.md)

## S

**Safetensors** : Format de sauvegarde de tenseurs sans exécution de code au chargement, conçu pour remplacer pickle et éviter l'exécution arbitraire. Standard sur Hugging Face. [ch. 13](13-securite/README.md)

**Sandbox** : Voir « Isolation ». [ch. 13](13-securite/README.md)

**Scaling law** : Voir « Loi d'échelle ». [ch. 05](05-llm/README.md)

**Scheduler (de learning rate)** : Programme qui fait varier le taux d'apprentissage au fil de l'entraînement (warmup puis décroissance cosinus ou linéaire). [ch. 03](03-reseaux-de-neurones/README.md)

**Self-attention (auto-attention)** : Attention où requêtes, clés et valeurs proviennent de la même séquence. C'est elle qui permet à chaque token de contextualiser les autres. [ch. 04](04-transformers/README.md)

**Self-RAG** : Variante de RAG où le modèle décide s'il a besoin de chercher, puis critique ses propres réponses au regard des documents récupérés. [ch. 07](07-rag/README.md)

**Sentence-transformers** : Bibliothèque de référence pour produire des embeddings de phrases et de documents, avec de nombreux modèles multilingues prêts à l'emploi. [ch. 07](07-rag/README.md)

**SFT (Supervised Fine-Tuning)** : Fine-tuning supervisé sur des exemples de dialogues ou d'instructions de qualité, première étape du post-entraînement. [ch. 05](05-llm/README.md)

**SGD (Stochastic Gradient Descent)** : Descente de gradient par mini-batchs, où le gradient est estimé sur un échantillon. Le bruit d'estimation est utile : il aide à sortir des minima trop nets. [ch. 01](01-fondamentaux/README.md)

**SigLIP** : Variante de CLIP utilisant une perte sigmoïde par paire plutôt qu'un softmax global, plus efficace à grand nombre d'exemples. [ch. 11](11-multimodal/README.md)

**SLM (Small Language Model)** : Modèle de langage de taille modeste (moins de dix milliards de paramètres), suffisant pour de nombreuses tâches spécialisées et exécutable en local. [ch. 15](15-ecosysteme/README.md)

**Softmax** : Fonction qui transforme un vecteur de scores en distribution de probabilités. Présente en sortie de presque tous les classifieurs. [ch. 02](02-mathematiques/README.md)

**Sous-apprentissage (underfitting)** : Modèle trop simple ou pas assez entraîné : il se trompe même sur les données d'entraînement. Traitement : plus de capacité, plus d'entraînement, moins de régularisation. [ch. 01](01-fondamentaux/README.md)

**Speculative decoding** : Générer plusieurs tokens candidats avec un petit modèle rapide, puis les valider en une passe avec le grand modèle. Accélération typique de 1,5 à 3 fois sans changer la distribution visée. [ch. 09](09-inference-optimisation/README.md)

**Spectral (biais)** : En analyse de Fourier, tendance d'un réseau à apprendre d'abord les composantes de basse fréquence des données, ce qui explique une partie de son comportement en début d'entraînement. [ch. 03](03-reseaux-de-neurones/README.md)

**Sparse (vecteur creux)** : Représentation où très peu de dimensions sont non nulles, typique des sacs de mots et de BM25. [ch. 02](02-mathematiques/README.md)

**Spectrogramme** : Représentation temps-fréquence d'un son, obtenue par transformée de Fourier à court terme. Entrée classique des modèles audio. [ch. 11](11-multimodal/README.md)

**SSL (auto-supervisé)** : Voir « Apprentissage auto-supervisé ». [ch. 01](01-fondamentaux/README.md)

**Supervision faible (weak supervision)** : Utiliser des étiquettes bruitées ou indirectes (règles, heuristiques, modèles) pour entraîner à moindre coût. [ch. 12](12-donnees/README.md)

**Supply chain (chaîne d'approvisionnement)** : Ensemble des composants tiers d'un système d'IA : bibliothèques, poids, adaptateurs, outils, serveurs MCP. Chaque maillon est une surface d'attaque. [ch. 13](13-securite/README.md)

**Sur-apprentissage (overfitting)** : Modèle qui mémorise le jeu d'entraînement et échoue sur des données nouvelles. Se détecte en comparant performance d'entraînement et de validation. [ch. 01](01-fondamentaux/README.md)

**SWE-bench** : Benchmark demandant de résoudre de vraies issues logicielles dans des dépôts existants, avec exécution des tests. Référence pour les agents de code. [ch. 08](08-agents/README.md)

**System prompt (invite système)** : Instructions de haut niveau qui cadrent le comportement du modèle pour toute la conversation. Il peut être extrait par un attaquant : ne jamais y placer de secret. [ch. 13](13-securite/README.md)

## T

**Température** : Paramètre qui aplatit ou aiguise la distribution des tokens avant tirage. Proche de 0 : quasi déterministe. Au-delà de 1 : créatif et imprévisible. [ch. 05](05-llm/README.md)

**Tensor core** : Unité matérielle spécialisée dans les produits de matrices à faible précision, présente depuis les générations Volta/Turing. C'est l'origine des gains de débit des GPU récents. [ch. 10](10-infrastructure/README.md)

**Tensor parallel** : Voir « Parallélisme de tenseurs ». [ch. 10](10-infrastructure/README.md)

**Test-time compute** : Calcul supplémentaire au moment de la réponse (échantillonnages multiples, vérification, recherche). Améliore les tâches de raisonnement, avec un coût par requête plus élevé. [ch. 05](05-llm/README.md)

**TF-IDF (Term Frequency - Inverse Document Frequency)** : Pondération lexicale qui valorise les termes fréquents dans un document mais rares dans le corpus. Ancêtre direct de BM25. [ch. 07](07-rag/README.md)

**Token** : Unité de texte manipulée par le modèle, entre le caractère et le mot. Compte environ 0,75 mot en anglais, souvent moins en français ; l'unité de facturation. [ch. 04](04-transformers/README.md)

**Tokenisation** : Découpage du texte en tokens selon un vocabulaire appris (BPE, WordPiece, SentencePiece, byte-level). [ch. 04](04-transformers/README.md)

**Token spécial** : Symbole réservé du vocabulaire : début et fin de séquence, séparateurs de rôles, marqueurs d'outils. Un template mal construit dégrade la qualité. [ch. 04](04-transformers/README.md)

**Tool calling (appel d'outil)** : Capacité d'un modèle à demander l'exécution d'une fonction décrite par un schéma, puis à exploiter le résultat. Base de tout agent outillé. [ch. 08](08-agents/README.md)

**Top-k** : Ne conserver que les k tokens les plus probables avant tirage. [ch. 05](05-llm/README.md)

**Top-p (nucleus sampling)** : Ne conserver que le plus petit ensemble de tokens totalisant p de probabilité (souvent 0,9 ou 0,95). Souvent préféré au top-k car adaptatif. [ch. 05](05-llm/README.md)

**TPOT / ITL (Time Per Output Token)** : Temps moyen par token généré après le premier. Détermine la fluidité perçue : en dessous de 20 tokens/s, la lecture devient pénible. [ch. 09](09-inference-optimisation/README.md)

**TPU (Tensor Processing Unit)** : Accélérateur conçu par Google pour le calcul de tenseurs, disponible via Google Cloud. Puissant, mais écosystème plus fermé que CUDA. [ch. 10](10-infrastructure/README.md)

**TRANSFORMERS (bibliothèque)** : Bibliothèque Hugging Face qui fournit des implémentations prêtes à l'emploi de milliers de modèles et de leurs tokenizers. [ch. 15](15-ecosysteme/README.md)

**Transformer** : Architecture introduite en 2017, fondée sur l'attention, qui remplace la récurrence par un traitement parallèle de la séquence. Base de tous les LLM actuels. [ch. 04](04-transformers/README.md)

**TRL (Transformer Reinforcement Learning)** : Bibliothèque Hugging Face pour le SFT, le DPO et le RLHF. [ch. 06](06-fine-tuning/README.md)

**TTFT (Time To First Token)** : Temps entre l'envoi de la requête et le premier token reçu. Dominé par la phase prefill et la longueur du prompt. [ch. 09](09-inference-optimisation/README.md)

**Turing (test de)** : Test proposé par Alan Turing en 1950 pour évaluer si une machine peut se faire passer pour un humain dans une conversation écrite. Ne mesure ni la compréhension ni la vérité. [ch. 17](17-au-dela/README.md)

**Typosquatting** : Publication de paquets ou de modèles aux noms très proches de ceux d'origine, pour piéger les téléchargements automatiques. [ch. 13](13-securite/README.md)

## U

**U-Net** : Architecture en encodeur-décodeur avec connexions de saut, très utilisée comme réseau de débruitage dans les modèles de diffusion d'images. [ch. 11](11-multimodal/README.md)

**Underfitting** : Voir « Sous-apprentissage ». [ch. 01](01-fondamentaux/README.md)

**Unified memory (mémoire unifiée)** : Mémoire partagée entre CPU et GPU (Apple Silicon, certains serveurs). Supprimé la copie explicite des tenseurs, mais la bande passante reste inférieure à celle d'une carte dédiée. [ch. 10](10-infrastructure/README.md)

**Utilisateur simulé (simulated user)** : Technique d'évaluation d'agents où un LLM joue le rôle du client ou de l'opérateur, pour tester automatiquement de nombreux scénarios. [ch. 08](08-agents/README.md)

## V

**Valeur (value)** : Voir « Clé / Requête / Valeur ». [ch. 04](04-transformers/README.md)

**Vanishing gradient (gradient qui disparaît)** : Atténuation du gradient en traversant de nombreuses couches ou une longue récurrence, qui empêche les couches profondes d'apprendre. Traité par les résidus, la normalisation et les activations modernes. [ch. 03](03-reseaux-de-neurones/README.md)

**VAE (Variational Autoencoder)** : Autoencodeur probabiliste dont l'espace latent est régularisé par une divergence KL, ce qui permet de générer en échantillonnant. [ch. 03](03-reseaux-de-neurones/README.md)

**Vector database** : Voir « Base de données vectorielle ». [ch. 07](07-rag/README.md)

**Vérification (vérifier)** : Programme qui valide automatiquement une réponse (compilation, tests, calcul formel). Consiste à déléguer la vérité au code plutôt qu'au modèle, démarche centrale des modèles de raisonnement récents. [ch. 05](05-llm/README.md)

**VLM (Vision-Language Model)** : Modèle qui traite simultanément images et texte : décrire, extraire, comparer, raisonner sur une image. [ch. 11](11-multimodal/README.md)

**ViT (Vision Transformer)** : Transformer appliqué aux images, qui découpe l'image en patchs traités comme des tokens. [ch. 11](11-multimodal/README.md)

**vLLM** : Moteur d'inférence à haut débit, connu pour PagedAttention et le batching continu. Référence pour servir un LLM en production sur GPU. [ch. 09](09-inference-optimisation/README.md)

**Vocabulaire (vocab)** : Ensemble fini de tokens connus d'un modèle. Typiquement 32 000 à 250 000 entrées ; un mot inconnu est découpé en sous-mots. [ch. 04](04-transformers/README.md)

**VRAM (vidéo RAM)** : Mémoire de la carte graphique. En pratique, c'est la contrainte dure : elle doit contenir les poids, le KV cache, les activations et le batch. [ch. 10](10-infrastructure/README.md)

**Warmup** : Phase initiale où le learning rate croît progressivement, ce qui évite de casser un modèle au démarrage de l'entraînement. [ch. 03](03-reseaux-de-neurones/README.md)

## W

**Wav2Vec** : Famille de modèles de représentation de la parole, entraînés de façon auto-supervisée sur de l'audio brut. [ch. 11](11-multimodal/README.md)

**Weaviate** : Base de données vectorielle open source avec schéma et filtrage structuré. [ch. 07](07-rag/README.md)

**WebDataset** : Format de stockage de grands jeux de données sous forme d'archives tar contenant des échantillons, optimisé pour le streaming séquentiel depuis le disque. [ch. 12](12-donnees/README.md)

**Weight decay** : Voir « L1 / L2 (régularisation) » ; en pratique, décroissance multiplicative des poids à chaque pas, découplée du gradient dans AdamW. [ch. 03](03-reseaux-de-neurones/README.md)

**Whisper** : Modèle de reconnaissance de la parole multilingue de OpenAI, référence de l'ASR open weights. [ch. 11](11-multimodal/README.md)

**WordPiece** : Algorithme de tokenisation de la famille BERT, proche de BPE mais avec un critère de vraisemblance. [ch. 04](04-transformers/README.md)

**World model (modèle du monde)** : Modèle qui apprend une représentation dynamique d'un environnement et permet de simuler les conséquences d'actions, pour planifier ou entraîner un agent. [ch. 17](17-au-dela/README.md)

## X

**XAI (eXplainable AI / IA explicable)** : Ensemble des méthodes visant à rendre compréhensible une décision de modèle : importance des variables, cartes de saillance, exemples contrefactuels. [ch. 14](14-ethique-societe/README.md)

## Y

**Yi** : Famille de modèles de langage ouverts publiés par 01.AI, disponibles en plusieurs tailles et en variante MoE. [ch. 15](15-ecosysteme/README.md)

## Z

**ZeRO (Zero Redundancy Optimizer)** : Technique de répartition des états d'optimiseur, des gradients puis des paramètres entre les GPU, pour entraîner de très grands modèles sans dupliquer l'état complet sur chaque carte. [ch. 10](10-infrastructure/README.md)

**Zero-shot** : Demander une tâche sans fournir aucun exemple, uniquement avec une consigne. Marche d'autant mieux que le modèle est grand et bien aligné. [ch. 05](05-llm/README.md)

## Acronymes

| Sigle | Anglais | Français |
|---|---|---|
| AGI | Artificial General Intelligence | intelligence artificielle générale |
| AI Act | Artificial Intelligence Act | règlement européen sur l'intelligence artificielle |
| ALiBi | Attention with Linear Biases | attention à biais linéaires |
| ASR | Automatic Speech Recognition | reconnaissance automatique de la parole |
| AWQ | Activation-aware Weight Quantization | quantification des poids sensible aux activations |
| BF16 | bfloat16 | flottant 16 bits à large exposant |
| BM25 | Best Matching 25 | fonction de classement lexical (Okapi BM25) |
| BPE | Byte Pair Encoding | encodage par paires d'octets |
| CFG | Classifier-Free Guidance | guidage sans classifieur |
| CLIP | Contrastive Language-Image Pré-entraînement | pré-entraînement contrastif texte-image |
| CNN | Convolutional Neural Network | réseau de neurones convolutif |
| CPU | Central Processing Unit | processeur central |
| CUDA | Compute Unified Device Architecture | architecture de calcul unifiée pour GPU NVIDIA |
| DPO | Direct Preference Optimization | optimisation directe sur les préférences |
| DVC | Data Version Control | versionnement des données |
| FFN | Feed-Forward Network | réseau à propagation avant |
| FGSM | Fast Gradient Sign Method | méthode rapide par signe du gradient |
| FLOP | Floating Point Operation | opération en virgule flottante |
| FP16 / FP32 / FP8 | Floating Point 16 / 32 / 8 bits | flottant 16 / 32 / 8 bits |
| FSDP | Fully Sharded Data Parallel | parallélisme de données entièrement partitionné |
| GAN | Generative Adversarial Network | réseau antagoniste génératif |
| GELU | Gaussian Error Linear Unit | unité linéaire à erreur gaussienne |
| GNN | Graph Neural Network | réseau de neurones sur graphes |
| GPU | Graphics Processing Unit | processeur graphique |
| GQA | Grouped Query Attention | attention à requêtes groupées |
| GPT | Generative Pre-trained Transformer | transformer génératif pré-entraîné |
| HBM | High Bandwidth Memory | mémoire à large bande passante |
| HNSW | Hierarchical Navigable Small World | petit monde navigable hiérarchique (index) |
| HyDE | Hypothetical Document Embeddings | embeddings de documents hypothétiques |
| ITL | Inter-Token Latency | latence entre tokens |
| IVE / IVF | Inverted File Index | index de fichiers inversés |
| JSONL | JSON Lines | JSON par ligne |
| KL | Kullback-Leibler (divergence) | divergence de Kullback-Leibler |
| KV cache | Key-Value cache | cache des clés et valeurs |
| LLM | Large Language Model | grand modèle de langage |
| LoRA | Low-Rank Adaptation | adaptation de rang faible |
| LSTM | Long Short-Term Memory | mémoire longue et court terme |
| MAE | Mean Absolute Error | erreur absolue moyenne |
| MCP | Model Context Protocol | protocole de contexte de modèle |
| ML | Machine Learning | apprentissage automatique |
| MLM | Masked Language Modeling | modélisation de langue masquée |
| MLOps | Machine Learning Opérations | industrialisation de l'apprentissage automatique |
| MLP | Multi-Layer Perceptron | perceptron multicouche |
| MMLU | Massive Multitask Language Understanding | compréhension linguistique multitâche massive |
| MoE | Mixture of Experts | mélange d'experts |
| MQA | Multi-Query Attention | attention à requêtes multiples |
| NF4 | NormalFloat 4 bits | format NormalFloat 4 bits |
| NLP | Natural Language Processing | traitement automatique du langage naturel |
| NPU | Neural Processing Unit | unité de calcul neuronal |
| OWASP | Open Worldwide Application Security Project | projet ouvert de sécurité applicative |
| PEFT | Parameter-Efficient Fine-Tuning | ajustement efficace en paramètres |
| PGD | Projected Gradient Descent | descente de gradient projetée |
| PII | Personally Identifiable Information | données à caractère personnel |
| PPO | Proximal Policy Optimization | optimisation de politique proximale |
| PUE | Power Usage Effectiveness | efficacité énergétique d'un datacenter |
| QAT | Quantization-Aware Training | entraînement conscient de la quantification |
| QLoRA | Quantized LoRA | LoRA sur modèle quantifié |
| RAG | Retrieval-Augmented Génération | génération augmentée par récupération |
| ReAct | Reasoning and Acting | raisonner puis agir |
| RGPD | General Data Protection Regulation | règlement général sur la protection des données |
| RL | Reinforcement Learning | apprentissage par renforcement |
| RLHF | Reinforcement Learning from Human Feedback | renforcement à partir du retour humain |
| RLVR | Reinforcement Learning with Verifiable Rewards | renforcement à récompenses vérifiables |
| RMSE | Root Mean Squared Error | racine de l'erreur quadratique moyenne |
| RNN | Recurrent Neural Network | réseau de neurones récurrent |
| RoPE | Rotary Position Embedding | encodage positionnel rotatif |
| RRF | Reciprocal Rank Fusion | fusion par rang réciproque |
| SGD | Stochastic Gradient Descent | descente de gradient stochastique |
| SFT | Supervised Fine-Tuning | ajustement supervisé |
| SigLIP | Sigmoid Loss for Language-Image Pré-entraînement | perte sigmoïde pour le pré-entraînement texte-image |
| SLM | Small Language Model | petit modèle de langage |
| SSM | State Space Model | modèle en espace d'état |
| SWE-bench | Software Engineering Benchmark | banc d'essai en génie logiciel |
| TF-IDF | Term Frequency - Inverse Document Frequency | fréquence du terme - fréquence inverse du document |
| TPOT | Time Per Output Token | temps par token produit |
| TPU | Tensor Processing Unit | unité de traitement de tenseurs |
| TTFT | Time To First Token | temps jusqu'au premier token |
| TTS | Text To Speech | synthèse vocale |
| VAE | Variational Autoencoder | autoencodeur variationnel |
| ViT | Vision Transformer | transformer de vision |
| VLM | Vision-Language Model | modèle vision-langage |
| VRAM | Video Random Access Memory | mémoire vidéo |
| XAI | eXplainable Artificial Intelligence | intelligence artificielle explicable |
| ZeRO | Zero Redundancy Optimizer | optimiseur sans redondance |

---

Ce glossaire comporte 365 entrées, plus cette table d'acronymes.

Il est vivant : si un terme manque, c'est un ajout à proposer (voir
[CONTRIBUTING.md](CONTRIBUTING.md)).

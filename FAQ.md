# Questions fréquentes

Ce document rassemble les réponses aux questions les plus fréquentes sur l'intelligence artificielle, l'apprentissage profond et les grands modèles de langage. Les questions sont regroupées par thèmes pour faciliter la navigation.

---

## 1. Débutant

### Q1.1 : Qu'est-ce que l'apprentissage automatique (Machine Learning) ?
L'apprentissage automatique est une branche de l'intelligence artificielle où les ordinateurs apprennent à partir de données sans être explicitement programmés pour chaque tâche. Il s'appuie sur des algorithmes pour détecter des motifs. Voir [chapitre 01](01-fondamentaux/README.md).

### Q1.2 : Quelle est la différence entre l'IA, le Machine Learning et le Deep Learning ?
L'IA est le domaine général. Le Machine Learning en est un sous-ensemble basé sur l'apprentissage par les données. Le Deep Learning est un sous-ensemble du Machine Learning utilisant des réseaux de neurones profonds. Voir [chapitre 01](01-fondamentaux/README.md).

### Q1.3 : Qu'est-ce qu'un réseau de neurones artificiels ?
C'est un modèle de calcul inspiré du cerveau humain, composé de couches de "neurones" interconnectés qui s'activent et ajustent leurs connexions pour apprendre des relations complexes. Voir [chapitre 03](03-reseaux-de-neurones/README.md).

### Q1.4 : Qu'est-ce qu'un "token" dans le contexte des LLM ?
Un token est l'unité de base de texte traitée par un modèle. Il peut s'agir d'un mot entier, d'une partie de mot ou d'un caractère. En moyenne, 100 tokens correspondent à environ 75 mots en anglais. Voir [chapitre 04](04-transformers/README.md).

### Q1.5 : Qu'est-ce qu'un embedding (plongement lexical) ?
Un embedding est la représentation d'un mot ou d'une phrase sous forme de vecteur numérique dans un espace à plusieurs dimensions, où la proximité géométrique indique une proximité sémantique. Voir [chapitre 04](04-transformers/README.md).

### Q1.6 : Quel langage de programmation dois-je apprendre pour l'IA ?
Python est le langage standard de l'industrie, soutenu par un vaste écosystème de bibliothèques comme PyTorch, TensorFlow et Hugging Face. Voir [chapitre 15](15-ecosysteme/README.md).

---

## 2. LLM et usage

### Q2.1 : Comment fonctionne l'attention dans un Transformer ?
Le mécanisme d'attention permet au modèle de peser l'importance relative de chaque mot par rapport aux autres dans une phrase, indépendamment de leur distance, pour mieux capturer le contexte. Voir [chapitre 04](04-transformers/README.md).

### Q2.2 : Qu'est-ce que la "température" lors de la génération de texte ?
La température contrôle l'aléa de la génération. Une valeur basse (ex: 0.2) produit un texte plus déterministe et prévisible, tandis qu'une valeur haute (ex: 0.8) encourage la créativité et la diversité. Voir [chapitre 05](05-llm/README.md).

### Q2.3 : Qu'est-ce que la fenêtre de contexte d'un LLM ?
C'est la quantité maximale de texte (en tokens) que le modèle peut traiter en une seule fois (entrée et sortie comprises). Elle varie de 4 000 tokens à plus d'un million pour certains modèles récents. Voir [chapitre 05](05-llm/README.md).

### Q2.4 : Pourquoi les modèles d'IA "hallucinent-ils" ?
Les hallucinations surviennent car les modèles prédisent le mot suivant le plus probable statistiquement, sans avoir de notion de vérité factuelle ou de validation externe. Voir [chapitre 05](05-llm/README.md).

### Q2.5 : Quelle est la différence entre un modèle de base et un modèle d'instructions ?
Le modèle de base est entraîné à prédire le mot suivant sur du texte brut. Le modèle d'instructions a subi un post-entraînement (RLHF/DPO) pour répondre poliment et précisément aux requêtes. Voir [chapitre 05](05-llm/README.md).

### Q2.6 : Comment les modèles de raisonnement (comme o1 ou DeepSeek-R1) fonctionnent-ils ?
Ces modèles utilisent des techniques de "chaîne de pensée" (Chain of Thought) internes, générant des étapes de réflexion cachées ou visibles avant de formuler leur réponse finale. Voir [chapitre 05](05-llm/README.md).

---

## 3. Entraînement et fine-tuning

### Q3.1 : Quand faut-il fine-tuner un modèle plutôt que de faire du RAG ?
Le fine-tuning est préférable pour changer le style, le format de sortie ou enseigner une syntaxe spécifique. Le RAG est idéal pour fournir des connaissances fraîches et factuelles sans modifier les poids. Voir [chapitre 06](06-fine-tuning/README.md) et [chapitre 07](07-rag/README.md).

### Q3.2 : Qu'est-ce que la méthode LoRA (Low-Rank Adaptation) ?
LoRA est une technique de fine-tuning efficace qui fige les poids d'origine et insère de petites matrices de rang faible entraînables, réduisant drastiquement les besoins en mémoire et en calcul. Voir [chapitre 06](06-fine-tuning/README.md).

### Q3.3 : Qu'est-ce que l'apprentissage par renforcement à partir des commentaires humains (RLHF) ?
Le RLHF ajuste un modèle de langage en entraînant un modèle de récompense basé sur des préférences humaines, puis en optimisant le LLM via PPO ou d'autres algorithmes d'apprentissage par renforcement. Voir [chapitre 05](05-llm/README.md).

### Q3.4 : Qu'est-ce que l'alignement par préférence directe (DPO) ?
Le DPO est une alternative simplifiée au RLHF qui optimise directement le modèle de langage sur des paires de réponses préférées et rejetées sans avoir besoin d'entraîner un modèle de récompense séparé. Voir [chapitre 05](05-llm/README.md).

### Q3.5 : Qu'est-ce que l'overfitting (surapprentissage) ?
L'overfitting se produit lorsqu'un modèle apprend par cœur les données d'entraînement et perd sa capacité à généraliser sur de nouvelles données non vues. On le détecte quand la perte de validation remonte. Voir [chapitre 01](01-fondamentaux/README.md).

### Q3.6 : Qu'est-ce que la distillation de modèle ?
C'est le processus consistant à entraîner un modèle plus petit ("élève") à reproduire le comportement et les prédictions d'un modèle plus grand et plus performant ("enseignant"). Voir [chapitre 06](06-fine-tuning/README.md).

---

## 4. Matériel et coûts

### Q4.1 : De combien de VRAM ai-je besoin pour faire tourner un modèle de 7 milliards de paramètres ?
En précision 16-bit, il faut environ 14 Go de VRAM. Avec une quantification en 4-bit (ex: INT4 ou Q4_K_M), la VRAM requise descend à environ 5 à 6 Go, ce qui permet de le faire tourner sur un GPU grand public. Voir [chapitre 10](10-infrastructure/README.md).

### Q4.2 : Quelle est la différence entre un GPU et un TPU ?
Un GPU est un processeur graphique généraliste très efficace pour le calcul parallèle. Un TPU (Tensor Processing Unit) est un circuit intégré propre à Google, hautement optimisé pour les opérations sur les tenseurs de l'IA. Voir [chapitre 10](10-infrastructure/README.md).

### Q4.3 : Combien coûte l'entraînement d'un grand modèle de langage ?
L'entraînement d'un modèle de pointe (comme Llama 3 70B) peut coûter plusieurs millions de dollars en temps de calcul GPU, tandis qu'un fine-tuning LoRA local sur un petit modèle coûte quelques centimes à quelques dizaines d'euros. Voir [chapitre 10](10-infrastructure/README.md).

### Q4.4 : Pourquoi la précision FP16 ou BF16 est-elle préférée au FP32 pour l'entraînement ?
Les formats 16-bit divisent par deux l'empreinte mémoire et accélèrent les calculs sur les architectures modernes (comme les Tensor Cores de NVIDIA) avec une perte de précision négligeable. Voir [chapitre 10](10-infrastructure/README.md).

### Q4.5 : Qu'est-ce que la quantification de modèle ?
La quantification consiste à convertir les poids d'un modèle d'une précision élevée (ex: FP16) vers une précision inférieure (ex: INT4, GGUF) afin de réduire la consommation mémoire et d'accélérer l'inférence. Voir [chapitre 06](06-fine-tuning/README.md).

### Q4.6 : Quel est l'impact environnemental de l'IA générative ?
L'entraînement et l'inférence des modèles consomment beaucoup d'électricité et d'eau (pour le refroidissement des centres de données). L'utilisation de modèles plus petits et d'architectures optimisées aide à réduire cette empreinte. Voir [chapitre 14](14-ethique-societe/README.md).

---

## 5. RAG (Retrieval-Augmented Generation)

### Q5.1 : Quel est le principe fondamental du RAG ?
Le RAG consiste à récupérer des documents pertinents dans une base de données externe en réponse à une requête, puis à insérer ces documents dans le prompt du LLM pour qu'il formule une réponse étayée et à jour. Voir [chapitre 07](07-rag/README.md).

### Q5.2 : Qu'est-ce que le "chunking" dans un pipeline RAG ?
Le chunking est le découpage d'un long document en fragments plus petits (ex: 500 caractères avec un recouvrement de 10%) pour s'assurer que les informations clés tiennent dans la fenêtre de contexte et restent ciblées. Voir [chapitre 07](07-rag/README.md).

### Q5.3 : Pourquoi utiliser une base de données vectorielle pour le RAG ?
Les bases de données vectorielles (comme Chroma, FAISS ou Pinecone) permettent d'effectuer des recherches de similarité sémantique ultra-rapides sur des millions de vecteurs (embeddings) de documents. Voir [chapitre 07](07-rag/README.md).

### Q5.4 : Qu'est-ce que le "reranking" (réordonnancement) ?
Le reranking consiste à réévaluer les documents retournés par une première recherche rapide à l'aide d'un modèle plus lent mais plus précis (un cross-encoder), afin de maximiser la pertinence des résultats insérés dans le prompt. Voir [chapitre 07](07-rag/README.md).

### Q5.5 : Comment évaluer la qualité d'un système RAG ?
On utilise des frameworks comme Ragas pour mesurer la fidélité de la réponse au contexte (faithfulness), la pertinence de la réponse à la question (answer relevance) et la pertinence du contexte récupéré (context recall). Voir [chapitre 07](07-rag/README.md).

### Q5.6 : Comment gérer les mises à jour de données dans un RAG ?
Il suffit de mettre à jour les documents et leurs embeddings dans la base vectorielle. Contrairement au fine-tuning, aucune ré-entraînement du modèle de langage n'est nécessaire pour actualiser ses connaissances. Voir [chapitre 07](07-rag/README.md).

---

## 6. Agents

### Q6.1 : Qu'est-ce qu'un agent autonome en IA ?
Un agent est un système qui utilise un LLM comme moteur de décision pour planifier des tâches, appeler des outils externes (calculatrice, recherche web, exécution de code) et réagir aux retours de son environnement. Voir [chapitre 08](08-agents/README.md).

### Q6.2 : Qu'est-ce que l'architecture ReAct (Reasoning and Acting) ?
ReAct combine le raisonnement ("pensée") et l'action (appel d'outil) dans une boucle fermée : l'agent pense à ce qu'il doit faire, exécute une action, observe le résultat, et réitère jusqu'à la résolution du problème. Voir [chapitre 08](08-agents/README.md).

### Q6.3 : Qu'est-ce que le protocole MCP (Model Context Protocol) ?
Le protocole MCP est une norme ouverte qui permet d'exposer de manière sécurisée des données, des outils et du contexte aux LLM et aux agents via des serveurs et clients interopérables. Voir [chapitre 08](08-agents/README.md).

### Q6.4 : Comment donner de la mémoire à un agent ?
On peut stocker l'historique des conversations dans une base de données, utiliser un résumé des échanges passés inséré dans le prompt, ou stocker des faits clés sous forme d'embeddings pour une récupération sémantique. Voir [chapitre 08](08-agents/README.md).

### Q6.5 : Qu'est-ce qu'un système multi-agents ?
C'est une architecture où plusieurs agents spécialisés (ex: un rédacteur, un relecteur, un développeur) collaborent, s'envoient des messages et valident mutuellement leur travail pour résoudre un problème complexe. Voir [chapitre 08](08-agents/README.md).

### Q6.6 : Quels sont les risques liés aux agents autonomes ?
Les risques incluent les boucles infinies de requêtes (génératrices de coûts), l'exécution incontrôlée de commandes système dangereuses, ou la manipulation par injection de prompts indirecte. Voir [chapitre 08](08-agents/README.md) et [chapitre 13](13-securite/README.md).

---

## 7. Métier et carrière

### Q7.1 : Quel est le rôle d'un ingénieur en IA (AI Engineer) ?
L'ingénieur en IA intègre des modèles de langage et d'apprentissage profond dans des applications concrètes, conçoit des architectures de données (RAG, agents) et optimise l'inférence des modèles. Voir [chapitre 15](15-ecosysteme/README.md).

### Q7.2 : Quelle est la différence entre un Data Scientist et un AI Engineer ?
Le Data Scientist se concentre souvent sur l'analyse de données, les statistiques et l'entraînement de modèles prédictifs classiques. L'AI Engineer est plus orienté développement logiciel et intégration d'API d'IA générative. Voir [chapitre 15](15-ecosysteme/README.md).

### Q7.3 : Faut-il un doctorat (PhD) pour travailler dans l'IA ?
Un doctorat est précieux pour la recherche fondamentale (ex: concevoir de nouvelles architectures chez OpenAI ou Meta), mais un profil d'ingénieur ou d'autodidacte avec un bon portfolio est suffisant pour la majorité des rôles appliqués. Voir [chapitre 15](15-ecosysteme/README.md).

### Q7.4 : Qu'est-ce qu'un portfolio d'IA et comment en construire un ?
Un portfolio d'IA regroupe des projets personnels documentés (ex: un bot RAG local, un outil d'automatisation avec agents, un fine-tuning de modèle). Publier le code sur GitHub et rédiger des explications claires est la meilleure approche. Voir [chapitre 15](15-ecosysteme/README.md) et le parcours de [progression](PROGRESSION.md).

### Q7.5 : Comment rester à jour dans le domaine de l'IA qui évolue très vite ?
Suivre des newsletters de référence (ex: TLDR AI, Batch), lire les résumés de papiers de recherche sur arXiv, participer à des communautés Discord actives et expérimenter régulièrement de nouveaux outils. Voir [chapitre 15](15-ecosysteme/README.md).

### Q7.6 : Les outils de génération de code vont-ils remplacer les développeurs ?
Non, ces outils agissent comme des multiplicateurs de productivité. Ils gèrent la syntaxe et les tâches répétitives, mais l'architecture, la sécurité, la logique métier complexe et la validation restent du ressort de l'humain. Voir [chapitre 14](14-ethique-societe/README.md).

---

## 8. Sécurité et éthique

### Q8.1 : Qu'est-ce qu'une injection de prompt (Prompt Injection) ?
C'est une attaque où un utilisateur insère des instructions malveillantes dans le prompt d'un LLM pour contourner ses filtres de sécurité et lui faire exécuter des actions non autorisées. Voir [chapitre 13](13-securite/README.md).

### Q8.2 : Qu'est-ce que le "Jailbreaking" ?
Le jailbreaking consiste à concevoir un prompt spécifique qui pousse le LLM à ignorer ses règles de modération et d'éthique pour générer du contenu illégal, dangereux ou inapproprié. Voir [chapitre 13](13-securite/README.md).

### Q8.3 : Qu'est-ce que l'alignement des modèles d'IA ?
L'alignement désigne l'ensemble des techniques visant à s'assurer que le comportement d'un modèle d'IA est conforme aux valeurs humaines, à l'éthique et aux intentions de ses concepteurs (HGG : Harmless, Honest, Helpful). Voir [chapitre 14](14-ethique-societe/README.md).

### Q8.4 : Qu'est-ce que l'AI Act de l'Union européenne ?
L'AI Act est le premier cadre réglementaire complet sur l'IA au monde. Il classifie les applications d'IA selon leur niveau de risque (minimal, limité, élevé, inacceptable) et impose des obligations strictes de transparence et de gouvernance. Voir [chapitre 14](14-ethique-societe/README.md).

### Q8.5 : Comment les biais s'introduisent-ils dans les modèles d'IA ?
Les biais proviennent principalement des données d'entraînement (textes historiques contenant des stéréotypes, déséquilibres démographiques, etc.). Le modèle reproduit et amplifie ces motifs statistiques lors de sa génération. Voir [chapitre 14](14-ethique-societe/README.md).

### Q8.6 : Qu'est-ce que le "red teaming" en IA ?
Le red teaming consiste à simuler des attaques et à tester de manière agressive un modèle ou une application d'IA pour identifier ses failles de sécurité, ses vulnérabilités aux injections de prompts et ses comportements indésirables avant son déploiement. Voir [chapitre 13](13-securite/README.md).
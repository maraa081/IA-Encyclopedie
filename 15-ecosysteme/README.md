# Écosystème, outils et acteurs

> Explorer le paysage dynamique de l'intelligence artificielle : les frameworks, les bibliothèques, les fournisseurs d'API, les modèles ouverts et fermés, et les outils qui permettent d'entraîner, de servir et de monitorer des systèmes à grande échelle.

## 1. Le paysage des acteurs

Le secteur de l'IA est structuré par des acteurs aux modèles économiques et aux philosophies très différents.
On peut les classer en quatre grandes catégories :

*   **Les géants du cloud (hyperscalers)** :
    Microsoft (Azure, partenaire d'OpenAI), Google (GCP, DeepMind, Gemini), Amazon (AWS, Bedrock).
    Ils fournissent l'infrastructure matérielle (GPU, TPU) et les services cloud nécessaires à l'entraînement et au service de modèles massifs.
*   **Les laboratoires indépendants** :
    OpenAI, Anthropic, Mistral AI, Cohere, xAI.
    Leur cœur de métier est la recherche et la création de modèles frontières, souvent propriétaires, accessibles via API.
*   **Les champions de l'open source** :
    Meta (Llama), Mistral AI (Mistral, Mixtral), Alibaba (Qwen), DeepSeek, Google (Gemma), Microsoft (Phi), Allen Institute (OLMo).
    Ils publient les poids de leurs modèles, ce qui permet à la communauté de les inspecter, de les affiner et de les déployer localement.
*   **Les fournisseurs d'outillage** :
    NVIDIA (matériel et écosystème CUDA), Hugging Face (hub, bibliothèques), Weights & Biases (suivi d'expériences), LangChain (orchestration).

La tension entre modèles fermés et modèles ouverts est le moteur principal de l'innovation actuelle.
Les modèles fermés sont souvent en avance sur les benchmarks généraux, mais les modèles ouverts comblent l'écart avec quelques mois de retard, tout en offrant le contrôle et la confidentialité.

## 2. Frameworks de deep learning

Les frameworks sont les bibliothèques logicielles qui abstraient la complexité des calculs tensoriels et de la rétropropagation (le calcul automatique des gradients).

*   **PyTorch** (Meta) :
    C'est aujourd'hui le standard de fait dans la recherche et de plus en plus dans l'industrie.
    Son approche dynamique (define-by-run) construit le graphe de calcul à la volée, ce qui facilité le débogage étape par étape.
    L'écosystème est immense et presque tous les modèles publiés sur Hugging Face fournissent un code PyTorch.
*   **JAX** (Google) :
    Orienté vers le calcul haute performance, il compile les opérations vers le GPU ou le TPU via XLA.
    Il repose sur la programmation fonctionnelle (fonctions pures, immutabilité).
    Il est très apprécié pour les entraînements massifs et la recherche sur les architectures, mais sa courbe d'apprentissage est plus raide.
*   **TensorFlow / Keras** (Google) :
    Historiquement dominant, surtout en production industrielle grâce à ses outils de déploiement (TFX, TF Serving).
    Keras fournit une API de haut niveau très lisible.
    Il est toutefois en net recul face à la flexibilité de PyTorch pour la recherche.

Quand utiliser quoi ?
Utilise PyTorch par défaut, car c'est ce que la communauté publie.
Choisis JAX si tu entraînes sur des TPU ou si tu veux maximiser le débit de calcul sur un cluster homogène.
N'utilise TensorFlow que si tu dois intégrer un système existant qui en dépend déjà.

## 3. Les bibliothèques de l'écosystème

L'écosystème Python de l'IA se compose de briques spécialisées qui se composent bien.

| Bibliothèque | Rôle principal | Points clés |
|---|---|---|
| `transformers` | Charger et exécuter des modèles pré-entraînés | Supporte LLM, vision, audio ; API unifiée |
| `datasets` | Charger et manipuler des jeux de données | Streaming, cache mémoire, formats mémoire-mappés |
| `tokenizers` | Découper le texte en tokens | Très rapide (Rust), entraînement BPE/WordPiece |
| `accelerate` | Entraîner sur CPU/GPU/TPU multi-nœuds | Absorbe la complexité de la distribution |
| `peft` | Fine-tuning paramétrique efficace | LoRA, QLoRA, préfix-tuning |
| `trl` | Post-entraînement par RLHF/DPO | Implémentations de référence SFT, PPO, DPO |
| `vLLM` | Servir des LLM à haut débit | PagedAttention, batching continu |
| `bitsandbytes` | Quantification 8 et 4 bits | Rend possible le chargement de gros modèles |
| `timm` | Modèles de vision état de l'art | Collection de ViT, ResNet, ConvNeXt |
| `torchvision` | Datasets et transformations d'images | Référence pour la vision classique |
| `scikit-learn` | ML classique et métriques | Indispensable hors deep learning |
| `xgboost` / `lightgbm` | Arbres boostés sur données tabulaires | Souvent supérieurs au DL sur le tabulaire |
| `sentence-transformers` | Embeddings de phrases | Base du retrieval sémantique |
| `langchain` | Orchestration d'applications LLM | Chaînes, agents, connecteurs |
| `llamaindex` | Indexation et RAG | Très fort sur les pipelines documentaires |
| `faiss` | Recherche de plus proches voisins | Index vectoriels à grande échelle |
| `gradio` / `streamlit` | Interfaces Web rapides | Prototyper une démo en quelques lignes |

Le choix d'une bibliothèque tient souvent à l'interopérabilité : une brique Hugging Face se combine naturellement avec une autre.

## 4. Hugging Face : le hub central

Hugging Face est devenu, en quelques années, le "GitHub de l'IA".
C'est la plateforme de référence pour tout développeur souhaitant manipuler des modèles pré-entraînés.

*   **Model Hub** :
    Des centaines de milliers de modèles hébergés (LLM, vision, audio, diffusion).
    Chaque modèle est documenté par une "model card" qui précise l'usage prévu, les données d'entraînement et les limites.
*   **Datasets** :
    Une collection immense de jeux de données normalisés, chargeables en une ligne pour l'entraînement ou le fine-tuning.
*   **Spaces** :
    Des applications Web interactives (souvent Gradio ou Streamlit) pour tester un modèle dans le navigateur sans installation locale.
*   **Safetensors** :
    Un format de sérialisation sécurisé pour les poids, qui remplace le format pickle. Il évite l'exécution de code arbitraire au chargement.
*   **Gating** :
    Certains modèles exigent une autorisation et l'acceptation d'une licence avant le téléchargement (par exemple les modèles Llama).
*   **Bibliothèques Python** :
    `transformers`, `datasets`, `tokenizers`, `accelerate`, `diffusers`, `peft`, `trl`.

Le rôle central de Hugging Face en fait le point d'entrée obligé pour quiconque veut utiliser un modèle sans le réentraîner.

## 5. Fournisseurs d'API

Pour beaucoup d'applications, la voie la plus rapide est d'appeler un modèle par API plutôt que de le déployer soi-même.

| Fournisseur | Modèles phares | Positionnement |
|---|---|---|
| OpenAI | GPT-4o, GPT-4o mini, o-séries | Référence grand public, forte notoriété |
| Anthropic | Claude 3.5 Sonnet, Claude 3 Opus | Contexte long, raisonnement, sûreté |
| Google | Gemini 1.5 Pro, Gemini Flash | Intégration Workspace, très long contexte |
| Mistral AI | Mistral Large, Small, Codestral | Acteur européen, poids ouverts et API |
| Cohere | Command R, Embed | Focalisé RAG et usage entreprise |
| DeepSeek | DeepSeek-V3, DeepSeek-R1 | Très bon rapport qualité/prix, raisonnement |
| xAI | Grok | Intégration à la plateforme X |
| Groq | Inférence accélérée (LPU) | Latence très faible sur modèles ouverts |
| Together / Fireworks | Hébergement de modèles ouverts | Serving managé à haut débit |
| OpenRouter | Agrégateur multi-fournisseurs | Une seule API pour beaucoup de modèles |

Le choix dépend de trois critères : la qualité requise, le coût par million de tokens et les contraintes de localisation des données.
Un agrégateur comme OpenRouter simplifie la comparaison mais ajoute une couche d'intermédiation.

## 6. Les modèles ouverts

Les modèles ouverts sont publiés avec leurs poids, ce qui permet le déploiement local et le fine-tuning.

| Modèle | Organisation | Tailles typiques | Licence |
|---|---|---|---|
| Llama 3.1 / 3.3 | Meta | 8B, 70B, 405B | Licence communautaire Meta |
| Mistral / Mixtral | Mistral AI | 7B, 8x7B, 8x22B | Apache 2.0 (selon version) |
| Qwen 2.5 | Alibaba | 0.5B a 72B | Apache 2.0 (selon version) |
| DeepSeek-V3 / R1 | DeepSeek | ~671B (MoE) | Licence ouverte spécifique |
| Gemma 2 | Google | 2B, 9B, 27B | Licence Gemma |
| Phi-3 / Phi-4 | Microsoft | 3.8B, 14B | Licence MIT |
| OLMo | Allen Institute | 7B, 13B | Apache 2.0, données ouvertes |
| Command R | Cohere | 35B, 104B | CC-BY-NC (non commercial) |

Les licences sont un point d'attention : Apache 2.0 et MIT sont très permissives.
Les licences communautaires (Meta, Gemma) imposent des restrictions d'usage, par exemple au-delà d'un certain nombre d'utilisateurs mensuels.
Les licences OpenRAIL ajoutent des clauses d'usage responsable.
Command R est sous CC-BY-NC, donc interdit en usage commercial.

## 7. Les modèles frontières fermés

Les modèles frontières fermés restent la référence sur les benchmarks généraux et les tâches de raisonnement complexe.

*   **GPT-4o et les modèles o** (OpenAI) :
    Très multimodaux, avec une famille dédiée au raisonnement (chaîne de pensée interne).
*   **Claude 3.5 Sonnet et Opus** (Anthropic) :
    Réputés pour la longueur de contexte, le code et le suivi d'instructions précises.
*   **Gemini 1.5 et 2.x** (Google) :
    Contexte très long (jusqu'à plusieurs millions de tokens), intégration multimodale native.

Leur positionnement repose sur la performance brute et la simplicité d'accès.
Leur faiblesse est la dépendance à un fournisseur, le coût par requête et l'envoi de données à un tiers.

## 8. Les leaderboards

Comparer les modèles est difficile car les benchmarks saturent et fuient.

*   **LMSYS Chatbot Arena** :
    Un classement par préférence humaine sur des duels à l'aveugle, avec un score Elo.
    C'est la référence la plus respectée pour l'usage conversationnel.
*   **Open LLM Leaderboard** (Hugging Face) :
    Une évaluation standardisée de modèles ouverts sur plusieurs tâches automatisées.
*   **Artificial Analysis** :
    Un comparatif orienté pratique : latence, débit, coût et qualité par modèle.

Aucun leaderboard n'est neutre : chacun privilégie certaines tâches et peut être optimisé par des modèles qui surentraînent sur les benchmarks.
Utilise-les comme point de départ, jamais comme vérité finale.

## 9. Outils de développement et MLOps

De la manipulation locale au déploiement en cluster, l'outillage couvre toute la chaîne.

*   **Notebooks (Jupyter, Colab)** : exploration interactive et partage de résultats.
*   **git / DVC** : versionner le code et, séparément, les données et modèles volumineux.
*   **MLflow / Weights & Biases** : tracer les expériences, les hyperparamètres et les métriques.
*   **Docker** : figer l'environnement pour reproduire une exécution à l'identique.
*   **Kubernetes** : orchestrer des services conteneurisés à l'échelle.
*   **Ray** : distribuer entraînement et inférence sur des clusters de GPU.
*   **Ollama** : lancer un LLM local en une commande, avec un serveur compatible API.
*   **LM Studio** : interface graphique pour télécharger et tester des modèles ouverts sur son poste.

En pratique, un développeur commence souvent par Ollama ou LM Studio pour prototyper, puis passe à vLLM et Docker pour la production.

## 10. Comment choisir un modèle

Voici un arbre de décision simple pour orienter le choix.

```
Besoin de confidentialité forte (données sensibles) ?
  oui -> déploiement local d'un modèle ouvert (Llama, Qwen, Mistral)
  non -> continuer

Besoin de la meilleure qualité possible sur des tâches variees ?
  oui -> API d'un modèle frontière ferme (GPT-4o, Claude, Gemini)
  non -> continuer

Contrainte de coût serrée et gros volume ?
  oui -> modèle ouvert héberge (Together, Fireworks) ou API à bas coût (DeepSeek)
  non -> continuer

Besoin d'une tâche spécialisée (code, embeddings, vision) ?
  oui -> choisir un modèle dédié a cette tâche
  non -> utiliser un modèle generaliste de milieu de gamme

Contrainte matérielle locale (VRAM limitée) ?
  oui -> modèle quantifie (4 bits) de 7 a 14B, via Ollama
  non -> viser un modèle 30B+ ou une API
```

Le critère determinant est presque toujours le compromis entre contrôle, coût et qualité.

## 11. Cas pratique : premier appel API et premier modèle local

Voici deux approches pour démarrer concrètement.

```python
# Approche 1 : appel d'une API (exemple générique, adapté selon le fournisseur)
# Installation : pip install openai
from openai import OpenAI

client = OpenAI(api_key="TA_CLE_API")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Tu réponds en français, de façon concise."},
        {"role": "user", "content": "Explique le fine-tuning en deux phrases."},
    ],
)
print(response.choices[0].message.content)
```

```bash
# Approche 2 : modèle local via Ollama (aucune cle API, tout reste sur la machine)
# Installation : voir ollama.com
ollama pull llama3.1:8b
ollama run llama3.1:8b "Explique le fine-tuning en deux phrases."

# Ollama expose aussi une API compatible OpenAI sur le port 11434
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Explique le fine-tuning en deux phrases.",
  "stream": false
}'
```

Ces deux extraits montrent la même tâche accomplie soit par un service distant, soit par un modèle local.

## Ce qu'il faut retenir

- L'écosystème IA combine des acteurs fermés (OpenAI, Anthropic, Google) et ouverts (Meta, Mistral, Qwen, DeepSeek).
- PyTorch s'est imposé comme le standard de fait pour la recherche et une grande partie de l'industrie.
- Hugging Face est la plateforme centrale pour partager et utiliser des modèles, des datasets et des applications.
- Le choix entre API et déploiement local dépend de la confidentialité, du coût et de la maintenance.
- Les bibliothèques se composent bien : transformers, datasets, peft, trl, vLLM forment un pipeline complet.
- Les modèles ouverts affichent des tailles de quelques milliards à plusieurs centaines de milliards de paramètres.
- Les licences des modèles ouverts varient : Apache et MIT sont permissives, les licences communautaires sont plus restrictives.
- Les leaderboards (Chatbot Arena, Open LLM Leaderboard) donnent une orientation, pas une vérité absolue.
- L'outillage MLOps (Docker, Kubernetes, Ray, W&B) est indispensable pour passer du prototype à la production.
- Le choix d'un modèle se résume souvent à un arbitrage entre contrôle, coût et qualité.

## Erreurs fréquentes / idées reçues

- Hugging Face n'héberge que des modèles -> Faux. C'est aussi une plateforme de datasets, d'applications et de partage communautaire.
- PyTorch est réservé à la recherche -> Faux. Il est massivement utilisé en production pour le service de modèles à grande échelle.
- Les modèles fermés seront toujours supérieurs aux modèles ouverts -> Idée reçue. L'écart se réduit vite et les modèles ouverts gagnent sur la confidentialité et le contrôle.
- Un modèle ouvert est gratuit -> Faux. La licence peut l'être, mais l'infrastructure (GPU, électricité, maintenance) ne l'est pas.
- Déployer un LLM exige une armée d'ingénieurs -> Faux. Des outils comme Ollama ou vLLM rendent le déploiement accessible à un développeur seul.
- Meilleur sur un leaderboard signifie meilleur pour mon usage -> Faux. Les benchmarks saturent et ne reflètent pas ton domaine ni tes contraintes.

## Pour aller plus loin

- [chapitre 06](../06-fine-tuning/README.md) : Pour affiner un modèle ouvert avec LoRA et QLoRA.
- [chapitre 09](../09-inference-optimisation/README.md) : Pour servir efficacement un modèle avec vLLM et la quantification.
- [chapitre 05](../05-llm/README.md) : Pour comprendre ce que recouvrent les familles de modèles citées ici.
- [chapitre 16](../16-pratique/README.md) : Pour mettre en pratique l'outillage dans des tutoriels.
- Documentation Hugging Face (huggingface.co/docs) : Référence des bibliothèques transformers, datasets et peft.
- Documentation PyTorch (pytorch.org) : Guides et tutoriels officiels.
- State of AI Report (stateof.ai) : Synthèse annuelle du paysage et des tendances.

# Inférence : optimiser et servir un modèle

> L'inférence est l'étape où le modèle de langage entraîné est exécuté pour générer des prédictions (tokens). En production, l'enjeu principal se déplace de la puissance pure de calcul vers l'optimisation des flux mémoires, la latence et la réduction drastique des coûts opérationnels.

## 1. Anatomie d'une inférence : Prefill vs Decode

L'inférence d'un modèle autorégressif (comme les Transformers de type décodeur) n'est pas un bloc homogène. Elle se sépare en deux phases fondamentales dont les profils matériels sont radicalement différents.

### La phase de Prefill (traitement du prompt)
Lors de cette première étape, le modèle reçoit l'intégralité du prompt d'entrée. Il calcule les représentations d'attention pour tous les tokens d'entrée en une seule passe parallèle. 
- **Régime de fonctionnement** : Cette phase est **compute-bound** (limitée par la puissance de calcul). Le processeur passe son temps à effectuer des multiplications de matrices denses sur les Tensor Cores.
- **But** : Initialiser les états internes et générer le premier token de sortie.

### La phase de Decode (génération de tokens)
Une fois le premier token produit, le modèle entre dans une boucle séquentielle. Il génère un token, l'ajoute à son entrée, puis génère le suivant.
- **Régime de fonctionnement** : Cette phase est **memory-bound** (limitée par la bande passante mémoire). Pour générer chaque unique token, le GPU doit charger la totalité des poids du modèle depuis sa mémoire globale (VRAM) vers ses registres locaux (SRAM). Les cœurs de calcul passent la majorité de leur temps à attendre les données.
- **But** : Produire la suite du texte de manière fluide, un token après l'autre.

### Schéma ASCII de l'architecture d'inférence

```text
       +--------------------------------------------+
       |             Prompt utilisateur             |
       +--------------------------------------------+
                             |
                             v
               +---------------------------+
               |     Phase de PREFILL      |  <--- Très intense en calcul
               |  (Calcul parallèle initial)|       (Compute-bound)
               +---------------------------+
                             |
                             +------------------------+
                             | Stockage K, V          |
                             v                        v
               +---------------------------+    +-----------+
               |      Phase de DECODE      | <| | KV Cache  | <--- Évite le recalcul
               |  (Génération séquentielle)|    +-----------+      (Memory-bound)
               +---------------------------+
                             |
                             +--> [Nouveau Token] --+
                                                    | (Boucle de retour)
                                                    v
```

---

## 2. Débits et latences : les métriques clés

L'évaluation de la performance d'un système de service (serving) de LLM repose sur plusieurs métriques temporelles spécifiques.

### TTFT (Time-To-First-Token)
Le TTFT est le délai mesurant le temps écoulé entre l'envoi de la requête par l'utilisateur et l'affichage du tout premier caractère généré. Cette latence est dominée par la phase de prefill. Elle est critique pour l'interactivité d'un chatbot (une valeur cible est généralement inférieure à 800 ms).

### TPOT (Time-Per-Output-Token) ou ITL (Inter-Token Latency)
Cette métrique correspond au temps moyen requis pour générer chaque token individuel après le premier. Elle dépend quasi exclusivement de la vitesse de la phase de decode. Pour une lecture fluide, un TPOT de l'ordre de 15 à 50 ms par token est optimal.

### Débit (Throughput)
Le débit global est exprimé en **tokens par seconde** sur l'ensemble du système. Il mesure le volume total de tokens traités ou générés pour toutes les requêtes actives simultanément.

### Impact du Batching
Le *batching* consiste à regrouper plusieurs requêtes d'utilisateurs distincts pour les exécuter dans une seule instruction matérielle (SIMT). 
- **Arbitrage** : Un batching plus élevé augmente le débit global (throughput) car le coût de chargement des poids du modèle depuis la VRAM est amorti sur plusieurs requêtes. Cependant, cela augmente légèrement le TPOT pour chaque utilisateur en raison de la contention sur les ressources et du traitement de requêtes plus longues.

```text
Débit vs Latence (Courbe typique) :
Latence (ms)
  ^
  |                                 / (Saturé)
  |                                /
  |                       *-------*
  |              *-------* (Zone optimale)
  |       *-----*
  +----------------------------------------> Débit (Tokens/s)
```

---

## 3. Le KV Cache : principes et implémentation

Au cours de la génération séquentielle, le calcul d'attention pour le token *t* nécessite la connaissance des représentations d'attention de tous les tokens passés (de *1* à *t-1*). Recalculer ces tenseurs à chaque étape serait d'une inefficacité extrême. On stocke donc les vecteurs de Clés (*Key*) et Valeurs (*Value*) dans un tampon mémoire appelé **KV Cache**.

### Formule de taille du KV Cache
Le KV Cache consomme une quantité considérable de VRAM. Pour un modèle donné, sa taille totale en octets se calcule comme suit :

$$\text{Taille} = 2 \times N_{\text{couches}} \times N_{\text{têtes}} \times D_{\text{tête}} \times P \times L_{\text{séquence}} \times B_{\text{batch}}$$

Où :
- $2$ représente les deux tenseurs stockés (K et V).
- $N_{\text{couches}}$ est le nombre de couches de transformeur du modèle.
- $N_{\text{têtes}}$ est le nombre de têtes d'attention (ou têtes de clés/valeurs si optimisé).
- $D_{\text{tête}}$ est la dimension de chaque tête.
- $P$ est la précision numérique en octets (ex: 2 pour FP16 ou BF16, 1 pour FP8, 4 pour FP32).
- $L_{\text{séquence}}$ est la longueur cumulée du contexte et de la génération.
- $B_{\text{batch}}$ est le nombre de requêtes simultanées dans le batch.

### Exemple chiffré : Llama-3-8B
Prenons un modèle Llama 3 8B fonctionnant en précision native FP16 (2 octets) :
- Couches ($N_{\text{couches}}$) : 32
- Têtes de KV ($N_{\text{têtes}}$) : 8 (utilisation de GQA)
- Dimension de tête ($D_{\text{tête}}$) : 128
- Précision ($P$) : 2 octets
- Longueur de contexte ($L_{\text{séquence}}$) : 4096 tokens
- Taille du batch ($B_{\text{batch}}$) : 16 requêtes

$$\text{Taille} = 2 \times 32 \times 8 \times 128 \times 2 \times 4096 \times 16 = 17 \ 179 \ 869 \ 184 \text{ octets} \approx 16 \text{ GB}$$

Dans cet exemple, le KV cache à lui seul nécessite autant de VRAM (16 GB) que le stockage des poids du modèle lui-même !

### Évolutions architecturales : MQA et GQA
Pour limiter cette explosion mémoire, les architectures ont évolué depuis l'attention classique (MHA - Multi-Head Attention) :
- **MQA (Multi-Query Attention)** : Une seule tête de clé et de valeur est partagée pour toutes les têtes de requêtes (Query). Cela réduit la taille du cache d'un facteur égal au nombre de têtes de Query, mais peut légèrement dégrader la qualité des représentations.
- **GQA (Grouped-Query Attention)** : Un compromis optimal où les têtes de Query sont regroupées (par exemple, 8 groupes), et chaque groupe partage une seule tête de clé et de valeur. C'est l'architecture standard de Llama 3 et de nombreux LLM récents.

### PagedAttention
Avant l'introduction de PagedAttention par les créateurs de vLLM, la mémoire du KV Cache devait être allouée de manière contiguë pour la longueur de contexte maximale théorique de chaque requête. Cela provoquait une fragmentation massive (jusqu'à 60% de mémoire gâchée par de l'espace alloué mais inutilisé).
- **Principe** : PagedAttention divise le KV Cache en petits blocs non contigus (similaires aux pages physiques des OS). Les blocs sont alloués dynamiquement au fur et à mesure de la génération, éliminant la fragmentation interne et permettant de doubler le débit opérationnel à ressources matérielles constantes.

### Quantification du cache
Pour réduire davantage l'empreinte mémoire, les moteurs modernes proposent de quantifier le KV Cache en FP8 ou en INT8. Cela divise par deux l'espace mémoire nécessaire au détriment d'une infime perte de précision sur les relations d'attention lointaines.

---

## 4. Moteurs de service (Serving Engines)

Le choix du moteur d'inférence est l'arbitrage architectural le plus structurant pour mettre un modèle en production.

| Moteur | Cible principale | Avantages majeurs | Inconvénients / Limites |
| :--- | :--- | :--- | :--- |
| **vLLM** | Production Cloud, API de masse | PagedAttention, Batching continu, très fort débit | Temps d'initialisation, rigide pour les architectures exotiques |
| **SGLang** | Requêtes complexes, Agents | Vitesse extrême, Prompt Caching natif, syntaxe optimisée | Moins standardisé dans les infrastructures d'entreprise |
| **TGI (Text Generation Inference)** | Enterprise de confiance (Hugging Face) | Production robuste, hautement sécurisé, support complet des modèles HF | Moins flexible pour le prototypage rapide en dehors de l'éco-système HF |
| **TensorRT-LLM** | Performance maximale sur GPU NVIDIA | Compilation ultra-optimisée, support natif des puces Tensor Cores | Compilation lente et complexe, strictement lié au matériel NVIDIA |
| **llama.cpp** | Local, Edge, CPU, Mac | Pas de dépendance lourde, hautement portable (C/C++), exécution CPU | Débit limité par rapport aux frameworks GPU spécialisés |
| **Ollama** | Développeur individuel, Prototypage | Expérience utilisateur exceptionnelle, installation en une ligne | Difficile à scaler pour des charges de production massives |

---

## 5. Ordonnancement et gestion des flux

L'optimisation du serving ne se limite pas à l'exécution de kernels mathématiques rapides. La gestion de la file d'attente des requêtes joue un rôle prépondérant.

### Batching continu (Continuous Batching)
Dans un système de batching classique (statique), le GPU attend que toutes les phrases du groupe soient complétées avant de renvoyer les réponses et d'accepter de nouvelles tâches. 
- Le **batching continu** résout ce problème en fonctionnant au niveau du token. Dès qu'une requête du batch produit son token de fin (`<|eot_id|>`), elle est immédiatement extraite du batch de calcul et remplacée par une nouvelle requête en attente.

### Chunked Prefill
Lorsqu'un prompt extrêmement long (ex: document de 32 000 tokens) entre dans la file d'attente, sa phase de prefill monopolise les ressources du GPU pendant plusieurs secondes, gelant temporairement la génération des tokens (decode) des autres utilisateurs actifs.
- Le **chunked prefill** découpe le long prompt en morceaux (chunks) de 512 ou 1024 tokens. Le moteur entrelace le traitement de ces morceaux de prefill avec les étapes régulières de decode des autres requêtes en cours, lissant ainsi la latence globale.

### Cache de préfixe (Prefix Caching)
Dans de nombreuses applications d'agents ou de RAG, les requêtes successives partagent un préfixe commun substantiel (ex: instructions système, base de connaissances fixe). Le moteur d'inférence identifie ces blocs de texte récurrents et conserve leur KV cache en mémoire. Lorsqu'une nouvelle requête arrive avec le même préfixe, le prefill de cette section est évité, ramenant le TTFT à quelques millisecondes.

---

## 6. Quantification côté inférence (Quantization)

La quantification est la réduction du format de stockage des poids et/ou des activations du modèle afin de réduire son empreinte VRAM et d'exploiter des unités de calcul plus rapides.

### Formats majeurs de quantification

- **INT8 (Quantification 8-bits)** : Permet de diviser par deux la mémoire requise par rapport au FP16 d'origine. Les poids sont mis à l'échelle pour s'adapter à l'intervalle $[-128, 127]$. La perte de qualité de réponse est virtuellement nulle sur les modèles de taille moyenne (> 13B).
- **INT4 (Quantification 4-bits)** : Réduit la taille par quatre. Les méthodes comme **GPTQ** ou **AWQ** (Activation-aware Weight Quantization) trient les poids en fonction de l'importance de leurs activations pour protéger les canaux de calcul sensibles. La perte de perplexité est mesurable mais reste très acceptable pour la plupart des cas d'usage conversationnels.
- **GGUF** : Développé pour `llama.cpp`, ce format intègre le modèle et ses métadonnées dans un fichier unique. Il utilise des méthodes de quantification par blocs (variantes Q2_K, Q4_K_M, Q8_0) permettant de répartir finement les bits entre les différentes couches pour un ratio performance/taille optimal.
- **FP8** : Format de choix sur les puces de génération récente (NVIDIA H100, Blackwell). Il combine l'efficacité de stockage proche du format 8-bits standard avec une plage dynamique préservée, limitant drastiquement la dégradation du modèle sans nécessiter de calibration complexe.

### Bitsandbytes et quantification à la volée
La bibliothèque `bitsandbytes` permet de charger des modèles Hugging Face directement en 8-bits ou 4-bits (avec le type NF4 pour QLoRA) sans pré-quantification lourde. C'est l'outil de référence pour le prototypage rapide et le fine-tuning sur machine unique.

---

## 7. Accélération de la génération : Spéculative Decoding et alternatives

La génération séquentielle classique impose d'exécuter l'intégralité du grand modèle pour chaque token produit. Plusieurs techniques visent à contourner ce goulot d'étranglement.

### Décodage spéculatif (Speculative Decoding)
Cette approche utilise deux modèles : un modèle cible très grand et lent (ex: Llama 3 70B) et un modèle de brouillon (draft) très rapide et léger (ex: Llama 3 8B).
1. Le modèle draft génère rapidement une suite de $K$ tokens spéculatifs.
2. Le modèle cible valide ces $K$ tokens en une seule passe parallèle rapide (car c'est un calcul de prefill, donc compute-bound et parallélisable).
3. Si le grand modèle rejette le token à la position $i < K$, on conserve les $i-1$ tokens acceptés plus un token corrigé par le grand modèle, puis on recommence.
- **Gain** : Multiplie la vitesse par un facteur de $1.5$ à $2.5$ selon l'accord sémantique entre les deux modèles, sans aucune altération de la qualité finale du texte.

### Medusa et Lookahead
- **Medusa** : Au lieu d'un modèle draft séparé, on ajoute des têtes de prédiction supplémentaires (heads) au sommet du modèle principal. Chaque tête tente de prédire les tokens futurs en parallèle ($t+1, t+2, t+3$).
- **Lookahead** : Exploite des techniques de recherche sémantique locale dans l'historique récent pour anticiper les séquences probables sans nécessiter de second modèle.

---

## 8. Parallélisme côté inférence

Pour les modèles dont l'échelle dépasse la capacité mémoire d'une unique carte graphique (ex: un modèle de 405 milliards de paramètres nécessite ~810 GB en FP16), il est obligatoire de distribuer l'inférence sur plusieurs puces.

- **Tensor Parallelism (TP)** : Les matrices de poids de chaque couche individuelle sont découpées et réparties sur les différents GPU. Durant le forward, les GPU s'échangent les résultats intermédiaires via des interconnexions ultra-rapides (NVLink). Le TP est extrêmement efficace mais exige une bande passante inter-GPU très élevée.
- **Pipeline Parallelism (PP)** : Le modèle est découpé par groupes de couches (ex: couches 1 à 16 sur le GPU 0, 17 à 32 sur le GPU 1). Les activations transitent séquentiellement d'un GPU à l'autre. Moins exigeant en bande passante réseau, le PP souffre de temps d'attente (bubbles) où certains GPU restent inactifs en attendant les résultats du précédent.

---

## 9. Inférence des architectures de mélange d'experts (MoE)

Les architectures MoE (comme Mixtral 8x7B ou DeepSeek-V3) posent des défis uniques à l'inférence.
- Un modèle MoE possède un grand nombre de paramètres totaux (ex: 47 milliards pour Mixtral) mais n'active qu'une fraction de ceux-ci (ex: 13 milliards) pour chaque token via un routeur sémantique.
- **Arbitrage de la VRAM** : Pour exécuter l'inférence, la totalité des experts doit résider en VRAM, ce qui impose une grande quantité de mémoire physique (besoin d'une carte de type RTX 3090/4090 ou de plusieurs cartes). Cependant, la vitesse de calcul par token (decode) reste équivalente à celle d'un modèle beaucoup plus petit (le modèle actif de 13B), offrant un excellent débit par rapport à sa capacité sémantique théorique.

---

## 10. Modélisation économique : Coûts d'inférence

La décision d'héberger son propre moteur d'inférence ou d'utiliser des API managées (comme OpenAI, Anthropic ou Groq) repose sur un calcul précis du seuil de rentabilité.

### Calcul du coût d'un GPU loué
Supposons la location d'une instance GPU de type NVIDIA A100 (80 GB) à un tarif moyen de $1.50$ de l'heure.
- Capacité moyenne en production (avec vLLM et batching continu) : $1500$ tokens par seconde générés.
- Nombre de tokens produits par heure : 
$$1500 \times 3600 = 5 \ 400 \ 000 \text{ tokens/heure}$$
- Coût de revient pour 1 million de tokens générés : 
$$\frac{\$1.50}{5.4} \approx \$0.28 \text{ / million de tokens}$$

### Seuil de rentabilité (API vs GPU)
Si une API managée équivalente facture le million de tokens sortants à $\$1.00$, le déploiement sur GPU dédié devient économiquement viable dès lors que le taux d'utilisation de la machine dépasse un seuil d'activité continu :
$$\text{Seuil d'utilisation} = \frac{\text{Coût d'hébergement brut}}{\text{Équivalent coût API}} = \frac{\$0.28}{\$1.00} \approx 28\%$$

Si l'application génère du trafic constant représentant plus de 28% de la capacité de la carte sur 24 heures, l'hébergement dédié est plus économique. En dessous, le paiement à l'usage des API est préférable.

---

## 11. Optimisation des coûts de prompt

Le prompt d'entrée (contexte) représente souvent la majeure partie du volume de données traité par les LLM. Réduire sa taille ou son coût de traitement est prioritaire.

- **Mise en cache du contexte (Context Caching)** : De nombreux fournisseurs d'API (comme Anthropic ou DeepSeek) proposent des réductions tarifaires substantielles (jusqu'à -90%) pour les parties de prompts qui sont répétées à l'identique entre plusieurs requêtes, car elles exploitent le cache de préfixe matériel.
- **Moteurs de compression de prompt** : Des outils spécialisés (comme LLMLingua) analysent la structure sémantique du prompt d'entrée et suppriment les mots de liaison, les répétitions et les éléments d'information redondants sans altérer la capacité de réponse du modèle, réduisant la taille du prompt de 20% à 50%.

---

## Exemple de code : Script de Benchmark d'inférence basique

Voici un exemple simple en Python permettant de mesurer le TTFT et le débit d'un serveur d'inférence local (compatible OpenAI API) sous une charge séquentielle :

```python
import time
import requests

API_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
PAYLOAD = {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Rédige une dissertation sur l'histoire de l'électricité."}],
    "stream": True,
    "max_tokens": 150
}

def benchmark_inference():
    start_time = time.time()
    ttft = None
    token_count = 0
    
    response = requests.post(API_URL, json=PAYLOAD, headers=HEADERS, stream=True)
    
    for chunk in response.iter_lines():
        if chunk:
            if ttft is None:
                ttft = time.time() - start_time
                print(f"[METRIC] Time To First Token (TTFT): {ttft:.3f} secondes")
            token_count += 1

    total_time = time.time() - start_time
    tpot = (total_time - ttft) / max(1, token_count - 1) if token_count > 1 else 0
    throughput = token_count / total_time
    
    print(f"[METRIC] Temps total d'exécution: {total_time:.3f} secondes")
    print(f"[METRIC] Nombre estimé de tokens générés: {token_count}")
    print(f"[METRIC] Time Per Output Token (TPOT): {tpot * 1000:.2f} ms")
    print(f"[METRIC] Débit moyen: {throughput:.2f} tokens/seconde")

if __name__ == "__main__":
    print("Démarrage du benchmark d'inférence...")
    try:
        benchmark_inference()
    except Exception as e:
        print(f"[ERREUR] Impossible de joindre le serveur de serving: {e}")
```

---

## Ce qu'il faut retenir
- L'inférence comporte une phase intense en calcul (prefill) et une phase limitée par la bande passante mémoire (decode).
- Le **KV Cache** évite les recalculs mais impose une pression gigantesque sur la VRAM du GPU.
- Le **batching continu** et **PagedAttention** sont les innovations majeures qui ont rendu le serving de masse viable financièrement.
- La quantification en **INT4 / FP8** permet d'exécuter des modèles performants sur du matériel beaucoup plus accessible avec un impact négligeable sur la précision.
- Les architectures MoE demandent beaucoup de VRAM pour stocker leurs experts mais calculent à la vitesse d'un petit modèle.

## Erreurs fréquentes / idées reçues
- *Idée reçue : "Pour doubler la vitesse d'inférence d'un décodeur, il suffit de doubler la puissance de calcul du GPU"* -> Faux. En phase de decode, la vitesse dépend presque exclusivement de la bande passante mémoire (HBM/VRAM), pas des TFLOPS bruts.
- *Idée reçue : "Le décodage spéculatif altère la qualité des réponses car le petit modèle écrit une partie du texte"* -> Faux. Le grand modèle vérifie mathématiquement chaque token proposé. Si un seul token s'écarte de sa propre distribution de probabilité, il le rejette et le recalcule. La qualité finale est strictement identique à celle du grand modèle seul.
- *Idée reçue : "Il faut toujours utiliser le plus grand batch possible"* -> Faux. Un batch trop important peut saturer le KV cache, dégrader le TTFT des utilisateurs à des niveaux inacceptables et provoquer des erreurs de dépassement de mémoire (Out Of Memory).

## Pour aller plus loin
- [chapitre 10](../10-infrastructure/README.md) pour approfondir le fonctionnement de la VRAM, des bus de communication et des différentes gammes de GPU.
- Papier de recherche fondateur : *Orca: A Distributed Serving System for Transformer-Based Generative Models* (introduction du continuous batching).
- Papier de recherche de vLLM : *Efficient Memory Management for Large Language Model Serving with PagedAttention*.
- Dépôt de référence pour le déploiement local performant : [llama.cpp](https://github.com/ggerganov/llama.cpp).

# Aide-mémoire

Ce document compile les formules indispensables, les commandes fréquentes, les ordres de grandeur physiques et financiers, ainsi que des arbres de décision pour guider vos projets d'IA et de traitement du langage naturel.

---

## 1. Formules essentielles

### Attention produit scalaire normé (Scaled Dot-Product Attention)
$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- $Q$ (Query), $K$ (Key), $V$ (Value) de dimension $d_k$.
- Le facteur $\sqrt{d_k}$ évite la saturation du softmax pour les grandes dimensions.

### Multi-Head Attention (MHA) vs Multi-Query Attention (MQA) vs Grouped-Query Attention (GQA)
- **MHA** : Chaque tête de Query possède sa propre tête de Key et de Value.
- **MQA** : Toutes les têtes de Query partagent une seule tête de Key et de Value.
- **GQA** : Les têtes de Query sont regroupées, et chaque groupe partage une tête de Key et de Value.
$$\text{GQA} \implies \text{Nombre de têtes de Query} = G \times \text{Nombre de têtes KV}$$

### Softmax
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
- Transforme un vecteur de scores réels en une distribution de probabilités (somme égale à 1).

### Activation SwiGLU (utilisée dans Llama 2 & 3, Mistral)
$$\text{SwiGLU}(x) = \text{Swish}_{\beta}(xW) \otimes xV = \left( \frac{xW}{1 + e^{-\beta xW}} \right) \otimes xV$$
- Remplace l'activation GELU traditionnelle dans les couches Feed-Forward pour une convergence plus rapide.

### Entropie croisée (Cross-Entropy)
$$H(p, q) = -\sum_{i=1}^{N} p_i \log(q_i)$$
- Mesure l'écart entre la distribution réelle $p$ et la distribution prédite $q$. Utilisée pour l'entraînement des classificateurs et des LLM.

### Divergence de Kullback-Leibler (KL Divergence)
$$D_{KL}(P \parallel Q) = \sum_{i=1}^{N} P(i) \log\left(\frac{P(i)}{Q(i)}\right)$$
- Mesure la perte d'information si la distribution $Q$ est utilisée pour approximer la distribution réelle $P$. Utilisée en RLHF et distillation.

### Perplexité (PPL)
$$PPL(X) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{<i}) \right) = e^{H(X)}$$
- Mesure la qualité d'un modèle de langage. Plus la perplexité est basse, plus le modèle est confiant dans ses prédictions.

### Taille du KV Cache (par token généré)
$$\text{Taille}_{\text{KV}} = 2 \times n_{\text{couches}} \times n_{\text{têtes\_kv}} \times d_{\text{tête}} \times \text{octets\_par\_précision}$$
- Pour Llama 3 8B en FP16 ($n_{\text{couches}}=32, n_{\text{têtes\_kv}}=8, d_{\text{tête}}=128$, 2 octets) :
  $$\text{Taille}_{\text{KV}} = 2 \times 32 \times 8 \times 128 \times 2 = 131\,072 \text{ octets} \approx 131 \text{ Ko par token}$$
- Pour une séquence de 4096 tokens, cela représente environ 536 Mo par utilisateur actif.

### Estimation de la VRAM pour l'inférence
$$\text{VRAM}_{\text{total}} \approx \left( \frac{P \times 2}{\text{Facteur de quantification}} \times 1.2 \right) + \text{VRAM}_{\text{KV\_Cache}} + \text{VRAM}_{\text{contexte}}$$
- $P$ est le nombre de paramètres en milliards.
- Le facteur 1.2 ajoute une marge de 20% pour les activations de l'infrastructure d'inférence.

### Consommation VRAM pour l'entraînement (par paramètre)
- **FP32 Entier (Full FT)** : $\approx 16 \text{ octets}$ par paramètre (4 pour le poids, 4 pour le gradient, 8 pour l'optimiseur AdamW).
- **FP16 / BF16 Entier (Full FT)** : $\approx 12 \text{ octets}$ par paramètre (2 pour le poids, 2 pour le gradient, 8 pour l'optimiseur AdamW).
- **LoRA (avec poids figés en FP16)** : $\approx 2 \text{ octets} \times P_{\text{figés}} + 12 \text{ octets} \times P_{\text{entraînables}}$.
- **QLoRA (avec poids figés en INT4 NF4)** : $\approx 0.5 \text{ octets} \times P_{\text{figés}} + 12 \text{ octets} \times P_{\text{entraînables}}$.

### FLOPs d'entraînement
$$\text{FLOPs} \approx 6 \times P \times N_{\text{tokens}}$$
- Étape de rétropropagation (backward) requiert environ deux fois plus de calculs que la passe avant (forward).
- Pour un modèle de 7 milliards de paramètres entraîné sur 2 000 milliards (2T) de tokens :
  $$\text{FLOPs} = 6 \times 7\times 10^9 \times 2\times 10^{12} = 8.4\times 10^{22} \text{ FLOPs}$$

### Similarité cosinus
$$\text{Similarité}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$
- Mesure la colinéarité entre deux vecteurs d'embeddings. Indépendante de la norme des vecteurs.

### Mise à jour du gradient (AdamW)
$$g_t = \nabla_{\theta} \mathcal{L}(\theta_t)$$
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad ; \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$
- Utilise des moyennes mobiles exponentielles des gradients ($m_t$) et de leurs carrés ($v_t$), avec régularisation de découplage de poids (weight decay $\lambda$).

---

## 2. Commandes utiles

### Gestion de paquets (pip / uv)
```bash
# Installation ultra-rapide avec uv
uv pip install torch transformers accelerate peft datasets trl faiss-cpu vllm sentencepiece
# Vérifier la configuration CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Téléchargement de modèles (HuggingFace CLI)
```bash
# Se connecter au Hub HuggingFace
huggingface-cli login
# Télécharger un dépôt complet localement
huggingface-cli download meta-llama/Llama-3-8B-Instruct --local-dir ./llama3-8b
# Télécharger un fichier GGUF spécifique
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

### Serveur d'inférence haute performance (vLLM)
```bash
# Lancer un serveur OpenAI compatible pour un modèle local ou HF
vllm serve facebook/opt-125m --port 8000 --gpu-memory-utilization 0.90
# Requête test avec curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "facebook/opt-125m", "messages": [{"role": "user", "content": "Bonjour"}]}'
```

### Serveur d'inférence local (Ollama)
```bash
# Lancer Ollama en arrière-plan
ollama serve
# Télécharger et exécuter un modèle
ollama run llama3
# Utiliser l'API locale pour la génération de texte
curl http://localhost:11434/api/generate -d '{"model": "llama3", "prompt": "Pourquoi le ciel est bleu?"}'
```

### Fusionner un adaptateur LoRA avec le modèle de base
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
peft_model = PeftModel.from_pretrained(base_model, "./lora-adapter")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./llama-3-8b-merged")
```

### Quantification de modèle avec BitsAndbytes (4-bit / 8-bit)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", quantization_config=double_quant_config)
```

---

## 3. Ordres de grandeur

### Tailles de modèles caractéristiques (Paramètres actifs)
- **Petits (1B - 3B)** : TinyLlama (1.1B), Qwen 2.5 (1.5B), Gemma 2 (2B). Usage local sur smartphone ou CPU.
- **Moyens (7B - 14B)** : Llama 3 (8B), Mistral (7B), Qwen 2.5 (14B). Standard pour serveurs d'entreprise.
- **Grands (30B - 70B)** : Llama 3 (70B), Qwen 2.5 (72B). Raisonnement avancé, besoin de matériel dédié.
- **Géants (100B+)** : DeepSeek-V3 (671B au total, 37B actifs), GPT-4 (MoE, confidentiel). APIs hébergées.

### VRAM requise (Modèle seul, sans KV cache)
| Taille modèle | Précision FP16 | Quantification INT8 | Quantification INT4 |
| :--- | :--- | :--- | :--- |
| **1B à 3B** | 3 Go à 6 Go | 1.5 Go à 3 Go | 1 Go à 2 Go |
| **7B à 9B** | 14 Go à 18 Go | 7 Go à 9 Go | 4.5 Go à 6 Go |
| **14B à 20B** | 28 Go à 40 Go | 14 Go à 20 Go | 9 Go à 12 Go |
| **70B à 72B** | 140 Go à 144 Go | 70 Go à 72 Go | 40 Go à 45 Go |

### Tarification GPU cloud (Moyenne observée en 2026)
- **RTX 4090** (24 Go VRAM) : 0.50 $ à 0.80 $ / heure.
- **A10G** (24 Go VRAM) : 1.00 $ à 1.50 $ / heure.
- **A100** (80 Go VRAM) : 1.50 $ à 2.50 $ / heure.
- **H100** (80 Go VRAM) : 2.50 $ à 4.50 $ / heure.

### Performance d'inférence (Tokens par seconde par GPU)
- **RTX 4090** (sur Llama-3-8B-Q4) : 40 à 60 tokens/s.
- **A100** (sur Llama-3-8B-FP16, batch=1) : 70 à 100 tokens/s.
- **A100** (sur Llama-3-70B-FP16, avec vLLM) : 15 à 25 tokens/s.

### Datasets et volumes de pré-entraînement typiques
- **GPT-3** (2020) : 300 milliards de tokens.
- **Llama 1** (2023) : 1.4 billion (trillion) de tokens.
- **Llama 3** (2024) : 15 billions (trillions) de tokens.
- **DeepSeek-V3** (2024) : 14.8 billions (trillions) de tokens.

---

## 4. Arbres de décision

### Choix technologique : Fine-tuning vs RAG vs Prompt
```
                                 [Besoin métier]
                                        |
                 +----------------------+----------------------+
                 |                                             |
         [Nouvelles données]                          [Nouveau format/style]
                 |                                             |
        +--------+--------+                           +--------+--------+
        |                 |                           |                 |
[Mises à jour]    [Données statiques]         [Exemples fournis]  [Sans exemples]
  rapides              complexes                   en prompt            |
        |                 |                           |                 |
     v  v              v  v                        v  v              v  v
    [ RAG ]       [Fine-tuning]                 [ Prompt ]       [Fine-tuning]
```

### Choix du modèle et de la quantification
```
                              [Ressources dispo]
                                      |
              +-----------------------+-----------------------+
              |                                               |
         [GPU < 16 Go]                                   [GPU > 24 Go]
              |                                               |
      +-------+-------+                               +-------+-------+
      |               |                               |               |
[Modèle <= 8B]  [Modèle <= 14B]                 [Modèle <= 70B] [Modèle > 70B]
      |               |                               |               |
   v  v            v  v                            v  v            v  v
[Q4_K_M GGUF]    [Q4/Q5 GGUF]                    [FP16 / INT8]   [AWQ / GPTQ]
  sur Ollama       sur vLLM                        sur vLLM        sur vLLM
```

---

## 5. Tableaux comparatifs

### Formats de précision numérique
| Format | Bits | Exposant | Mantisse | Avantage | Inconvénient |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **FP32** | 32 | 8 | 23 | Haute précision, stable | Lent, gourmand en mémoire |
| **FP16** | 16 | 5 | 10 | Rapide, économe | Risque d'overflow / underflow |
| **BF16** | 16 | 8 | 7 | Même plage que FP32, stable | Légère baisse de précision locale |
| **INT8** | 8 | - | - | Très rapide, compact | Nécessite une calibration des poids |
| **INT4** | 4 | - | - | Empreinte minimale | Légère dégradation de la perplexité |

### Comparatif des méthodes de fine-tuning (PEFT)
| Méthode | Paramètres entraînés | Besoin en VRAM | Complexité | Risque d'oubli catastrophique |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | 100% | Élevé ($6\times P$) | Faible (standard) | Très élevé |
| **LoRA** | 0.1% à 1% | Moyen | Moyen (config) | Très faible |
| **QLoRA** | 0.1% à 1% | Faible | Moyen | Très faible |
| **Prefix Tuning** | < 1% | Moyen | Élevé | Faible |

### Comparatif des moteurs d'inférence
| Moteur | Usage principal | Atouts | Limites |
| :--- | :--- | :--- | :--- |
| **llama.cpp / Ollama** | Inférence locale & grand public | CPU/GPU hybride, léger | Moins performant pour de grands batchs |
| **vLLM** | Production, APIs, haute charge | PagedAttention, batching continu | Configuration mémoire rigide, GPU requis |
| **TGI (HuggingFace)** | Déploiements cloud industriels | Sécurisé, support natif HF | Moins flexible hors écosystème HF |
| **TensorRT-LLM** | Performance maximale (NVIDIA) | Très optimisé pour GPU Hopper | Compilation complexe, propriétaire |

---

## 6. Liste de contrôle d'un projet d'IA
- [ ] **Données** : Volume suffisant, nettoyage fait, RGPD respecté, pas de biais majeurs.
- [ ] **Infrastructure** : Budget GPU estimé, VRAM calculée, modèle de quantification choisi.
- [ ] **Évaluation** : Métriques claires définies (Perplexité, BLEU, Exactitude, Ragas).
- [ ] **Sécurité** : Filtres anti-injection configurés, accès API sécurisés, guardrails en place.
- [ ] **Déploiement** : Versioning des poids, monitoring de la dérive des données (data drift).

---
*Fin de l'aide-mémoire — À coller près de votre poste de travail.*
*Lien connexe : Consultez le chapitre sur l'infrastructure [10-infrastructure/README.md](10-infrastructure/README.md) et l'inférence [09-inference-optimisation/README.md](09-inference-optimisation/README.md) pour approfondir ces notions.*
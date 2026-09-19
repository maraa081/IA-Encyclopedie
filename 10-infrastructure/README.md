# Infrastructure : GPU, VRAM, cloud

> L'entraînement et l'inférence des modèles de langage modernes reposent sur une infrastructure matérielle massivement parallèle et une gestion fine de la mémoire vidéo, conditionnant à la fois la faisabilité technique et la viabilité économique des projets.

## 1. Pourquoi des GPU pour l'IA ?

Les processeurs graphiques (GPU) ont été détournés de leur usage initial (rendu 3D) car leur architecture massivement parallèle est parfaitement adaptée aux calculs requis par les réseaux de neurones.

### Architecture massivement parallèle
Contrairement aux CPU (Central Processing Units) qui possèdent un petit nombre de cœurs (8 à 64) conçus pour exécuter des tâches séquentielles complexes, les GPU possèdent des milliers de cœurs simples. Ces cœurs sont regroupés en *Streaming Multiprocessors* (SM).

### Modèle SIMT (Single Instruction Multiple Threads)
Le GPU utilise le modèle SIMT : il exécute la même instruction sur des milliers de fils d'exécution (threads) simultanément. C'est idéal pour traiter des vecteurs et des matrices, où la même opération mathématique doit être appliquée à des millions de valeurs numériques.

### Multiplication matricielle (GEMM)
Le cœur des réseaux de neurones, et particulièrement des Transformers, repose sur le produit de matrices. Cette opération, appelée GEMM (General Matrix Multiplication), se prête parfaitement à la parallélisation. Un GPU peut diviser une multiplication de matrices géantes en milliers de petites multiplications traitées en parallèle.

### Bande passante mémoire
L'enjeu majeur en IA n'est pas seulement la vitesse de calcul brute (FLOPS), mais la vitesse à laquelle les données (poids du modèle, activations) sont acheminées vers les cœurs de calcul.
- **HBM (High Bandwidth Memory)** : Les GPU haut de gamme (A100, H100) utilisent une mémoire empilée verticalement appelée HBM. Elle offre une bande passante de plusieurs téraoctets par seconde (TB/s), là où un CPU plafonne à quelques dizaines de GB/s sur sa RAM DDR standard.

### Tensor Cores (NVIDIA)
Depuis l'architecture Volta, NVIDIA intègre des unités matérielles spécialisées appelées *Tensor Cores*. Ces unités sont optimisées pour effectuer des opérations de multiplication-accumulation matricielle (D = A * B + C) en une seule instruction. Elles supportent nativement des formats de précision réduite (FP16, BF16, FP8, INT8), multipliant par 10 ou 20 le débit de calcul par rapport aux cœurs CUDA classiques.

---

## 2. Comparaison des architectures de calcul

Le paysage du calcul haute performance est diversifié, chaque architecture ayant ses propres forces.

| Caractéristique | CPU (Intel/AMD) | GPU (NVIDIA/AMD) | TPU (Google) | NPU/ASIC (Edge) | FPGA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Parallélisme** | Faible (Cœurs lourds) | Très élevé (SIMT) | Très élevé (Systolique) | Élevé (Ciblé) | Variable |
| **Flexibilité** | Totale (Universel) | Élevée (CUDA/C++) | Faible (JAX/TF) | Très faible | Élevée |
| **VRAM / HBM** | RAM Système lente | Très rapide (HBM3) | Très rapide (HBM) | Limitée (SRAM) | Variable |
| **Précision** | FP64/FP32 | FP16/BF16/FP8/INT8 | BF16/INT8 | INT8/INT4 | Custom |
| **Consommation** | 150-300W | 300-700W | Élevée | Très faible (mW-W)| Moyenne |
| **Usage Type** | Logique, DB, OS | Training, Serving | Google Cloud Scale | Mobile, IoT | Prototypage |

---

## 3. Écosystèmes logiciels : CUDA, ROCm et Triton

Le matériel n'est rien sans la pile logicielle permettant de l'exploiter.

### CUDA (NVIDIA)
CUDA est la plateforme de calcul parallèle propriétaire de NVIDIA. Sa domination est due à sa maturité et à son écosystème de bibliothèques ultra-optimisées :
- **cuDNN** : Algorithmes pour les réseaux de neurones profonds.
- **cuBLAS** : Opérations d'algèbre linéaire.
- **NCCL** (NVIDIA Collective Communications Library) : Optimise les échanges de données entre plusieurs GPU.

### ROCm (AMD)
L'alternative ouverte d'AMD. Bien que moins mature que CUDA, ROCm progresse rapidement grâce au support de frameworks comme PyTorch. Elle permet d'exécuter des modèles IA sur les puces MI200/MI300.

### Triton (OpenAI)
Triton est un langage et un compilateur open-source conçu par OpenAI pour simplifier l'écriture de "kernels" GPU (morceaux de code exécutés sur la puce) hautement performants sans avoir à coder en CUDA C++. Il permet d'atteindre des performances proches du code expert avec une syntaxe proche de Python.

### Fusion de kernels
Une optimisation logicielle critique consiste à "fusionner" plusieurs opérations successives (ex: Linéaire + ReLU + Dropout) en un seul passage sur le GPU. Cela réduit drastiquement les transferts coûteux entre la VRAM et les registres du processeur.

---

## 4. VRAM : ce qui la remplit et comment l'estimer

La mémoire vidéo (VRAM) est la ressource la plus précieuse. Un manque de VRAM conduit immédiatement à une erreur fatale `Out Of Memory` (OOM).

### Les composants de la consommation mémoire
1. **Poids du modèle** : Le volume brut des paramètres.
   - Exemple : Un modèle de 7 milliards de paramètres (7B) en précision FP16 utilise : $7 \times 10^9 \times 2 \text{ octets} = 14 \text{ GB}$.
2. **États d'optimiseur** : Lors de l'entraînement avec Adam, on stocke la moyenne et la variance de chaque gradient. Cela nécessite 8 octets supplémentaires par paramètre si on travaille en FP32.
3. **Gradients** : Les dérivés nécessaires à la mise à jour des poids (2 à 4 octets par paramètre).
4. **Activations** : Résultats intermédiaires stockés pendant la passe "forward" pour être réutilisés pendant la passe "backward". Leur taille dépend du batch size et de la longueur de séquence.
5. **KV Cache** : Réservé à l'inférence, il stocke l'attention des tokens passés.

### Formule d'estimation de VRAM pour l'entraînement
Pour un entraînement standard avec l'optimiseur Adam en précision mixte :
$$\text{VRAM}_{\text{entraînement}} \approx \Phi \times (2_{\text{weights}} + 2_{\text{gradients}} + 12_{\text{optimizer}}) + \text{Activations}$$
Où $\Phi$ est le nombre de milliards de paramètres. Soit environ 16 à 18 GB par milliard de paramètres, hors activations.

### Techniques d'optimisation
- **Offloading** : Déplacer temporairement les poids ou les états de l'optimiseur vers la RAM système (CPU) quand ils ne sont pas utilisés.
- **Gradient Checkpointing** : Ne pas stocker toutes les activations. On en jette certaines et on les recalcule si besoin. Cela économise de la VRAM au prix d'un temps de calcul accru (~30% plus lent).
- **ZeRO (Zero Redundancy Optimizer)** : Technique de DeepSpeed qui partitionne les états de l'optimiseur, les gradients et les poids sur plusieurs GPU, permettant d'entraîner des modèles bien plus grands que la mémoire d'une seule carte.

---

## 5. Précision numérique : FP32, BF16, FP8 et plus

Le choix du format de données influe sur la stabilité de l'entraînement et la vitesse d'inférence.

- **FP32 (Single Précision)** : 32 bits. Très précis, large plage. Trop lent et gourmand pour le calcul massif.
- **TF32 (Tensor Float 32)** : Format interne NVIDIA. Garde l'exposant du FP32 pour la stabilité, mais réduit la mantisse. Permet d'accélérer les calculs sans changer le code utilisateur.
- **FP16 (Half Précision)** : 16 bits. Risque d'underflow/overflow (valeurs trop petites ou trop grandes qui deviennent zéro ou infini). Nécessite un "loss scaling".
- **BF16 (Brain Float 16)** : Créé par Google. Possède le même exposant que le FP32 mais une mantisse courte. C'est le standard actuel car il est très stable et ne nécessite pas de gestion complexe de l'échelle.
- **FP8 (E4M3 / E5M2)** : Nouveau standard supporté par les architectures Hopper (H100). Offre un débit de calcul doublé par rapport au FP16 avec une précision suffisante pour l'entraînement à grande échelle.
- **INT8 / INT4** : Quantification pour l'inférence. Permet de diviser par 4 ou 8 l'usage mémoire.

---

## 6. Parallélisme d'entraînement et communication

Passer d'un GPU à un cluster de milliers de puces nécessite des stratégies de distribution complexes.

### Les types de parallélisme
1. **Data Parallelism (DP)** : On réplique le modèle entier sur chaque GPU. Chaque GPU traite une partie différente des données. À la fin de chaque étape, les GPU synchronisent leurs gradients.
2. **Tensor Parallelism (TP)** : On découpe les matrices de poids d'une même couche sur plusieurs GPU. Nécessite une communication constante et ultra-rapide (NVLink).
3. **Pipeline Parallelism (PP)** : On découpe le modèle par couches (ex: couches 1-20 sur GPU 1, 21-40 sur GPU 2). Le calcul circule comme dans une usine.
4. **Sequence Parallelism** : On découpe la séquence de tokens elle-même sur plusieurs GPU.
5. **Expert Parallelism** : Utilisé pour les modèles MoE (Mixture of Experts) pour répartir les experts sur différents nœuds.

### Protocoles de communication
- **NVLink** : Pont matériel haute vitesse entre GPU dans un même serveur.
- **InfiniBand / RoCE** : Réseau ultra-basse latence pour connecter des serveurs entre eux. Indispensable pour éviter que le réseau ne devienne le goulot d'étranglement lors de l'entraînement distribué.

---

## 7. Coût d'entraînement et infrastructure matérielle

### Ordres de grandeur
Entraîner un modèle comme Llama 3 405B nécessite des ressources colossales :
- Des dizaines de milliers de GPU H100.
- Une consommation électrique comparable à celle d'une petite ville.
- Un coût matériel estimé à plusieurs centaines de millions de dollars.

### Évolutions des générations NVIDIA
- **V100 (Volta)** : Introduction des Tensor Cores.
- **A100 (Ampere)** : Standard de l'industrie, introduction de la mémoire HBM2e (40/80 GB).
- **H100 (Hopper)** : Accélération majeure du format FP8 et du Transformer Engine.
- **B200 (Blackwell)** : Promet des gains massifs en performance énergétique et en capacité mémoire (HBM3e).

---

## 8. Énergie et empreinte carbone

Les datacenters IA sont des gouffres énergétiques.
- **PUE (Power Usage Effectiveness)** : Ratio de l'énergie totale sur l'énergie IT brute. Un PUE de 1.10 est considéré comme excellent.
- **Refroidissement liquide (Direct-to-Chip)** : Devenu standard pour les serveurs de H100. L'eau circule directement sur les plaques de refroidissement des puces, éliminant les ventilateurs bruyants et inefficaces.
- **Empreinte carbone** : Estimer les émissions d'un entraînement dépend de la source d'énergie locale. Un entraînement au Québec (Hydro) ou en France (Nucléaire) est nettement moins carboné qu'en Allemagne ou au Texas.

---

## 9. Infrastructures Cloud et alternatives locales

### Les types d'instances
- **AWS (Amazon)** : Instances P4d/P5 (H100), Trainium (ASIC propriétaire).
- **GCP (Google)** : Instances A3 (H100) et surtout les TPU v4/v5p.
- **Azure (Microsoft)** : Surtout utilisé par OpenAI pour ses entraînements massifs.
- **RunPod / Lambda Labs** : Alternatives agiles et moins chères pour la location de GPU à l'heure.

### Instances Spot
Technique d'économie majeure : louer des machines dont personne ne veut à un instant T. Réduction de prix de 70 à 90%. Risque : la machine peut être réclamée par le fournisseur à tout moment. Nécessite des systèmes de sauvegarde de checkpoints très fréquents.

### Local (Laptop / Station de travail)
- **Unified Memory (Apple Silicon)** : Les Mac M2/M3 Ultra permettent d'utiliser jusqu'à 192 GB de RAM comme VRAM, une aubaine pour faire tourner de gros modèles quantifiés sans acheter de GPU professionnels.
- **Laptop** : Limité à l'inférence de petits modèles (8B) quantifiés en 4 bits (INT4).

---

## 10. Alternatives à NVIDIA

Bien que NVIDIA possède 95% du marché, des alternatives émergent :
- **AMD Instinct (MI300X)** : Très compétitif sur le plan matériel (plus de VRAM HBM3 que le H100). Le défi reste logiciel (ROCm vs CUDA).
- **TPU (Google)** : Très performant pour l'entraînement massif mais enferme l'utilisateur dans l'écosystème Google Cloud.
- **Trainium/Inferentia (AWS)** : Puces à coût réduit pour les utilisateurs fidèles à Amazon.

---

## Exemple : Monitoring GPU en Python

Le script suivant permet de surveiller l'utilisation de la VRAM en temps réel pendant un processus d'inférence.

```python
import subprocess
import time

def get_gpu_memory_usage():
    """Appelle nvidia-smi pour extraire l'usage mémoire."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding='utf-8'
        )
        used, total = map(int, output.strip().split(','))
        return used, total
    except Exception:
        return 0, 0

def monitor_training(interval=1):
    print("Démarrage du monitoring VRAM...")
    print("Temps (s) | Usage (MB) | Total (MB) | %")
    print("-" * 40)
    start_time = time.time()
    try:
        while True:
            used, total = get_gpu_memory_usage()
            if total > 0:
                percent = (used / total) * 100
                elapsed = int(time.time() - start_time)
                print(f"{elapsed:9} | {used:10} | {total:10} | {percent:.1f}%")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring arrêté.")

if __name__ == "__main__":
    monitor_training()
```

---

## Ce qu'il faut retenir
- Les GPU dominent l'IA grâce au parallélisme massif et à la bande passante mémoire (HBM).
- La VRAM est la contrainte limitante : elle stocke poids, états d'optimiseur, gradients et activations.
- Le format **BF16** est le standard de stabilité pour l'entraînement actuel.
- L'infrastructure cloud (instances spot) permet de réduire les coûts, mais le local (Apple Silicon) est viable pour l'inférence.
- NVIDIA garde son avantage grâce à l'écosystème **CUDA**, mais AMD et les TPU Google sont des alternatives sérieuses.

## Erreurs fréquentes / idées reçues
- *"Un GPU de jeu (RTX 4090) est inutile pour l'IA"* -> Faux, c'est une excellente puce pour le prototypage et l'inférence grâce à ses 24 GB de VRAM et ses Tensor Cores.
- *"La puissance du GPU se mesure en TFLOPS"* -> Faux, pour l'inférence de LLM, c'est la bande passante mémoire (GB/s) qui est le goulot d'étranglement.
- *"Plus on met de GPU, plus l'entraînement est rapide"* -> Pas toujours. Au-delà d'un certain point, le temps passé à faire communiquer les GPU entre eux dépasse le gain de calcul.

## Pour aller plus loin
- [chapitre 09](../09-inference-optimisation/README.md) pour les détails sur l'optimisation logicielle de l'inférence.
- Documentation NVIDIA sur le [Transformer Engine](https://docs.nvidia.com/).
- Comparaisons de performance GPU sur [Lambda Labs](https://lambdalabs.com/blog/).
- Rapports sur l'efficacité énergétique des datacenters IA (Green500).

# Agents IA

> Un agent IA est un système dans lequel un grand modèle de langage ne se contente pas de
> répondre : il décide d'actions, appelle des outils, observe les résultats et répète
> jusqu'à atteindre un objectif. Ce chapitre explique la boucle agentique, le tool
> calling, les frameworks, les questions de fiabilité et de sécurité.

## 1. Agent, chatbot, workflow : les distinctions

Trois objets souvent confondus sous le mot "agent".

- **Chatbot** : le modèle génère une réponse textuelle à un message. Une entrée, une
  sortie, aucun effet sur le monde extérieur.
- **Workflow** : un enchaînement d'étapes défini à l'avance par le développeur (appeler
  l'outil A, puis B, puis formater). Le LLM peut remplir une étape, mais le chemin est
  fixe et déterministe.
- **Agent** : le modèle choisit lui-même la séquence d'actions. Le nombre d'étapes n'est
  pas fixé à l'avance, le chemin dépend des observations. C'est cette autonomie de
  décision qui définit l'agent, pas la présence d'outils.

Un système réel combine souvent les trois : un workflow de haut niveau qui appelle des
sous-agents là où une décision dynamique est nécessaire. Le choix se fait sur le besoin
de contrôle : plus l'agent est autonome, plus il est flexible mais plus il est imprévisible
et difficile à déboguer.

Comparatif :

| Dimension | Chatbot | Workflow | Agent |
|---|---|---|---|
| Chemin d'exécution | fixe | fixe | décidé par le modèle |
| Outils | non | oui | oui |
| Prédictibilité | élevée | élevée | faible |
| Coût | minimal | faible | variable, souvent élevé |
| Cas d'usage | question/réponse | pipeline métier stable | tâche ouverte, exploration |

## 2. Boucle perception / raisonnement / action

Le cœur d'un agent est une boucle. À chaque tour : il perçoit l'état (message, résultat
d'outil), il raisonne sur la prochaine action, il agit (appelle un outil), puis observe
le résultat, et recommence jusqu'à un critère d'arrêt (objectif atteint, budget épuisé,
demande d'aide).

```
        +-------------------------------------------+
        |                                           |
        v                                           |
  [Perception] --> [Raisonnement LLM] --> [Action / outil]
   message,          "quelle est la         appel API,
   résultat          prochaine étape ?"     recherche, code
   d'outil                |                      |
        ^                 v                      v
        |            [Decision]            [Observation]
        |          répondre / agir /       résultat de
        +--------- se stopper <--------    l'action

  Conditions d'arrêt : objectif atteint, budget de tokens épuisé,
  nombre maximal d'étapes atteint, ou demande de confirmation a l'humain.
```

Le raisonnement peut être plus ou moins structuré :

- **ReAct** (Reasoning and Acting) : le modèle alterne une pensée (reasoning) et une
  action (acting), en écrivant explicitement son raisonnement avant chaque appel d'outil.
  Simple et efficace, c'est le cycle de base.
- **Plan-and-exécute** : le modèle produit d'abord un plan complet en plusieurs étapes,
  puis exécute chaque étape. Moins de risque de dérive, mais le plan initial peut être
  mauvais et difficile à corriger en cours de route.
- **Réflexion** (reflexion / self-critique) : l'agent critique sa propre production et
  itère pour l'améliorer. Coûteux (plusieurs passes) mais améliore les tâches de
  rédaction ou de code.

```
ReAct : boucle pensée -> action -> observation
  Pensée  : "Je dois connaître la météo a Paris pour adapter le conseil."
  Action  : get_weather(city="Paris")
  Obs.    : {"temp": 12, "pluie": true}
  Pensée  : "Il pleut, je recommande un parapluie."
  Action  : respond(...)

Plan-and-exécute :
  Plan    : 1) chercher les ventes 2024  2) comparer a 2023  3) rédiger le résumé
  Exécute : étape 1 -> étape 2 -> étape 3
```

## 3. Tool calling et function calling

Le `tool calling` (ou `function calling`) est le mécanisme par lequel un LLM demande
l'exécution d'une fonction au lieu de produire directement du texte. Le modèle ne
"exécute" rien lui-même : il émet une intention structurée, et c'est le code appelant
qui exécute réellement l'outil et renvoie le résultat.

On décrit chaque outil au modèle par un schéma JSON :

```json
{
  "name": "search_flights",
  "description": "Recherche des vols entre deux villes à une date donnée",
  "parameters": {
    "type": "object",
    "properties": {
      "from": { "type": "string", "description": "Ville de départ (code IATA)" },
      "to":   { "type": "string", "description": "Ville d'arrivee (code IATA)" },
      "date": { "type": "string", "description": "Date au format AAAA-MM-JJ" }
    },
    "required": ["from", "to", "date"]
  }
}
```

Points clés d'implémentation :

- **tool_choice** : contrôle si le modèle doit appeler un outil. Valeurs typiques :
  `auto` (il décide), `required` (il doit en appeler un), ou un outil nommé précisément.
  Utile pour forcer un comportement sans prompt fragile.
- **Gestion des erreurs** : une API peut échouer, renvoyer un format inattendu ou un
  message vide. Il faut renvoyer au modèle un message d'erreur clair plutôt que planter,
  pour qu'il puisse retenter ou changer d'approche. Ne jamais faire confiance à ce que
  l'outil renvoie.
- **Validation des arguments** : les arguments produits par le modèle doivent être
  validés (types, plages, permissions) avant exécution. Un modèle peut halluciner des
  paramètres plausibles mais dangereux.
- **Ne jamais exécuter aveuglément** : surtout pour les outils à effet (envoi d'email,
  suppression de fichier, paiement). Voir la section fiabilité.

```python
# Pseudo-code d'une boucle d'agent avec tool calling
messages = [{"role": "user", "content": "Quel temps a Paris ?"}]
while True:
    reponse = llm.chat(messages, tools=TOOLS, tool_choice="auto")
    messages.append(reponse)
    if reponse.finish_reason == "tool_calls":
        for appel in reponse.tool_calls:
            args = valider(appel.arguments)          # vérifier avant d'exécuter
            resultat = executer_outil(appel.name, args)
            messages.append({"role": "tool", "content": resultat})
    else:
        break   # le modèle a produit une réponse finale
print(reponse.content)
```

## 4. Mémoire

Un LLM n'a aucune mémoire entre deux appels : tout ce qu'il "sait" de la conversation
doit être dans le contexte envoyé. La mémoire d'un agent est donc un problème
d'ingénierie, pas une propriété du modèle.

- **Court terme** : l'historique récent vit directement dans la fenêtre de contexte. Il
  faut le tronquer quand il devient trop long, au risque de perdre des informations
  anciennes.
- **Long terme** : les informations durables (préférences, faits, expériences passées)
  sont stockées dans une base vectorielle et retrouvées par similarité, exactement comme
  dans un RAG (voir le [chapitre 07](../07-rag/README.md)).
- **Résumé** : on résume périodiquement les échanges anciens pour conserver l'essentiel
  en peu de tokens.
- **Compaction** : quand le contexte approche de sa limite, on fusionne les plus anciens
  échanges en un résumé compact et on repart avec ce résumé plus les échanges récents.
  C'est le mécanisme utilisé par les agents longue durée.

```
Mémoire d'agent :

   [Contexte courant]        [Résumé compressé]        [Base vectorielle]
   échanges récents          faits anciens             souvenirs récupérables
   (fidélité maximale)       (perte de détail)         (par similarité)
        |                         |                         |
        +-- max ~50% du -------- + -- compaction ---------- + -- retrieval a
            contexte               périodique                la demande
```

## 5. MCP (Model Context Protocol)

Avant MCP, chaque intégration (connecter un LLM à une base de données, à un dépôt Git,
à un service de fichiers) demandait un code d'adaptation spécifique à ce LLM et à cet
outil. Le résultat était un entrelacs de connecteurs non réutilisables.

**MCP** est un protocole standardisé, proposé par Anthropic, qui joue le rôle d'un
"port USB" pour les outils : on écrit un serveur une fois, et n'importe quel client
compatible peut le consommer.

- **Serveur MCP** : expose des capacités (tools, resources, prompts) selon un format
  standard.
- **Client MCP** : l'application qui héberge le modèle (IDE, agent, chat) se connecte aux
  serveurs et présente leurs capacités au modèle.

```
[Client MCP]  <-- protocole standard -->  [Serveur MCP]
  IDE / agent                              base de données
  héberge le LLM                           Git, fichiers
        |                                    navigateur
        v
   le LLM voit les outils du serveur comme n'importe quel outil
```

L'intérêt : découplage. On peut brancher un serveur MCP de fichiers internes à n'importe
quel client, et changer de modèle sans réécrire les connecteurs. Le revers : un serveur
MCP malveillant est une surface d'attaque directe (accès aux données, exécution),
d'où la nécessité de ne connecter que des serveurs de confiance (voir le
[chapitre 13](../13-securite/README.md)).

## 6. Agents de code et computer use

Les agents de code sont l'application la plus mature des agents. Ils lisent et écrivent
des fichiers, lancent des tests, interprètent les erreurs et proposent des correctifs.

| Agent | Environnement | Particularité |
|---|---|---|
| Claude Code | terminal | agent en ligne de commande, lit/écrit le dépôt |
| Cursor | IDE | édition assistée, contexte du projet |
| GitHub Copilot | IDE | complétion et chat intégrés |
| OpenClaw | hôte / messagerie | agent personnel avec outils et canaux |
| navigateurs agentiques | web | pilotage d'interface graphique |

Le **computer use** pousse l'idée plus loin : l'agent voit une capture d'écran et
contrôle souris et clavier pour utiliser des logiciels sans API. Puissant (tout logiciel
devient automatisable) mais fragile et lent, car il imite un humain au lieu d'appeler un
service.

Points d'attention pour les agents de code :

- **Sandbox obligatoire** : exécuter le code de l'agent dans un conteneur isolé, sans
  accès au réseau ou aux secrets de production par défaut.
- **Révision humaine** : un diff proposé par un agent doit être relu avant d'être
  fusionné, surtout s'il touche la sécurité ou l'infrastructure.
- **Budget** : un agent qui boucle sur des tests peut consommer beaucoup de tokens très
  vite ; il faut un plafond de coût.

## 7. Frameworks

| Framework | Éditeur | Type | Points forts |
|---|---|---|---|
| LangChain | communauté | bibliothèque | écosystème très large, connecteurs nombreux |
| LangGraph | LangChain | graphe d'états | contrôle fin des boucles et transitions |
| LlamaIndex | LlamaIndex | données | orienté RAG et indexation |
| CrewAI | CrewAI | multi-agents | rôles et collaboration simples à décrire |
| AutoGen | Microsoft | multi-agents | dialogue entre agents, recherche |
| OpenAI Agents SDK | OpenAI | agents/tools | léger, natif tool calling |
| Smolagents | Hugging Face | agents code | agents qui écrivent du Python |

Conseil : les frameworks évoluent vite et masquent parfois la logique réelle. Pour
apprendre, écrire d'abord une boucle d'agent à la main (une centaine de lignes) rend le
mécanisme évident ; on adopte ensuite un framework pour les connecteurs et la gestion
d'état. Voir aussi le [chapitre 15](../15-ecosysteme/README.md).

## 8. Multi-agents

Un système multi-agents fait collaborer plusieurs modèles avec des rôles distincts :
un orchestrateur qui découpe le travail, des spécialistes qui l'exécutent, un vérificateur
qui contrôle.

- **Rôles** : séparation des responsabilités (chercheur, rédacteur, critique).
- **Orchestration** : l'orchestrateur distribue les tâches et compile les résultats.
- **Débats** : plusieurs agents argumentent et l'on retient la réponse la plus robuste ;
  améliore la qualité sur les tâches ambiguës, mais multiplie le coût.
- **Coûts** : chaque agent est un appel LLM ; n agents qui discutent font croître le
  coût rapidement (souvent de façon plus que linéaire).
- **Quand ça ne vaut pas le coup** : pour une tâche déterministe, un workflow ou une
  simple fonction Python est plus fiable, plus rapide et moins cher. Le multi-agents se
  justifie surtout quand les sous-tâches sont réellement indépendantes et bénéficient
  d'un contexte isolé.

```
Orchestrateur
   |--- sous-tâche A --> agent chercheur   --+
   |--- sous-tâche B --> agent rédacteur    --+--> synthèse --> agent critique
   |--- sous-tâche C --> agent vérificateur --+
        (coût total = somme de tous les appels, plus la synthèse)
```

## 9. Fiabilité et garde-fous

Un agent autonome échoue de façons nouvelles. Les problèmes les plus fréquents :

- **Boucles infinies** : l'agent retente la même action qui échoue. Il faut une limite
  dure sur le nombre d'étapes et sur les répétitions identiques.
- **Dérive** : après beaucoup d'étapes, l'agent s'éloigne de l'objectif initial. On
  milite pour réinjecter l'objectif dans le contexte périodiquement.
- **Propagation d'erreur** : une erreur précoce (mauvais document, mauvaise décision)
  se répercute sur tout ce qui suit.
- **Garde-fous** : validation des entrées et sorties, filtres de sécurité, listes
  d'outils autorisés par contexte.
- **Sandbox** : tout code exécuté passe par un environnement isolé.
- **Permissions** : principe du moindre privilège ; un agent ne doit disposer que des
  droits strictement nécessaires.
- **Human-in-the-loop** : pour les actions irréversibles (paiement, suppression, envoi),
  demander une confirmation humaine explicite.
- **Budget de tokens** : plafonner le coût par tâche est aussi important qu'une limite
  d'étapes ; c'est le garde-fou qui protège la facture.

```
Garde-fous d'exécution (pseudo-configuration)

  max_steps            : 25
  max_tokens_par_tâche : 200 000
  outils_autorises     : [recherche, lecture_fichier]      # pas d'ecriture
  actions_sensibles    : [envoi_email, paiement, suppression]  -> confirmation humaine
  sandbox              : docker, réseau_désactive, pas de secrets
```

## 10. Évaluation des agents

Évaluer un agent ne se réduit pas à vérifier la réponse finale : c'est la qualité de la
trajectoire qui compte (a-t-il utilisé les bons outils, dans le bon ordre, sans détours
coûteux ?). On parle d'évaluation par trajectoire (`trajectory évaluation`).

| Benchmark | Domaine | Ce qu'il mesure |
|---|---|---|
| SWE-bench | code | résolution d'issues GitHub réelles |
| GAIA | usage général | tâches multi-étapes avec outils et web |
| WebArena | web | navigation et tâches dans des sites simulés |
| AgentBench | varié | raisonnement et actions sur plusieurs environnements |
| tau-bench | dialogue + outils | respect de politiques et de contraintes |

Tous ces benchmarks mesurent un **taux de succès** sur des tâches vérifiables, pas une
qualité subjective. Ordre de grandeur : les meilleurs agents résolvent aujourd'hui une
fraction des tâches du monde réel, pas leur totalité ; les chiffres dépendent fortement
du domaine et évoluent vite, il faut donc lire les scores avec leur date et leur
protocole.

## 11. Sécurité des agents

La sécurité des agents est traitée en détail au [chapitre 13](../13-securite/README.md).
Le risque principal est l'**injection de prompt** : un agent qui lit des données
externes (web, emails, documents, sorties d'outils) peut y rencontrer des instructions
malveillantes qu'il interprète comme légitimes. Comme l'agent a des outils, l'injection
peut se traduire par des actions concrètes (exfiltration de données, envoi de messages,
suppression de fichiers), et non par du simple texte.

Règles minimales :

- Traiter tout contenu externe comme non fiable, y compris les résultats d'outils.
- Ne jamais donner à un agent des permissions qu'un humain n'aurait pas dans le même rôle.
- Combiner défense applicative (validation, sandbox) et défense humaine (confirmation des
  actions sensibles).
- Un LLM aligné n'est pas un garde-fou suffisant : l'alignement réduit les comportements
  indésirables, il ne bloque pas une injection bien construite.

## Ce qu'il faut retenir
- Un agent se définit par sa capacité à décider lui-même une séquence d'actions, pas par
  la simple présence d'outils.
- La boucle perception / raisonnement / action se répète jusqu'à un critère d'arrêt.
- Le tool calling fait émettre au modèle une intention structurée en JSON que le code
  exécute ; le modèle n'exécute jamais rien lui-même.
- ReAct alterne pensée et action ; plan-and-exécute établit un plan avant d'agir.
- La mémoire d'un agent est un problème d'ingénierie : contexte, résumé, compaction,
  base vectorielle.
- MCP standardisé les connecteurs d'outils et découple le modèle des intégrations.
- Les agents de code sont l'application la plus mature, mais exigent sandbox et révision.
- Le multi-agents améliore certaines tâches au prix d'un coût qui croît vite.
- La fiabilité passe par des limites d'étapes, un budget de tokens, une sandbox, des
  permissions minimales et un human-in-the-loop.
- L'injection de prompt est le risque de sécurité principal des agents.

## Erreurs fréquentes / idées reçues
- Idée reçue : un agent est un chatbot avec des outils -> la différence est l'autonomie
  de décision sur la séquence d'actions.
- Idée reçue : plus d'agents font forcément un meilleur système -> le coût et la
  complexité augmentent souvent plus vite que la qualité.
- Idée reçue : le tool calling sécurise les actions -> il décrit l'intention ; la
  sécurité vient de la validation et des permissions côté code.
- Idée reçue : il suffit d'un modèle bien aligné pour éviter les abus -> l'alignement est
  statistique, pas une barrière dure ; l'injection de prompt contourne les garde-fous
  probabilistes.
- Idée reçue : un agent autonome n'a pas besoin de limite d'étapes -> sans plafond, il
  peut boucler et consumer un budget de tokens important.
- Idée reçue : le computer use remplace les API -> il est plus lent et plus fragile, à
  réserver aux logiciels sans API.

## Pour aller plus loin
- [Chapitre 07 - RAG](../07-rag/README.md) : la mémoire long terme d'un agent repose sur
  un RAG.
- [Chapitre 13 - Sécurité](../13-securite/README.md) : injection de prompt, guardrails,
  red teaming.
- [Chapitre 15 - Écosystème](../15-ecosysteme/README.md) : frameworks, libs et
  fournisseurs.
- [Chapitre 16 - Pratique](../16-pratique/README.md) : tutoriel de construction d'agent.
- Documentation MCP (Model Context Protocol) : modelcontextprotocol.io.
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022,
  arXiv:2210.03629).
- Documentation LangGraph et OpenAI Agents SDK : boucles d'agents et tool calling.
- Benchmark SWE-bench : évaluation des agents de code sur des issues réelles.

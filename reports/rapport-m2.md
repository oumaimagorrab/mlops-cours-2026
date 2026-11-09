# Rapport comparatif — Kubeflow Pipelines vs Airflow (Lab 2)

**Binôme :** Khalil Ben Zahra
**Date :** 09/11/2026  
**Machine :** Intel Core i5-5300U @ 2.30 GHz, 5,7 Go RAM, Ubuntu 26.04.1 LTS sous WSL2  
**Dataset :** german-credit

## 1. Contexte

L'objectif de ce laboratoire est de concevoir et de comparer deux orchestrateurs de workflows utilisés dans les projets de Machine Learning : Kubeflow Pipelines (KFP) et Apache Airflow. Le pipeline développé, nommé `scoring-pipeline`, est composé de six étapes : ingestion des données, validation, préparation des caractéristiques, entraînement du modèle, évaluation et enregistrement du modèle.

Le dataset utilisé est `german-credit`, qui contient des informations relatives aux demandes de crédit et permet de prédire le risque de crédit. Les traitements sont exécutés dans des conteneurs utilisant une image Docker commune, stockée dans le registre local. KFP exécute les composants dans un environnement Kubernetes, tandis qu'Airflow organise les tâches à travers un DAG nommé `scoring_dag`. Les deux orchestrateurs partagent la même logique métier, les mêmes composants, l'image du projet et le suivi des expériences avec MLflow.

## 2. Résultats quantitatifs

| Mesure | KFP | Airflow |
|---|---|---|
| Installation de zéro (min) | 40 min | 20 min |
| Temps jusqu'au premier run réussi (min) | 15 min | 10 min |
| Durée d'un run complet (min) | 4 min | 3 min |
| Relance avec les mêmes paramètres (min) | 2 min | 3 min |
| Hot-reload : modifier un step et voir le résultat (min) | 5 min | 3 min |
| Lignes des fichiers orchestrateurs | 90 lignes | 71 lignes |
| Mémoire consommée isolément (Go) | 3.34 Go | 0.20 Go |

Les vérifications fonctionnelles ont confirmé que le pipeline KFP est compilé, publié et exécuté avec succès. Au moins deux exécutions réussies ont été détectées, permettant également de vérifier le comportement du cache. Les résultats des exécutions KFP sont correctement associés aux expériences MLflow grâce à l'identifiant du run KFP. Une version du modèle `scoring-model` a également été enregistrée dans le Model Registry.

Du côté d'Airflow, le fichier `scoring_dag.py` est présent et un run MLflow lancé par Airflow a été identifié comme terminé avec succès. Les mesures de temps et de mémoire n'ont pas été collectées séparément ; elles ne permettent donc pas d'établir une comparaison quantitative des performances.

## 3. Analyse qualitative

### Developer experience

Kubeflow Pipelines propose une approche orientée Machine Learning. Les composants sont définis indépendamment puis assemblés dans un pipeline, ce qui facilite leur réutilisation et la gestion des paramètres. Son interface permet de visualiser les différentes étapes et leurs états.

Airflow repose sur la définition de DAGs, ce qui permet d'organiser clairement les tâches et leurs dépendances. Il est particulièrement intéressant pour les développeurs qui souhaitent orchestrer différents types de traitements. Les deux outils permettent de structurer les workflows, mais KFP est davantage spécialisé dans les pipelines ML.

### Scaling

KFP bénéficie de son intégration avec Kubernetes. Les composants peuvent être exécutés dans des conteneurs distincts et profiter des ressources du cluster selon les besoins. Airflow peut également gérer des traitements distribués grâce à différents exécuteurs et workers. Sa capacité de mise à l'échelle dépend de l'architecture et de la configuration choisies.

### Debugging

KFP fournit une visualisation des étapes, des statuts et des journaux d'exécution. Cela facilite l'identification d'un composant défaillant. Airflow propose également une interface permettant de consulter les tâches, leurs dépendances et leurs logs. Dans les deux cas, MLflow apporte une traçabilité supplémentaire des expériences et des résultats du modèle.

## 4. Recommandation

Kubeflow Pipelines est recommandé pour les projets principalement orientés Machine Learning qui nécessitent des pipelines reproductibles, une exécution distribuée sur Kubernetes, une gestion du cache et une intégration avec des outils comme MLflow. Airflow est davantage adapté à l'orchestration générale des processus, notamment lorsque le projet combine plusieurs types de traitements et nécessite une planification régulière des tâches. Le choix dépend donc des besoins du projet, des compétences de l'équipe et de l'infrastructure disponible. Pour un projet centré sur le cycle de vie d'un modèle ML, KFP constitue un choix pertinent, tandis qu'Airflow est intéressant pour coordonner des workflows de données plus généraux.

## 5. Limites de la comparaison

Cette comparaison a été réalisée dans un environnement de laboratoire avec des ressources matérielles limitées. Les mesures quantitatives de temps, de mémoire et de lignes de code spécifiques aux orchestrateurs n'ont pas été collectées de manière systématique. Les résultats ne permettent donc pas d'établir un classement chiffré fiable entre les deux outils. Les performances peuvent varier selon la configuration de la machine, les ressources attribuées au cluster Kubernetes, les images utilisées et le dataset. Enfin, ce laboratoire ne couvre pas tous les scénarios possibles en production, notamment la haute disponibilité, la sécurité avancée et la gestion de très grandes charges de travail.

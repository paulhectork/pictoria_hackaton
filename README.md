# Hackaton Pictoria (19-20.12.24)

code et documentation produits pendant le hackaton Pictoria à la BnF, les 19 et 20 décembre 2024.

développé avec [Marina Hervieu](https://github.com/MapunaH) et [Sarah Marcq](https://github.com/s-marcq).

## Objectifs

l'objectif de la chaîne de traitement était de développer un modèle de vision pour la classification et
labellisation automatique d'images en utilisant [`panoptic`](https://github.com/CERES-Sorbonne/Panoptic).
Le jeu de données contenait 40.000 images issues des archives du Web de la BnF sur le centenaire de la
Première guerre mondiale.

## Chaîne de traitement

- avec `panoptic`, et son implémentation de `CLIP`, on réalise un premier clustering (classification des images
en catégories, sans labellisation). puis, on labellise les clusters en donnant un titre à chacun. cette opération
a été réalisée sur 14.000 images, ce qui a permis d'identifier 42 classes.

- deux exports ont été réalisés, qui forment deux datasets pour l'entraînement d'un modèle deux vision: le premier
contient les données de `train`, le deuxième de `validation`. la structure des exports panoptic est:

|path|label|
|----|-----|
|chemin absolu vers le fichier image|classe à laquelle appartient l'image|

- à partir de là, `create_datasets.py` permet de produire 2 datasets en réorganisant les fichiers images.

- ensuite, `pictoria_model.ipynb` permet de générer le modèle

## Structure

```
/
|_create_datasets.py    : produire le dataset selon la table contenant les classes panoptic
|_pictoria_model.ipynb : générer le modèle à partir de ces datasets
|_data/                       : jeux de données
  |_<metadata-table-name.csv> : le CSV exporté de panoptic
  |_dataset_train/            : jeu de données de train
  |_dataset_valid/            : jeu de données de validation
```

## License

GNU GPL 3.0

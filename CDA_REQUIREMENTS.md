# Notes de cadrage CDA

Ce document reprend les notes fournies pour orienter la suite du projet.
Les références numérotées de ces notes ne sont pas accompagnées de leurs sources :
les exigences exactes du dossier et du jury restent à vérifier dans le référentiel
fourni par l'école. Il ne constitue pas une attestation de conformité.

## Correspondance avec l'application actuelle

| Besoin | État actuel | Travail restant |
| --- | --- | --- |
| Comptes, connexion et sessions | Implémentés dans `src/web/`, mots de passe hachés avec scrypt | Passer à Argon2 ou bcrypt si ce choix est retenu dans le cahier des charges, avec migration des anciens hachages |
| Rôles USER / ADMIN | Pas de rôle administrateur | Définir les permissions et les fonctions d'administration avant de les ajouter |
| Base SQL et relations | Modèles SQLAlchemy, migrations Alembic, PostgreSQL prévu pour l'exécution locale | Valider sur PostgreSQL et préparer les MCD/MLD du schéma effectivement utilisé |
| Studio personnel | Projets privés, téléversement, analyse, édition du prompt, demande de génération | Valider l'inférence réelle avec les modèles et le GPU |
| Historique et galerie | Historique et images organisés par projet | Ajouter une galerie globale, des filtres et les favoris |
| Profil et préférences | Email visible ; pas de formulaire de préférences | Définir puis ajouter les préférences modifiables |
| Suppression | Suppression d'un projet et nettoyage de ses fichiers ; suppression des sources temporaires | Ajouter la suppression individuelle des images et de l'audio avec une politique explicite pour les analyses associées |
| Sécurité | Sessions HttpOnly, CSRF, contrôle de propriété, ORM, validation taille/extension, décodage dans le worker, rendu texte sans HTML utilisateur | Renforcer la validation du contenu des fichiers et compléter les protections avant une exposition publique |

## Les quatre concepts métier principaux

Les notes proposent `Users`, `Tracks`, `Prompts` et `Generations`. Le code actuel
sépare certaines responsabilités pour conserver la traçabilité scientifique :

- **Utilisateur** : `users`, avec les sessions dans `sessions`.
- **Morceau** : `audio_assets`, rattaché à un projet appartenant à un utilisateur ;
  ses analyses et caractéristiques sont conservées dans `analyses`.
- **Prompt** : texte et paramètres enregistrés dans `generations` ; il n'existe
  pas encore de table `prompts` indépendante.
- **Génération** : `generations` et son fichier dans `generated_images`.

Les tables `moods`, `analysis_moods` et `generation_moods` distinguent la
prédiction du modèle des choix créatifs de l'utilisateur. Elles matérialisent
des associations plusieurs-à-plusieurs. Les autres tables techniques conservent
les versions de modèles et les demandes de suppression de fichiers.

Le schéma actuel compte 12 tables physiques. Les « 4 à 5 tables principales »
des notes peuvent servir à présenter les concepts métier, mais ne décrivent pas
le schéma physique existant. Si une limite stricte est imposée par l'école,
il faudra adapter le modèle explicitement, sans présenter un MLD différent du code.

## Préparation du dossier de projet

Prévoir un résumé français et anglais, le contexte personnel rédigé par le candidat,
le périmètre et les besoins, les rôles et cas d'utilisation, un planning réel,
les maquettes, les MCD/MLD, l'architecture, des extraits de code expliqués,
les preuves de tests, une recherche technique en anglais avec synthèse française,
et le bilan personnel.

Les dates du Gantt, le parcours, les décisions et les résultats doivent refléter
le travail réellement effectué. Les captures devront distinguer les essais
avec fixtures des exécutions réelles du pipeline. L'application actuelle est
locale ; un déploiement SaaS reste une évolution, pas une fonctionnalité livrée.

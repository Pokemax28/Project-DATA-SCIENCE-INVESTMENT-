import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("viridis")

DATA_PATH = "Data/dvf_clean.parquet"
FILTERED_PATH = "Data/dvf_filtered.parquet"

# ==========================
# 1️⃣ Chargement
# ==========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("⚠️ Fichier dvf_clean.parquet introuvable !")

df = pd.read_parquet(DATA_PATH)
print(f"✅ {len(df):,} lignes chargées depuis {DATA_PATH}")

# ==========================
# 2️⃣ Nettoyage / filtrage assoupli
# ==========================
df_clean = df.copy()

# Vérifie présence colonnes
required_cols = {"valeur_fonciere", "surface_reelle_bati"}
if not required_cols.issubset(df_clean.columns):
    raise ValueError(f"Colonnes manquantes : {required_cols - set(df_clean.columns)}")

# Retire les valeurs manquantes
df_clean = df_clean[df_clean["surface_reelle_bati"].notna()]
df_clean = df_clean[df_clean["valeur_fonciere"].notna()]

# Garde uniquement les surfaces et valeurs réalistes
df_clean = df_clean[
    (df_clean["surface_reelle_bati"] > 8) &
    (df_clean["surface_reelle_bati"] < 5000) &
    (df_clean["valeur_fonciere"] > 1000) &
    (df_clean["valeur_fonciere"] < 5e7)
]

# Calcule le prix/m²
df_clean["prix_m2"] = df_clean["valeur_fonciere"] / df_clean["surface_reelle_bati"]
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=["prix_m2"])

print(f"📉 {len(df_clean):,} lignes après nettoyage des surfaces/prix valides.")

# Sauvegarde
df_clean.to_parquet(FILTERED_PATH, compression="snappy")
print(f"💾 Dataset filtré sauvegardé : {FILTERED_PATH}")

# ==========================
# 3️⃣ GRAPHE 3 — Répartition des types de biens
# ==========================
if "type_local" in df_clean.columns:
    counts = df_clean["type_local"].dropna().value_counts()
    if not counts.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title("Répartition des types de biens vendus")
        plt.xlabel("Type de bien")
        plt.ylabel("Nombre de ventes")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Aucune donnée exploitable dans 'type_local'.")
else:
    print("⚠️ Colonne 'type_local' absente du dataset.")

# ==========================
# 4️⃣ GRAPHE 6 — Prix au m² par type de bien
# ==========================
if {"type_local", "prix_m2"}.issubset(df_clean.columns):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_clean, x="type_local", y="prix_m2", showfliers=False)
    plt.title("Prix au m² selon le type de bien")
    plt.xlabel("Type de bien")
    plt.ylabel("Prix au m² (€)")
    plt.ylim(0, df_clean["prix_m2"].quantile(0.95))  # limite 95% pour lisibilité
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Colonnes nécessaires manquantes pour le boxplot (graphe 6).")

# ==========================
# 5️⃣ GRAPHE 7 — Top 10 communes les plus chères
# ==========================
if {"commune", "prix_m2"}.issubset(df_clean.columns):
    top_communes = (
        df_clean.groupby("commune", as_index=False)
        .agg(nb=("prix_m2", "count"), median_prix_m2=("prix_m2", "median"))
        .query("nb >= 30")  # au moins 30 ventes pour éviter les biais
        .sort_values("median_prix_m2", ascending=False)
        .head(10)
    )

    if not top_communes.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=top_communes, x="median_prix_m2", y="commune")
        plt.title("Top 10 communes les plus chères (prix médian au m²)")
        plt.xlabel("Prix médian au m² (€)")
        plt.ylabel("Commune")
        plt.tight_layout()
        plt.show()

        print("\n💡 Top 10 communes les plus chères :")
        print(top_communes)
    else:
        print("⚠️ Pas assez de communes avec nb≥30 pour le graphe 7.")
else:
    print("⚠️ Colonnes manquantes pour le graphe 7.")

# ==========================
# 6️⃣ GRAPHE 8 — Relation surface / valeur foncière
# ==========================
if {"surface_reelle_bati", "valeur_fonciere"}.issubset(df_clean.columns):
    # On prend un échantillon pour accélérer l’affichage
    sample = df_clean.sample(min(len(df_clean), 20000), random_state=42)
    plt.figure(figsize=(9, 5))
    plt.scatter(
        sample["surface_reelle_bati"],
        sample["valeur_fonciere"],
        s=8,
        alpha=0.3
    )
    plt.title("Relation Surface (m²) vs Valeur foncière (€)")
    plt.xlabel("Surface réelle bâtie (m²)")
    plt.ylabel("Valeur foncière (€)")
    plt.xlim(0, df_clean["surface_reelle_bati"].quantile(0.99))
    plt.ylim(0, df_clean["valeur_fonciere"].quantile(0.99))
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Colonnes manquantes pour le graphe 8.")

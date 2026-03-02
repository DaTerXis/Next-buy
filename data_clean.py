import pandas as pd

print("DÉMARRAGE DU DATA CLEANING GLOBAL")

print("1/3 - ettoyage et fusion du catalogue")
df_products = pd.read_csv('products.csv')
df_aisles = pd.read_csv('aisles.csv')
df_departments = pd.read_csv('departments.csv')

df_catalog = pd.merge(df_products, df_aisles, on='aisle_id', how='left')
df_catalog = pd.merge(df_catalog, df_departments, on='department_id', how='left')


print("2/3 - nettoyage de l'historique des commandes")
noms_colonnes = ['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']
df_orders = pd.read_csv('orders.csv', skiprows=1, names=noms_colonnes)

df_orders = df_orders.dropna(subset=['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day'])

# Logique de la première commande 
condition_premiere_commande = (df_orders['order_number'] == 1.0) & (df_orders['days_since_prior_order'].isna())
df_orders.loc[condition_premiere_commande, 'days_since_prior_order'] = 0.0
df_orders = df_orders.dropna(subset=['days_since_prior_order'])

# Optimisation en entiers 
cols_to_int = ['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']
df_orders[cols_to_int] = df_orders[cols_to_int].astype(int)


print("3/3 - Lecture du gros fichier et création du Master Dataset...")
dtypes_opti = {
    'order_id': 'Int32',
    'product_id': 'Int32',
    'add_to_cart_order': 'Int16',
    'reordered': 'Int8'
}
df_order_products = pd.read_csv('order_products.csv', dtype=dtypes_opti)

# On fusionne TOUT 
df_master = pd.merge(df_order_products, df_orders, on='order_id', how='inner')
df_master = pd.merge(df_master, df_catalog, on='product_id', how='inner')

# Sauvegarde finale
print("💾 Sauvegarde en cours (ça peut prendre 1 à 2 minutes)...")
df_master.to_csv('master_dataset.csv', index=False)

print("✅ SUCCÈS TOTAL ! Le fichier 'master_dataset.csv' est prêt !")

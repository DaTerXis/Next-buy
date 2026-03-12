"""
Feature Engineering Module for NextBuy Project

Provides comprehensive feature engineering for predicting user purchase behavior.
Creates 50+ engineered features from user history, product popularity, 
user-product interactions, recency, and cart position patterns.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


class FeatureEngineer:
    """
    Generates feature-engineered dataset for NextBuy recommendation/prediction models.
    
    Features created:
    - User History (16 features): user behavior patterns
    - Product Popularity (14 features): global product statistics
    - User-Product Interaction (17 features): specific interaction patterns
    - Recency (3 features): time-based features
    
    Total: 50 engineered features
    """
    
    def __init__(self, df):
        """
        Initialize the FeatureEngineer with a master dataset.
        
        Args:
            df (pd.DataFrame): Master dataset with merged order, product, and user data
        """
        self.df = df.copy()
        self.df_features = None
        
    def create_all_features(self, verbose=False):
        """
        Pipeline to create all features in sequence.
        
        Args:
            verbose (bool): Print progress information
            
        Returns:
            pd.DataFrame: Features dataset with all engineered features
        """
        if verbose:
            print("Creating user history features...")
        df = self._create_user_features()
        
        if verbose:
            print("Creating product popularity features...")
        df = self._create_product_features(df)
        
        if verbose:
            print("Creating user-product interaction features...")
        df = self._create_user_product_features(df)
        
        if verbose:
            print("Creating recency features...")
        df = self._create_recency_features(df)
        
        self.df_features = df
        return df
    
    def _create_user_features(self, df=None):
        """Create 16 user history features."""
        if df is None:
            df = self.df.copy()
        
        # User aggregations
        user_agg = df.groupby('user_id').agg({
            'order_id': 'nunique',  # total orders
            'product_id': 'nunique',  # unique products
            'add_to_cart_order': 'mean',  # avg basket size
            'reordered': 'sum',  # total reorders
        }).rename(columns={
            'order_id': 'user_total_orders',
            'product_id': 'user_unique_products',
            'add_to_cart_order': 'user_avg_basket_size',
            'reordered': 'user_total_reorders'
        })
        
        # Basket size stats
        order_sizes = df.groupby(['user_id', 'order_id'])['product_id'].count().reset_index()
        order_sizes.columns = ['user_id', 'order_id', 'basket_size']
        
        basket_stats = order_sizes.groupby('user_id')['basket_size'].agg([
            ('user_max_basket_size', 'max'),
            ('user_min_basket_size', 'min'),
            ('user_std_basket_size', 'std')
        ])
        
        user_agg = user_agg.join(basket_stats)
        
        # Days between orders
        days_between = df.groupby(['user_id', 'order_id'])['days_since_prior_order'].first().reset_index()
        days_stats = days_between.groupby('user_id')['days_since_prior_order'].agg([
            ('user_avg_days_between_orders', 'mean'),
            ('user_std_days_between_orders', 'std')
        ])
        
        user_agg = user_agg.join(days_stats)
        
        # Reorder rate
        user_agg['user_reorder_rate'] = (
            user_agg['user_total_reorders'] / user_agg['user_total_orders']
        ).fillna(0)
        
        # Order frequency
        user_agg['user_order_frequency'] = (
            user_agg['user_total_orders'] / (user_agg['user_avg_days_between_orders'].fillna(1) + 1)
        )
        
        # Product diversity
        user_agg['user_product_diversity'] = (
            user_agg['user_unique_products'] / user_agg['user_total_orders']
        ).fillna(0)
        
        # Preferred day of week and hour
        dow_pref = df.groupby('user_id')['order_dow'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        hour_pref = df.groupby('user_id')['order_hour_of_day'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
        
        user_agg['user_preferred_dow'] = dow_pref
        user_agg['user_preferred_hour'] = hour_pref
        
        # Max order number and unique departments/aisles
        user_agg['user_max_order_number'] = df.groupby('user_id')['order_number'].max()
        user_agg['user_unique_departments'] = df.groupby('user_id')['department_id'].nunique()
        user_agg['user_unique_aisles'] = df.groupby('user_id')['aisle_id'].nunique()
        
        # Merge back to original
        df = df.merge(user_agg.reset_index(), on='user_id', how='left')
        
        # Fill NaNs
        numeric_cols = [c for c in df.columns if c.startswith('user_')]
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def _create_product_features(self, df):
        """Create 14 product popularity features."""
        
        # Product aggregations
        product_agg = df.groupby('product_id').agg({
            'order_id': 'nunique',  # orders_count
            'user_id': 'nunique',  # users_count
            'reordered': 'sum',  # reorders_count
        }).rename(columns={
            'order_id': 'product_orders_count',
            'user_id': 'product_users_count',
            'reordered': 'product_reorders_count'
        })
        
        # Reorder rate
        product_agg['product_reorder_rate'] = (
            product_agg['product_reorders_count'] / product_agg['product_orders_count']
        ).fillna(0)
        
        # Cart position stats
        cart_stats = df.groupby('product_id')['add_to_cart_order'].agg([
            ('product_avg_cart_position', 'mean'),
            ('product_std_cart_position', 'std'),
            ('product_min_cart_position', 'min'),
            ('product_max_cart_position', 'max')
        ])
        
        product_agg = product_agg.join(cart_stats)
        
        # Popularity rank and score
        product_agg['product_popularity_rank'] = product_agg['product_orders_count'].rank(ascending=False).astype(int)
        product_agg['product_popularity_score'] = (
            (product_agg['product_orders_count'] - product_agg['product_orders_count'].min()) /
            (product_agg['product_orders_count'].max() - product_agg['product_orders_count'].min() + 1)
        )
        
        # Loyalty score
        product_agg['product_loyalty_score'] = product_agg['product_reorder_rate'].fillna(0)
        
        # Cart position indicators
        product_agg['product_is_early_cart'] = (product_agg['product_avg_cart_position'] <= 3).astype(int)
        product_agg['product_is_late_cart'] = (product_agg['product_avg_cart_position'] > 10).astype(int)
        
        # User penetration
        total_users = df['user_id'].nunique()
        product_agg['product_user_penetration'] = (
            product_agg['product_users_count'] / total_users
        )
        
        # Merge back
        df = df.merge(product_agg.reset_index(), on='product_id', how='left')
        
        # Fill NaNs
        numeric_cols = [c for c in df.columns if c.startswith('product_')]
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def _create_user_product_features(self, df):
        """Create 17 user-product interaction features."""
        
        # User-product aggregations
        up_agg = df.groupby(['user_id', 'product_id']).agg({
            'order_id': 'nunique',  # order_count
            'reordered': 'sum',  # reorder_count
            'add_to_cart_order': ['mean', 'std', 'min', 'max'],
            'order_number': ['min', 'max']
        }).reset_index()
        
        up_agg.columns = [
            'user_id', 'product_id',
            'up_order_count',
            'up_reorder_count',
            'up_avg_cart_position',
            'up_cart_std',
            'up_min_cart_position',
            'up_max_cart_position',
            'up_first_order_number',
            'up_last_order_number'
        ]
        
        # Reorder rate
        up_agg['up_reorder_rate'] = (
            up_agg['up_reorder_count'] / up_agg['up_order_count']
        ).fillna(0)
        
        # Product tenure (order span)
        up_agg['up_product_tenure'] = (
            up_agg['up_last_order_number'] - up_agg['up_first_order_number'] + 1
        ).astype(int)
        
        # Purchase frequency
        user_total_orders = df.groupby('user_id')['order_id'].nunique().reset_index()
        user_total_orders.columns = ['user_id', 'total_orders']
        up_agg = up_agg.merge(user_total_orders, on='user_id', how='left')
        up_agg['up_purchase_frequency'] = (
            up_agg['up_order_count'] / up_agg['total_orders']
        ).fillna(0)
        up_agg = up_agg.drop(columns=['total_orders'])
        
        # Consistency score (low std = consistent)
        up_agg['up_consistency_score'] = (
            1.0 / (1.0 + up_agg['up_cart_std'].fillna(0))
        )
        
        # Early/late cart indicators
        up_agg['up_typically_early'] = (up_agg['up_avg_cart_position'].fillna(0) <= 5).astype(int)
        up_agg['up_cart_consistency'] = (up_agg['up_cart_std'].fillna(0) < 3).astype(int)
        
        # Merge back
        df = df.merge(up_agg, on=['user_id', 'product_id'], how='left')
        
        # Fill NaNs
        numeric_cols = [c for c in df.columns if c.startswith('up_')]
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def _create_recency_features(self, df):
        """Create 3 recency features."""
        
        # Days since first order per user
        first_order_date = df.groupby('user_id')['days_since_prior_order'].transform('min')
        df['days_since_first_order'] = df.groupby('user_id')['order_number'].transform(
            lambda x: (x.max() - x.min()) * 30  # Approximate: assume ~30 days apart on average
        )
        
        # Orders since first
        df['orders_since_first'] = df.groupby('user_id')['order_number'].transform(
            lambda x: x.max() - x.min()
        )
        
        # Recency score (inverse of order number)
        max_order_num = df['order_number'].max()
        df['order_recency_score'] = (
            (max_order_num - df['order_number']) / max_order_num
        )
        
        df[['days_since_first_order', 'orders_since_first', 'order_recency_score']] = \
            df[['days_since_first_order', 'orders_since_first', 'order_recency_score']].fillna(0)
        
        return df
    
    def get_feature_summary(self, df_features=None):
        """
        Print a summary of created features.
        
        Args:
            df_features (pd.DataFrame): Optional features dataframe. Uses self.df_features if not provided.
        """
        if df_features is None:
            df_features = self.df_features
            
        if df_features is None:
            print("No features created yet. Call create_all_features() first.")
            return
        
        original_cols = [
            'order_id', 'product_id', 'add_to_cart_order', 'reordered', 'user_id',
            'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order',
            'product_name', 'aisle_id', 'department_id', 'aisle', 'department'
        ]
        
        new_features = [c for c in df_features.columns if c not in original_cols]
        user_features = [c for c in new_features if c.startswith('user_')]
        product_features = [c for c in new_features if c.startswith('product_')]
        up_features = [c for c in new_features if c.startswith('up_')]
        other_features = [c for c in new_features if c not in user_features + product_features + up_features]
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Total new features created: {len(new_features)}")
        print(f"  • User history features: {len(user_features)}")
        print(f"  • Product popularity features: {len(product_features)}")
        print(f"  • User-product interaction features: {len(up_features)}")
        print(f"  • Recency features: {len(other_features)}")
        print(f"\nDataset shape: {df_features.shape}")
        print(f"Memory usage: {df_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("="*60 + "\n")
    
    def save_features(self, df_features, output_path):
        """
        Save the features dataset to CSV.
        
        Args:
            df_features (pd.DataFrame): Features dataframe to save
            output_path (str): Output CSV file path
        """
        df_features.to_csv(output_path, index=False)
        print(f"✓ Features saved to {output_path}")


if __name__ == '__main__':
    """
    Command-line interface for feature engineering.
    
    Usage:
        python feature_engineering.py           # Full dataset
        python feature_engineering.py sample    # Sample of 1000 users
    """
    import sys
    
    # Read master dataset
    print("Loading master dataset...")
    df_master = pd.read_csv('master_dataset.csv')
    print(f"Loaded {len(df_master):,} rows")
    
    # Check for sample mode
    sample_mode = len(sys.argv) > 1 and sys.argv[1] == 'sample'
    
    if sample_mode:
        print("Running in SAMPLE mode (1000 users)...")
        sample_users = df_master['user_id'].dropna().unique()[:1000]
        df_input = df_master[df_master['user_id'].isin(sample_users)].copy()
        output_file = 'features_dataset_sample.csv'
    else:
        print("Running on FULL dataset...")
        df_input = df_master.copy()
        output_file = 'features_dataset.csv'
    
    print(f"Input shape: {df_input.shape}")
    
    # Feature engineering
    engineer = FeatureEngineer(df_input)
    df_features = engineer.create_all_features(verbose=True)
    engineer.get_feature_summary(df_features)
    engineer.save_features(df_features, output_file)
    
    print(f"✅ Complete! Features saved to {output_file}")

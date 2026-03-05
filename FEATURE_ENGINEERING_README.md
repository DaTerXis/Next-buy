# Feature Engineering for NextBuy Project

## Overview

This module provides comprehensive feature engineering for predicting user purchase behavior in grocery ordering systems. It creates **50+ engineered features** from user history, product popularity, user-product interactions, recency, and cart position patterns.

## Features Created

### 1. User History Features (16 features)
Features that capture overall user behavior patterns:

- `user_total_orders` - Total number of orders placed by the user
- `user_total_products` - Total number of products ordered by the user
- `user_avg_basket_size` - Average number of items per order
- `user_max_basket_size` - Maximum basket size
- `user_min_basket_size` - Minimum basket size
- `user_avg_days_between_orders` - Average days between consecutive orders
- `user_std_days_between_orders` - Standard deviation of days between orders
- `user_reorder_rate` - Percentage of user's purchases that are reorders
- `user_preferred_dow` - User's most frequent day of week for ordering
- `user_preferred_hour` - User's most frequent hour of day for ordering
- `user_order_frequency` - Orders per day (activity level)
- `user_max_order_number` - Total number of orders placed
- `user_unique_products` - Number of unique products purchased
- `user_product_diversity` - Ratio of unique products to total products
- `user_unique_departments` - Number of unique departments shopped from
- `user_unique_aisles` - Number of unique aisles shopped from

### 2. Product Popularity Features (14 features)
Global product statistics and popularity metrics:

- `product_orders_count` - How many orders contain this product
- `product_users_count` - How many unique users bought this product
- `product_reorders_count` - Total number of times product was reordered
- `product_reorder_rate` - Percentage of purchases that are reorders
- `product_avg_cart_position` - Average position in shopping cart
- `product_std_cart_position` - Standard deviation of cart position
- `product_min_cart_position` - Minimum cart position
- `product_max_cart_position` - Maximum cart position
- `product_popularity_rank` - Rank by total orders (1 = most popular)
- `product_popularity_score` - Normalized popularity (0-1 scale)
- `product_loyalty_score` - How often it gets reordered (loyalty metric)
- `product_is_early_cart` - Binary: typically added in first 3 positions
- `product_is_late_cart` - Binary: typically added after 10th position
- `product_user_penetration` - Percentage of all users who bought this

### 3. User-Product Interaction Features (17 features)
Specific interaction patterns between users and products:

- `up_order_count` - How many times user ordered this product
- `up_reorder_count` - How many times user reordered this product
- `up_reorder_rate` - User's reorder rate for this specific product
- `up_avg_cart_position` - User's average cart position for this product
- `up_min_cart_position` - Earliest position user added this product
- `up_max_cart_position` - Latest position user added this product
- `up_first_order_number` - Which order number user first bought this
- `up_last_order_number` - Which order number user last bought this
- `up_purchase_frequency` - Ratio of orders containing this product
- `up_product_tenure` - How many orders span from first to last purchase
- `up_consistency_score` - How consistently user buys this product
- `up_cart_avg` - User's average cart position for this product
- `up_cart_std` - Standard deviation of cart position
- `up_cart_min` - Minimum cart position
- `up_cart_max` - Maximum cart position
- `up_cart_consistency` - Binary: whether user is consistent in cart position
- `up_typically_early` - Binary: user typically adds this early in cart

### 4. Recency Features (3 features)
Time-based features for modeling temporal patterns:

- `order_recency_score` - Recency score (higher for more recent orders)
- `days_since_first_order` - Days elapsed since user's first order
- `orders_since_first` - Number of orders since user's first order

## Usage

### Basic Usage (Sample Mode - Recommended for Testing)

```python
# Run with a sample of 1000 users for quick testing
python feature_engineering.py sample
```

This will create `features_dataset_sample.csv` with features for 1000 users.

### Full Dataset

```python
# Run on the complete dataset
python feature_engineering.py
```

This will create `features_dataset.csv` with features for all users.

### Programmatic Usage

```python
from feature_engineering import FeatureEngineer
import pandas as pd

# Load your master dataset
df_master = pd.read_csv('master_dataset.csv')

# Initialize feature engineer
engineer = FeatureEngineer(df_master)

# Create all features
df_with_features = engineer.create_all_features(verbose=True)

# Save to file
engineer.save_features(df_with_features, 'my_features.csv')

# Get summary
engineer.get_feature_summary(df_with_features)
```

### Using Specific Feature Sets

```python
# Create only user features
user_features = engineer._create_user_features()

# Create only product features
product_features = engineer._create_product_features()

# Create only user-product interaction features
up_features = engineer._create_user_product_features()
```

## Output

The script generates a CSV file with:
- All original columns from the master dataset
- 50+ engineered features
- Same number of rows as input dataset

**Sample Output:**
```
============================================================
NEXTBUY PROJECT - FEATURE ENGINEERING
============================================================

📂 Loading master dataset...
   Loaded 13,691,080 rows

🔧 Starting Feature Engineering...
  1/5 - Building user history features...
  2/5 - Building product popularity features...
  3/5 - Building user-product interaction features...
  4/5 - Building recency features...
  5/5 - Building cart position features...
  🔗 Merging all feature sets...

✅ Feature engineering complete! Total features: 64

============================================================
FEATURE ENGINEERING SUMMARY
============================================================

📊 Total Features: 64
   - Original columns: 14
   - User features: 16
   - Product features: 14
   - User-Product features: 17
   - Recency features: 3
   - Cart position features: 17
```

## Feature Engineering for Machine Learning

These features are designed for predicting:
- Whether a user will purchase a specific product in their next order
- When a user might make their next purchase
- What products to recommend to users

### Recommended Feature Groups for Models

**For Purchase Prediction:**
- User history features (purchase patterns)
- Product popularity features (global trends)
- User-product interaction features (specific affinity)
- Recency features (temporal patterns)

**For Recommendation Systems:**
- `up_purchase_frequency` (how often user buys product)
- `product_popularity_score` (popular products)
- `up_reorder_rate` (user loyalty to product)
- `user_product_diversity` (user's exploration tendency)

**For Time-to-Next-Order Prediction:**
- `user_avg_days_between_orders`
- `user_std_days_between_orders`
- `user_order_frequency`
- `days_since_first_order`

## Performance Considerations

- **Sample Mode:** Process 1000 users - runs in ~1 minute
- **Full Dataset:** Process all 200K+ users - runs in ~3-5 minutes
- **Memory:** Requires ~2-3GB RAM for full dataset
- **Output Size:** Full features dataset ~1-2GB

## Integration with Machine Learning Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load features
df = pd.read_csv('features_dataset.csv')

# Select feature columns (exclude identifiers and target)
feature_cols = [col for col in df.columns if col.startswith(('user_', 'product_', 'up_', 'order_'))]
feature_cols = [col for col in feature_cols if col not in ['user_id', 'product_id', 'order_id']]

# Prepare data
X = df[feature_cols]
y = df['reordered']  # or your target variable

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

## Extending the Feature Engineering

To add custom features, extend the `FeatureEngineer` class:

```python
class CustomFeatureEngineer(FeatureEngineer):
    def create_custom_features(self):
        # Add your custom feature engineering logic
        pass
    
    def create_all_features(self, verbose=True):
        df_features = super().create_all_features(verbose)
        # Add custom features
        custom_features = self.create_custom_features()
        df_features = df_features.merge(custom_features, ...)
        return df_features
```

## License

This feature engineering module is part of the NextBuy prediction project.

---

**Created:** March 2026  
**Author:** NextBuy Project Team  
**Version:** 1.0

"""
CAFE Best Features Library
Generated on 2025-06-29T22:21:38.840429
Contains 6 high-performing features
"""

import pandas as pd
import numpy as np

def feature_1_ad_count_in_display(df):
    """
    (ad_count_in_display: The number of ads shown in the same display instance.)
    Improvement: +0.0067 (good)
    Dataset: click ad advertising content recommendation quality
    """
    df['ad_count_in_display'] = df.groupby('display_id')['ad_id'].transform('count')
    return df

def feature_2_ad_in_display_count(df):
    """
    (ad_in_display_count: The number of ads shown in the same display instance.)
    Improvement: +0.0067 (good)
    Dataset: click ad advertising content recommendation quality
    """
    df['ad_in_display_count'] = df.groupby('display_id')['ad_id'].transform('count')
    return df

def feature_3_document_id_y(df):
    """
    (ad_document_popularity)
    Improvement: +0.0033 (moderate)
    Dataset: click ad advertising content recommendation quality
    """
    ad_doc_counts = df['document_id_y'].map(df['document_id_y'].value_counts())
    df['ad_document_popularity'] = ad_doc_counts.fillna(0)
    return df

def feature_4_ad_id(df):
    """
    (ad_display_count: The number of times a specific ad has been shown in the dataset.)
    Improvement: +0.0024 (moderate)
    Dataset: click ad advertising content recommendation quality
    """
    ad_display_counts = df['ad_id'].map(df['ad_id'].value_counts())
    df['ad_display_count'] = ad_display_counts
    return df

def feature_5_feature_interaction_1_6_display_order(df):
    """
    (feature_interaction_1_6_display_order: Interaction term combining two key-value features and the display order.)
    Improvement: +0.0017 (moderate)
    Dataset: click advertising
    """
    df['feature_interaction_1_6_display_order'] = np.log1p(df['features_kv_bits_constrained_1'] * df['features_kv_bits_constrained_6'] * df['display_order'])
    return df

def feature_6_interaction_display_order_feat6(df):
    """
    (Interaction between display order and a key feature)
    Improvement: +0.0011 (moderate)
    Dataset: click advertising
    """
    df['interaction_display_order_feat6'] = df['display_order'] * df['features_kv_bits_constrained_6']
    return df


import os
import numpy as np
import pandas as pd
import json
import ast
import xgboost as xgb
import joblib
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from imblearn.under_sampling import RandomUnderSampler


class DataProcessor:
    """Class for processing data according to configuration"""

    def __init__(self, config):
        self.config = config

    def preprocess_sequence(self, sequence):
        """Process sequence data and create n-grams"""
        if pd.isna(sequence) or not isinstance(sequence, str):
            return ""
        steps = sequence.split(" > ")
        n_grams = []
        for n in range(
            self.config["n_gram_range"][0], self.config["n_gram_range"][1] + 1
        ):
            n_grams.extend(
                [" > ".join(steps[i : i + n]) for i in range(len(steps) - n + 1)]
            )
        return " ".join(n_grams)

    def check_sequence_anomalies(self, sequence, rules):
        """Check for anomalies in sequences based on rules"""
        if pd.isna(sequence) or sequence == "#":
            return [0] * len(rules)
        steps = sequence.split(" > ")
        anomalies = []
        for rule in rules:
            if rule["type"] == "repeated_step":
                counter = Counter(steps)
                anomalies.append(int(counter[rule["step"]] >= rule["min_repetitions"]))
            elif rule["type"] == "contains":
                anomalies.append(int(rule["term"] in sequence))
        return anomalies

    def process_data(self, df):
        """Process input dataframe according to configuration"""
        if df.empty:
            print("Warning: Empty DataFrame passed to process_data")
            return df, {}

        # Replace '#' with np.nan for consistent handling of missing data
        df = df.replace("#", np.nan)

        # Process sequence columns
        for col in self.config["input_columns"].values():
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", "")
                df[f"processed_{col}"] = df[col].apply(self.preprocess_sequence)
            else:
                print(f"Warning: Column '{col}' not found in the dataset")

        # Combine processed sequences
        processed_cols = [
            f"processed_{col}"
            for col in self.config["input_columns"].values()
            if f"processed_{col}" in df.columns
        ]
        if processed_cols:
            df["combined_sequence"] = df[processed_cols].agg(" ".join, axis=1)
        else:
            print("Warning: No processed columns found for combining sequences")
            df["combined_sequence"] = ""

        # Process categorical columns
        label_encoders = {}
        for col in self.config["categorical_columns"]:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].fillna("Unknown"))
                label_encoders[col] = le
            else:
                print(f"Warning: Categorical column '{col}' not found in the dataset")

        # Process multi-value columns
        for col in self.config["multi_value_columns"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) and x != "#" else []
                )
                unique_values = set(value for values in df[col] for value in values)
                for value in unique_values:
                    df[f"{col}_{value}"] = df[col].apply(
                        lambda x: 1 if value in x else 0
                    )
            else:
                print(f"Warning: Multi-value column '{col}' not found in the dataset")

        # Process anomaly rules
        for column, rules in self.config["anomaly_rules"].items():
            print(f"Processing rules for column: {column}")
            if column in df.columns:
                anomaly_results = df[column].apply(
                    lambda x: self.check_sequence_anomalies(x, rules)
                )
                for i, rule in enumerate(rules):
                    col_name = f"{column}_anomaly_{i}"
                    df[col_name] = anomaly_results.apply(lambda x: x[i])
                    print(f"Created anomaly column: {col_name}")
            else:
                print(
                    f"Warning: Column '{column}' not found in the dataset. Skipping its rules."
                )

        print(f"Processed data shape: {df.shape}")
        return df, label_encoders


class FeatureEngineer:
    """Class for feature engineering"""

    def __init__(self, config):
        self.config = config

    def create_features(self, df, vectorizer=None, mlb=None):
        """Create features from processed data"""
        print(f"Type of df: {type(df)}")
        print(
            f"Columns in df: {df.columns if isinstance(df, pd.DataFrame) else 'Not a DataFrame'}"
        )

        # Ensure df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected df to be a DataFrame, but got {type(df)}")

        # Check if 'combined_sequence' exists
        if "combined_sequence" not in df.columns:
            print(
                "Warning: 'combined_sequence' not found in DataFrame. Available columns:",
                df.columns,
            )
            df["combined_sequence"] = df[self.config["input_columns"].values()].apply(
                lambda row: " ".join(row.astype(str)), axis=1
            )

        # Create TF-IDF features
        if vectorizer is None:
            vectorizer = TfidfVectorizer(lowercase=False)
            tfidf_features = vectorizer.fit_transform(df["combined_sequence"])
        else:
            tfidf_features = vectorizer.transform(df["combined_sequence"])

        # Get encoded categorical features
        cat_columns = [
            f"{col}_encoded"
            for col in self.config["categorical_columns"]
            if f"{col}_encoded" in df.columns
        ]

        if cat_columns:
            cat_features = df[cat_columns].values
        else:
            cat_features = np.array([]).reshape(df.shape[0], 0)

        # Handle multi-value features
        multi_value_columns = self.config["multi_value_columns"]
        if (
            multi_value_columns
            and len(multi_value_columns) > 0
            and multi_value_columns[0] in df.columns
        ):
            if mlb is None:
                mlb = MultiLabelBinarizer()
                multi_value_features = mlb.fit_transform(
                    df[multi_value_columns].iloc[:, 0]
                )
            else:
                multi_value_features = mlb.transform(df[multi_value_columns].iloc[:, 0])
        else:
            multi_value_features = np.array([]).reshape(df.shape[0], 0)

        # Get anomaly features
        anomaly_columns = [col for col in df.columns if "_anomaly_" in col]
        if not anomaly_columns:
            print(
                "No anomaly columns found. Check your anomaly rules in the configuration."
            )
            anomaly_features = np.array([]).reshape(df.shape[0], 0)
        else:
            print(f"Found anomaly columns: {anomaly_columns}")
            anomaly_features = df[anomaly_columns].values

        # Combine all features
        feature_components = [tfidf_features.toarray()]

        if cat_features.shape[1] > 0:
            feature_components.append(cat_features)

        if multi_value_features.shape[1] > 0:
            feature_components.append(multi_value_features)

        if anomaly_features.shape[1] > 0:
            feature_components.append(anomaly_features)

        combined_features = np.hstack(feature_components)

        print(f"Combined feature shape: {combined_features.shape}")

        # Feature stats
        feature_stats = {
            "num_features": combined_features.shape[1],
            "feature_density": np.mean(combined_features != 0),
            "feature_sparsity": np.mean(combined_features == 0),
            "tfidf_features": tfidf_features.shape[1],
            "categorical_features": cat_features.shape[1],
            "multi_value_features": multi_value_features.shape[1],
            "anomaly_features": anomaly_features.shape[1],
        }

        print("\nFeature statistics:")
        for key, value in feature_stats.items():
            print(f"  {key}: {value}")

        return vectorizer, mlb, combined_features, feature_stats

    def analyze_feature_importance(self, model, feature_names, top_n=10):
        """Analyze feature importance from trained model"""
        importance_dict = model.get_score(importance_type="gain")
        importance_df = pd.DataFrame(
            {
                "Feature": [
                    (
                        feature_names[int(k[1:])]
                        if k[1:].isdigit() and int(k[1:]) < len(feature_names)
                        else k
                    )
                    for k in importance_dict.keys()
                ],
                "Importance": list(importance_dict.values()),
            }
        )

        top_features = importance_df.sort_values(by="Importance", ascending=False).head(
            top_n
        )
        return top_features

    def suggest_features(self, df, importance_df=None):
        """Suggest additional features based on data analysis"""
        suggestions = []

        # Check for temporal patterns
        if "DISPUTE_ID" in df.columns:
            suggestions.append(
                "Consider adding temporal features like day of week or time of day"
            )

        # Check for network-based features
        if any("SEQUENCE" in col for col in df.columns):
            suggestions.append(
                "Consider adding network-based features like sequence length or complexity"
            )

        # Suggest embeddings for sequence data
        if any(col in df.columns for col in self.config["input_columns"].values()):
            suggestions.append(
                "Consider using advanced sequence embeddings like BERT or Word2Vec"
            )

        # Use importance to suggest feature engineering
        if importance_df is not None and not importance_df.empty:
            low_importance = importance_df.nsmallest(5, "Importance")
            suggestions.append(
                f"Consider removing low importance features: {', '.join(low_importance['Feature'].tolist())}"
            )

        return suggestions

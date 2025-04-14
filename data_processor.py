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

    # Add this method to the FeatureEngineer class in data_processor.py

    def analyze_feature_importance(self, model, feature_names, top_n=10):
        """Analyze feature importance from trained model"""
        try:
            # Check if the model has the feature_importances_ attribute (like RandomForest)
            if hasattr(model, 'feature_importances_'):
                importance_dict = {
                    feature_names[i]: float(imp)
                    for i, imp in enumerate(model.feature_importances_)
                    if i < len(feature_names)
                }
            # Or if it's an XGBoost model
            elif hasattr(model, 'get_score'):
                # Try to get feature importance using XGBoost's get_score method
                try:
                    importance_dict = model.get_score(importance_type="gain")
                    # Convert feature indices to feature names
                    importance_dict = {
                        feature_names[int(k[1:])] if k[1:].isdigit() and int(k[1:]) < len(feature_names) else k: v
                        for k, v in importance_dict.items()
                    }
                except Exception as e:
                    print(f"Error getting XGBoost feature importance: {e}")
                    # Fallback to a simple dictionary with placeholder values
                    importance_dict = {
                        feature_names[i]: 1.0 / (i + 1)  # Simple decreasing importance
                        for i in range(min(len(feature_names), top_n))
                    }
            else:
                # Fallback if no standard method is available
                print("Warning: Model doesn't have standard feature importance. Using placeholders.")
                importance_dict = {
                    feature_names[i]: 1.0 / (i + 1)  # Simple decreasing importance
                    for i in range(min(len(feature_names), top_n))
                }

            # Convert to DataFrame for easier handling
            import pandas as pd
            importance_df = pd.DataFrame({
                "Feature": list(importance_dict.keys()),
                "Importance": list(importance_dict.values())
            })

            # Sort by importance
            importance_df = importance_df.sort_values(by="Importance", ascending=False)

            # Get top N features
            top_features = importance_df.head(top_n)

            return top_features

        except Exception as e:
            print(f"Error analyzing feature importance: {str(e)}")
            # Return a minimal DataFrame as fallback
            import pandas as pd
            return pd.DataFrame({
                "Feature": feature_names[:min(len(feature_names), top_n)],
                "Importance": [1.0 / (i + 1) for i in range(min(len(feature_names), top_n))]
            })


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
        """Process input dataframe with robust error handling and verbose logging."""
        try:
            print(f"Processing data with shape: {df.shape}")

            # Check for required columns and add placeholder if missing
            required_columns = list(self.config["input_columns"].values()) + self.config["categorical_columns"]
            for col in required_columns:
                if col not in df.columns:
                    print(f"Warning: Required column '{col}' not found, adding placeholder")
                    df[col] = "#"

            # Replace '#' with np.nan for consistent handling of missing data
            df = df.replace("#", np.nan)

            # Process sequence columns
            for col_key, col in self.config["input_columns"].items():
                if col in df.columns:
                    print(f"Processing sequence column: {col}")
                    # Ensure data is string type
                    df[col] = df[col].astype(str).replace("nan", "").replace("None", "")

                    # Create processed version
                    df[f"processed_{col}"] = df[col].apply(lambda x:
                                                           self.preprocess_sequence(x) if x and x != "nan" else "")

                    print(f"Created processed column: processed_{col}")
                else:
                    print(f"Warning: Column '{col}' not found")

            # Combine processed sequences
            processed_cols = [
                f"processed_{col}"
                for col in self.config["input_columns"].values()
                if f"processed_{col}" in df.columns
            ]

            if processed_cols:
                print(f"Combining processed columns: {processed_cols}")
                df["combined_sequence"] = df[processed_cols].agg(" ".join, axis=1)
            else:
                print("Warning: No processed columns for combining")
                df["combined_sequence"] = ""

            # Process categorical columns
            label_encoders = {}
            for col in self.config["categorical_columns"]:
                if col in df.columns:
                    print(f"Encoding categorical column: {col}")
                    le = LabelEncoder()
                    # Handle missing values
                    filled_values = df[col].fillna("Unknown").astype(str)
                    df[f"{col}_encoded"] = le.fit_transform(filled_values)
                    label_encoders[col] = le
                else:
                    print(f"Warning: Categorical column '{col}' not found")

            # Process multi-value columns - special handling for the context data format
            for col in self.config["multi_value_columns"]:
                if col in df.columns:
                    print(f"Processing multi-value column: {col}")
                    # Safe conversion to list with proper error handling
                    df[col] = df[col].apply(
                        lambda x: self._safe_literal_eval(x) if pd.notna(x) else []
                    )

                    # Create indicator columns for unique values
                    all_values = set()
                    for values in df[col]:
                        if isinstance(values, list):
                            all_values.update(values)

                    print(f"Creating {len(all_values)} indicator columns for {col}")
                    for value in all_values:
                        df[f"{col}_{value}"] = df[col].apply(
                            lambda x: 1 if isinstance(x, list) and value in x else 0
                        )
                else:
                    print(f"Warning: Multi-value column '{col}' not found")

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

                        # Count occurrences
                        anomaly_count = df[col_name].sum()
                        print(f"Anomaly count for {col_name}: {anomaly_count}")
                else:
                    print(f"Warning: Column '{column}' not found for anomaly rules")

            print(f"Processed data shape: {df.shape}")
            return df, label_encoders

        except Exception as e:
            import traceback
            print(f"Error in data processing: {str(e)}")
            print(traceback.format_exc())
            # Return original data to allow pipeline to continue
            return df, {}

    def _safe_literal_eval(self, value):
        """Safely evaluate a string representation of a list."""
        import ast

        if not isinstance(value, str):
            return []

        value = value.strip()
        if not value or value == "#":
            return []

        try:
            if value.startswith("[") and value.endswith("]"):
                return ast.literal_eval(value)
            else:
                # Handle non-list string values
                return [value]
        except (SyntaxError, ValueError):
            # For malformed strings, return as a single element
            return [value]

class FeatureEngineer:
    """Class for feature engineering with robust error handling"""

    def __init__(self, config):
        self.config = config

    def create_features(self, df, vectorizer=None, mlb=None):
        """Create features from processed data with detailed logging and error handling."""
        try:
            print(f"Starting feature engineering on data shape: {df.shape}")

            # Validate input
            if not isinstance(df, pd.DataFrame):
                print(f"Warning: Expected DataFrame but got {type(df)}")
                if isinstance(df, tuple) and len(df) > 0 and isinstance(df[0], pd.DataFrame):
                    print("Using first element of tuple as DataFrame")
                    df = df[0]
                else:
                    raise ValueError(f"Cannot process input type: {type(df)}")

            # Check for combined_sequence column
            if "combined_sequence" not in df.columns:
                print("Warning: 'combined_sequence' not found, creating from input columns")
                input_cols = [col for col in self.config["input_columns"].values() if col in df.columns]
                df["combined_sequence"] = df[input_cols].astype(str).apply(" ".join, axis=1)

            # Create TF-IDF features with error handling
            print("Creating TF-IDF features")
            try:
                if vectorizer is None:
                    vectorizer = TfidfVectorizer(
                        lowercase=False,
                        max_features=100,  # Limit features for small datasets
                        ngram_range=(1, 3),
                        min_df=1
                    )
                    tfidf_features = vectorizer.fit_transform(df["combined_sequence"].fillna(""))
                else:
                    tfidf_features = vectorizer.transform(df["combined_sequence"].fillna(""))

                print(f"Created TF-IDF features with shape: {tfidf_features.shape}")
            except Exception as e:
                print(f"Error in TF-IDF vectorization: {str(e)}")
                # Create empty feature matrix as fallback
                tfidf_features = np.zeros((df.shape[0], 1))

            # Get encoded categorical features
            print("Processing categorical features")
            cat_columns = [
                f"{col}_encoded"
                for col in self.config["categorical_columns"]
                if f"{col}_encoded" in df.columns
            ]

            if cat_columns:
                cat_features = df[cat_columns].fillna(0).values
                print(f"Created categorical features with shape: {cat_features.shape}")
            else:
                print("No categorical features found")
                cat_features = np.zeros((df.shape[0], 1))

            # Handle multi-value features
            print("Processing multi-value features")
            multi_value_columns = self.config["multi_value_columns"]
            if multi_value_columns and len(multi_value_columns) > 0:
                # Find all binary indicator columns created during preprocessing
                indicator_cols = [col for col in df.columns if
                                  any(col.startswith(f"{mv_col}_") for mv_col in multi_value_columns)]

                if indicator_cols:
                    multi_value_features = df[indicator_cols].fillna(0).values
                    print(f"Using {len(indicator_cols)} indicator columns for multi-value features")
                elif multi_value_columns[0] in df.columns:
                    # Try to use the MultiLabelBinarizer approach
                    try:
                        if mlb is None:
                            mlb = MultiLabelBinarizer()
                            multi_value_features = mlb.fit_transform(df[multi_value_columns[0]].apply(
                                lambda x: x if isinstance(x, list) else []
                            ))
                        else:
                            multi_value_features = mlb.transform(df[multi_value_columns[0]].apply(
                                lambda x: x if isinstance(x, list) else []
                            ))
                    except Exception as e:
                        print(f"Error in multi-label binarization: {str(e)}")
                        multi_value_features = np.zeros((df.shape[0], 1))
                else:
                    print(f"Multi-value column {multi_value_columns[0]} not found")
                    multi_value_features = np.zeros((df.shape[0], 1))
            else:
                print("No multi-value columns configured")
                multi_value_features = np.zeros((df.shape[0], 1))

            # Get anomaly features
            print("Processing anomaly features")
            anomaly_columns = [col for col in df.columns if "_anomaly_" in col]
            if anomaly_columns:
                anomaly_features = df[anomaly_columns].fillna(0).values
                print(f"Created {len(anomaly_columns)} anomaly features")
            else:
                print("No anomaly columns found")
                anomaly_features = np.zeros((df.shape[0], 1))

            # Add length-based features for sequences
            print("Adding sequence length features")
            seq_columns = [col for col in self.config["input_columns"].values() if col in df.columns]
            length_features = []
            for col in seq_columns:
                df[f"{col}_length"] = df[col].astype(str).apply(
                    lambda x: len(x.split(" > ")) if pd.notna(x) and " > " in x else 0
                )
                length_features.append(f"{col}_length")

            if length_features:
                seq_length_features = df[length_features].values
                print(f"Created {len(length_features)} sequence length features")
            else:
                seq_length_features = np.zeros((df.shape[0], 1))

            # Combine all features
            feature_components = []
            if tfidf_features.shape[1] > 0:
                feature_components.append(tfidf_features.toarray())
            if cat_features.shape[1] > 0:
                feature_components.append(cat_features)
            if multi_value_features.shape[1] > 0:
                feature_components.append(multi_value_features)
            if anomaly_features.shape[1] > 0:
                feature_components.append(anomaly_features)
            if seq_length_features.shape[1] > 0:
                feature_components.append(seq_length_features)

            # Ensure we have at least one feature
            if not feature_components:
                print("Warning: No features created, adding placeholder feature")
                feature_components.append(np.zeros((df.shape[0], 1)))

            # Combine features
            combined_features = np.hstack(feature_components)
            print(f"Final combined feature shape: {combined_features.shape}")

            # Feature stats
            feature_stats = {
                "num_features": combined_features.shape[1],
                "tfidf_features": tfidf_features.shape[1],
                "categorical_features": cat_features.shape[1],
                "multi_value_features": multi_value_features.shape[1],
                "anomaly_features": anomaly_features.shape[1],
                "sequence_length_features": seq_length_features.shape[1],
                "feature_density": float(np.mean(combined_features != 0))
            }

            print("\nFeature statistics:")
            for key, value in feature_stats.items():
                print(f"  {key}: {value}")

            return vectorizer, mlb, combined_features, feature_stats

        except Exception as e:
            import traceback
            print(f"Error in feature engineering: {str(e)}")
            print(traceback.format_exc())

            # Create minimal feature set to allow pipeline to continue
            features = np.zeros((df.shape[0], 5))
            feature_stats = {"error": str(e), "num_features": 5}

            return vectorizer, mlb, features, feature_stats

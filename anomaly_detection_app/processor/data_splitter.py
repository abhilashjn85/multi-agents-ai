import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.under_sampling import RandomUnderSampler


class DataSplitter:
    """Class for splitting data with custom anomaly ratios and better error handling"""

    def __init__(self, config):
        self.config = config

    def custom_train_test_split(self, features, labels, test_size=0.2, anomaly_ratio=0.1):
        """Split data with custom anomaly ratio and proper error handling"""
        try:
            print(f"Splitting data with test_size={test_size}, anomaly_ratio={anomaly_ratio}")
            print(f"Input features shape: {features.shape}, labels shape: {labels.shape}")

            # Basic validation
            if len(features) != len(labels):
                raise ValueError(f"Features and labels must have same length. Got {len(features)} and {len(labels)}")

            # Ensure labels are binary
            unique_labels = np.unique(labels)
            if len(unique_labels) > 2:
                print(f"Warning: Expected binary labels but found {len(unique_labels)} unique values")
                # Convert to binary based on non-zero values
                labels = (labels != 0).astype(int)

            # Simple train/test split first
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, stratify=labels, random_state=42
            )

            # Handle case with very few anomalies
            n_anomaly_train = np.sum(y_train == 1)
            if n_anomaly_train < 2:
                print(f"Warning: Only {n_anomaly_train} anomalies in training set. Using basic stratified split.")
                print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
                print(f"Train anomalies: {n_anomaly_train}, Test anomalies: {np.sum(y_test == 1)}")
                return X_train, X_test, y_train, y_test

            # Apply custom anomaly ratio using undersampling
            try:
                # Calculate target counts for each class
                n_train = len(X_train)
                n_train_anomalies = int(n_train * anomaly_ratio)
                n_train_normal = n_train - n_train_anomalies

                # Make sure we don't request more anomalies than available
                n_anomaly_available = np.sum(y_train == 1)
                if n_train_anomalies > n_anomaly_available:
                    print(f"Warning: Requested {n_train_anomalies} anomalies but only {n_anomaly_available} available")
                    n_train_anomalies = n_anomaly_available
                    n_train_normal = n_train - n_train_anomalies

                # Make sure we don't request more normal samples than available
                n_normal_available = np.sum(y_train == 0)
                if n_train_normal > n_normal_available:
                    print(f"Warning: Requested {n_train_normal} normal samples but only {n_normal_available} available")
                    n_train_normal = n_normal_available

                # Undersample both classes
                sampling_strategy = {0: n_train_normal, 1: n_train_anomalies}
                print(f"Undersampling to get {sampling_strategy} instances of each class")

                undersampler = RandomUnderSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=42
                )
                X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"Error in undersampling: {str(e)}")
                print("Using original train/test split")
                X_train_resampled, y_train_resampled = X_train, y_train

            # Print split information
            print(
                f"Original data: {len(labels)} samples, {np.sum(labels == 1)} anomalies ({np.mean(labels) * 100:.2f}%)")
            print(
                f"Training data: {len(y_train_resampled)} samples, {np.sum(y_train_resampled == 1)} anomalies ({np.mean(y_train_resampled) * 100:.2f}%)")
            print(f"Test data: {len(y_test)} samples, {np.sum(y_test == 1)} anomalies ({np.mean(y_test) * 100:.2f}%)")

            return X_train_resampled, X_test, y_train_resampled, y_test

        except Exception as e:
            import traceback
            print(f"Error in data splitting: {str(e)}")
            print(traceback.format_exc())

            # Fall back to basic train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=42
            )
            print("Using fallback train/test split due to error")
            return X_train, X_test, y_train, y_test

    def find_optimal_anomaly_ratio(self, features, labels, test_size=0.2, ratios=None):
        """Find optimal anomaly ratio with simplified approach for small datasets"""
        if ratios is None:
            ratios = [0.1, 0.15, 0.2, 0.25, 0.3]

        print(f"Finding optimal anomaly ratio among {ratios}")

        try:
            # For very small datasets, use a simple approach
            if len(features) < 20 or np.sum(labels == 1) < 5:
                print("Dataset too small for ratio optimization, using default ratio of 0.2")
                return 0.2, pd.DataFrame({'ratio': [0.2], 'score': [0.5]})

            best_ratio = None
            best_score = 0
            results = {}

            for ratio in ratios:
                print(f"\nTesting anomaly ratio: {ratio}")

                try:
                    # Split data with current ratio
                    X_train, X_test, y_train, y_test = self.custom_train_test_split(
                        features, labels, test_size=test_size, anomaly_ratio=ratio
                    )

                    # If we don't have enough samples, skip this ratio
                    if np.sum(y_train == 1) < 2 or np.sum(y_test == 1) < 2:
                        print(f"Skipping ratio {ratio} due to insufficient anomalies")
                        continue

                    # Train a simple model for evaluation
                    model = xgb.XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="auc",
                        use_label_encoder=False,
                        verbosity=0,
                        n_estimators=50,  # Use fewer estimators for speed
                        max_depth=3  # Simple model to avoid overfitting
                    )

                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:, 1]

                    # Calculate metrics
                    try:
                        roc = roc_auc_score(y_test, y_pred)
                    except Exception:
                        roc = 0.5  # Default for failed calculation

                    try:
                        precision, recall, _ = precision_recall_curve(y_test, y_pred)
                        pr_auc = auc(recall, precision)
                    except Exception:
                        pr_auc = 0.5  # Default for failed calculation

                    try:
                        y_pred_class = (y_pred > 0.5).astype(int)
                        report = classification_report(y_test, y_pred_class, output_dict=True)
                        class_1_metrics = report.get("1", {"precision": 0, "recall": 0, "f1-score": 0})
                    except Exception:
                        class_1_metrics = {"precision": 0, "recall": 0, "f1-score": 0}

                    # Store results
                    results[ratio] = {
                        "roc_auc": roc,
                        "pr_auc": pr_auc,
                        "class_1_precision": class_1_metrics["precision"],
                        "class_1_recall": class_1_metrics["recall"],
                        "class_1_f1": class_1_metrics["f1-score"],
                        "class_balance": float(np.mean(y_train)),
                    }

                    print(f"  ROC-AUC: {roc:.4f}, PR-AUC: {pr_auc:.4f}")
                    print(
                        f"  Anomaly precision: {class_1_metrics['precision']:.4f}, recall: {class_1_metrics['recall']:.4f}")

                    # Score that emphasizes PR-AUC for imbalanced data
                    combined_score = 0.4 * roc + 0.6 * pr_auc

                    if combined_score > best_score:
                        best_score = combined_score
                        best_ratio = ratio

                except Exception as e:
                    print(f"Error evaluating ratio {ratio}: {str(e)}")

            # If no valid ratio found, use default
            if best_ratio is None:
                print("No valid ratio found, using default of 0.2")
                best_ratio = 0.2
            else:
                print(f"\nBest anomaly ratio: {best_ratio} (Score: {best_score:.4f})")

            # Create results DataFrame
            results_df = pd.DataFrame.from_dict(results, orient="index")
            return best_ratio, results_df

        except Exception as e:
            print(f"Error in anomaly ratio optimization: {str(e)}")
            # Return a default ratio
            return 0.2, pd.DataFrame({'ratio': [0.2], 'score': [0.5]})

class GAOptimizer:
    """Class for optimizing model hyperparameters with genetic algorithm"""

    def __init__(self, config, X, y):
        self.config = config
        self.X = X
        self.y = y
        self.population_size = config["ga_params"]["population_size"]
        self.generations = config["ga_params"]["generations"]
        self.mutation_rate = config["ga_params"]["mutation_rate"]
        self.crossover_rate = config["ga_params"]["crossover_rate"]
        self.int_params = [
            "max_depth",
            "min_child_weight",
        ]  # Add any other integer parameters here

    def initialize_population(self):
        """Initialize a random population of parameters"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, bounds in self.config["model_params"].items():
                if param in self.int_params:
                    individual[param] = np.random.randint(bounds["min"], bounds["max"])
                else:
                    individual[param] = np.random.uniform(bounds["min"], bounds["max"])
            population.append(individual)
        return population

    def fitness(self, individual):
        """Calculate fitness using cross-validation with better error handling"""
        try:
            params = {
                **individual,
                "objective": self.config["objective"],
                "eval_metric": self.config["eval_metric"],
                "verbosity": 0,  # Suppress XGBoost messages
            }

            # Use XGBoost's native CV instead of scikit-learn's cross_val_score
            dtrain = xgb.DMatrix(self.X, label=self.y)

            # Use simpler evaluation for very small datasets
            if len(self.X) < 20 or len(self.y) < 20:
                # Use a simplified scoring for very small datasets
                n_normal = sum(self.y == 0)
                n_anomaly = sum(self.y == 1)

                if n_anomaly < 2:
                    # Too few anomalies for meaningful CV
                    return 0.5  # Default score

                # Simple train-test split instead of CV
                X_train, X_val, y_train, y_val = train_test_split(
                    self.X, self.y, test_size=0.3, random_state=42
                )

                # Use XGBoost directly rather than the sklearn wrapper
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)

                # Train the model
                bst = xgb.train(params, dtrain, num_boost_round=100)

                # Make predictions
                y_pred = bst.predict(dval)

                try:
                    score = roc_auc_score(y_val, y_pred)
                    return score
                except:
                    return 0.5  # Default score if calculation fails
            else:
                # For larger datasets, use XGBoost's built-in cross-validation
                try:
                    cv_results = xgb.cv(
                        params,
                        dtrain,
                        num_boost_round=100,
                        nfold=3,
                        metrics=[self.config["eval_metric"]],
                        early_stopping_rounds=20,
                        seed=42
                    )

                    # Get the best score (last value in the history)
                    metric_name = f'test-{self.config["eval_metric"]}-mean'
                    best_score = cv_results[metric_name].iloc[-1]
                    return best_score
                except Exception as inner_e:
                    print(f"Error in cross-validation: {inner_e}")
                    return 0.5  # Default score

        except Exception as e:
            print(f"Error calculating fitness: {e}")
            return 0.5  # Default score on error


    def select_parents(self, population, fitnesses):
        """Select parents for reproduction based on fitness"""
        total_fitness = sum(max(0, f) for f in fitnesses)
        if total_fitness <= 0:
            # If total fitness is zero or negative, select randomly
            parents = np.random.choice(range(len(population)), size=2, replace=False)
            return [population[i] for i in parents]

        selection_probs = [max(0, f) / total_fitness for f in fitnesses]
        parents = np.random.choice(
            range(len(population)), size=2, replace=False, p=selection_probs
        )
        return [population[i] for i in parents]

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        if np.random.random() < self.crossover_rate:
            crossover_point = np.random.randint(1, len(parent1))
            param_keys = list(parent1.keys())
            child = {}
            for i, key in enumerate(param_keys):
                if i < crossover_point:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]
        else:
            child = parent1.copy()
        return child

    def mutate(self, individual):
        """Mutate an individual's parameters"""
        for param, value in individual.items():
            if np.random.random() < self.mutation_rate:
                bounds = self.config["model_params"][param]
                if param in self.int_params:
                    individual[param] = np.random.randint(bounds["min"], bounds["max"])
                else:
                    individual[param] = np.random.uniform(bounds["min"], bounds["max"])
        return individual

    def optimize(self):
        """Run the genetic algorithm optimization"""
        print(
            f"Starting GA optimization with {self.population_size} individuals over {self.generations} generations"
        )
        print(
            f"Mutation rate: {self.mutation_rate}, Crossover rate: {self.crossover_rate}"
        )

        population = self.initialize_population()
        best_individual = None
        best_fitness = float("-inf")
        fitness_history = []
        generation_best = []

        for generation in range(self.generations):
            fitnesses = []
            for ind in population:
                try:
                    fitness = self.fitness(ind)
                    fitnesses.append(fitness)
                except Exception as e:
                    print(f"Error calculating fitness: {e}")
                    fitnesses.append(float("-inf"))

            new_best_index = np.argmax(fitnesses)
            generation_best.append(fitnesses[new_best_index])

            if fitnesses[new_best_index] > best_fitness:
                best_individual = population[new_best_index].copy()
                best_fitness = fitnesses[new_best_index]

            fitness_history.append(best_fitness)
            print(
                f"Generation {generation + 1}/{self.generations}, Best fitness: {best_fitness:.4f}"
            )

            new_population = []
            # Elitism - keep the best individual
            if best_individual:
                new_population.append(best_individual.copy())

            while len(new_population) < self.population_size:
                try:
                    parents = self.select_parents(population, fitnesses)
                    child = self.crossover(parents[0], parents[1])
                    child = self.mutate(child)
                    new_population.append(child)
                except Exception as e:
                    print(f"Error in reproduction: {e}")
                    # Add a random individual as fallback
                    individual = {}
                    for param, bounds in self.config["model_params"].items():
                        if param in self.int_params:
                            individual[param] = np.random.randint(
                                bounds["min"], bounds["max"]
                            )
                        else:
                            individual[param] = np.random.uniform(
                                bounds["min"], bounds["max"]
                            )
                    new_population.append(individual)

            population = new_population

        # Ensure integer parameters are integers
        if best_individual:
            for param in self.int_params:
                if param in best_individual:
                    best_individual[param] = int(best_individual[param])

            print("\nOptimization complete!")
            print("Best parameters found:")
            for param, value in best_individual.items():
                print(f"  {param}: {value}")
            print(f"Best fitness: {best_fitness:.4f}")

        return best_individual, best_fitness, fitness_history

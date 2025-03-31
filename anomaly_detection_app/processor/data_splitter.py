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
    """Class for splitting data with custom anomaly ratios"""

    def __init__(self, config):
        self.config = config

    def custom_train_test_split(
        self, features, labels, test_size=0.2, anomaly_ratio=0.01
    ):
        """Split data with custom anomaly ratio in training set"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, stratify=labels, random_state=42
        )

        # Count the number of samples in each class
        n_normal = np.sum(y_train == 0)
        n_anomaly = np.sum(y_train == 1)

        # Calculate the target number of samples for each class
        n_train = len(X_train)
        n_train_anomalies = min(int(n_train * anomaly_ratio), n_anomaly)
        n_train_normal = min(n_train - n_train_anomalies, n_normal)

        # Adjust anomaly count if necessary
        if n_train_normal + n_train_anomalies < n_train:
            n_train_anomalies = n_train - n_train_normal

        # Undersample both classes
        undersampler = RandomUnderSampler(
            sampling_strategy={0: n_train_normal, 1: n_train_anomalies}, random_state=42
        )
        X_train_resampled, y_train_resampled = undersampler.fit_resample(
            X_train, y_train
        )

        # Print split information
        print(
            f"Original data: {len(labels)} samples, {np.sum(labels == 1)} anomalies ({np.mean(labels) * 100:.2f}%)"
        )
        print(
            f"Training data: {len(y_train_resampled)} samples, {np.sum(y_train_resampled == 1)} anomalies ({np.mean(y_train_resampled) * 100:.2f}%)"
        )
        print(
            f"Test data: {len(y_test)} samples, {np.sum(y_test == 1)} anomalies ({np.mean(y_test) * 100:.2f}%)"
        )

        return X_train_resampled, X_test, y_train_resampled, y_test

    def find_optimal_anomaly_ratio(
        self, features, labels, test_size=0.2, ratios=[0.01, 0.05, 0.1, 0.15, 0.2]
    ):
        """Find optimal anomaly ratio for training"""
        best_ratio = None
        best_score = 0
        results = {}

        print(f"Finding optimal anomaly ratio among {ratios}...")

        for ratio in ratios:
            print(f"\nTesting anomaly ratio: {ratio}")
            X_train, X_test, y_train, y_test = self.custom_train_test_split(
                features, labels, test_size=test_size, anomaly_ratio=ratio
            )

            # Train a simple model to evaluate the ratio
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                verbosity=0,
            )

            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_pred)

            # Calculate precision-recall AUC which is better for imbalanced data
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            pr_auc = auc(recall, precision)

            # Get classification metrics
            y_pred_class = (y_pred > 0.5).astype(int)
            report = classification_report(y_test, y_pred_class, output_dict=True)

            results[ratio] = {
                "roc_auc": roc,
                "pr_auc": pr_auc,
                "class_1_precision": report["1"]["precision"],
                "class_1_recall": report["1"]["recall"],
                "class_1_f1": report["1"]["f1-score"],
                "class_balance": f"{np.mean(y_train):.3f}",
            }

            print(f"  ROC-AUC: {roc:.4f}, PR-AUC: {pr_auc:.4f}")
            print(
                f"  Anomaly precision: {report['1']['precision']:.4f}, recall: {report['1']['recall']:.4f}"
            )

            # Consider both metrics with emphasis on PR-AUC for imbalanced data
            combined_score = 0.4 * roc + 0.6 * pr_auc

            if combined_score > best_score:
                best_score = combined_score
                best_ratio = ratio

        print(f"\nBest anomaly ratio: {best_ratio} (Combined score: {best_score:.4f})")

        # Create a DataFrame for easier comparison
        results_df = pd.DataFrame.from_dict(results, orient="index")
        print("\nResults summary:")
        print(results_df)

        return best_ratio, results_df


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
        """Calculate fitness using cross-validation"""
        try:
            params = {
                **individual,
                "objective": self.config["objective"],
                "eval_metric": self.config["eval_metric"],
                "verbosity": 0,  # Suppress XGBoost messages
            }
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, self.X, self.y, cv=5, scoring="roc_auc")
            return np.mean(scores)
        except Exception as e:
            print(f"Error calculating fitness: {e}")
            return float("-inf")

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

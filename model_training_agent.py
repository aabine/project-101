import numpy as np
from typing import Any, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import autosklearn.classification
import shap
import logging
from micro_agent import MicroAgent
from prometheus_client import Counter, Gauge

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prometheus metrics
TASK_COUNTER = Counter('task_counter', 'Number of tasks performed', ['agent'])
AGENT_PERFORMANCE = Gauge('agent_performance', 'Performance metric of the agent', ['agent'])

class ModelTrainingAgent(MicroAgent):
    """
    Agent responsible for training machine learning models using AutoML.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the ModelTrainingAgent.

        Args:
            name (str): The name of the agent.
            config (Dict[str, Any]): Configuration dictionary for the agent.
        """
        super().__init__(name, config)
        self.automl_time_limit = config.get('automl_time_limit', 300)
        self.automl_per_run_limit = config.get('automl_per_run_limit', 30)
        self.automl_memory_limit = config.get('automl_memory_limit', 8192)

    def perform_task(self, input_data: Dict[str, Any]) -> Tuple[Any, float, float, float, np.ndarray]:
        """
        Train a model using AutoML and provide explainability.
        
        Args:
            input_data (Dict[str, Any]): The input data for model training.
                Expected keys: 'data', 'target'
        
        Returns:
            Tuple: The trained model, accuracy, F1 score, ROC AUC score, and feature importances.
        
        Raises:
            ValueError: If the input data is invalid.
        """
        logger.info(f"[{self.name}] Training a model with AutoML...")
        try:
            self._validate_input(input_data)
            X_train, X_test, y_train, y_test = self._prepare_data(input_data)
            
            automl = self._train_model(X_train, y_train)
            predictions, probas = self._make_predictions(automl, X_test)
            
            metrics = self._calculate_metrics(y_test, predictions, probas)
            feature_importances = self._explain_model(automl, X_train, X_test)
            
            self._log_results(metrics)
            self._update_metrics(metrics['accuracy'])
            
            return (automl, metrics['accuracy'], metrics['f1_score'], 
                    metrics['roc_auc'], feature_importances)
        except Exception as e:
            logger.error(f"[{self.name}] Error in model training: {str(e)}")
            raise

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate the input data."""
        if not isinstance(input_data, dict) or 'data' not in input_data or 'target' not in input_data:
            raise ValueError("Invalid input data format. Expected 'data' and 'target' keys.")

    def _prepare_data(self, input_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the data for model training."""
        return train_test_split(input_data['data'], input_data['target'], test_size=0.2, random_state=42)

    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> autosklearn.classification.AutoSklearnClassifier:
        """Train the AutoML model."""
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=self.automl_time_limit,
            per_run_time_limit=self.automl_per_run_limit,
            metric=autosklearn.metrics.f1,
            memory_limit=self.automl_memory_limit
        )
        automl.fit(X_train, y_train)
        return automl

    def _make_predictions(self, model: autosklearn.classification.AutoSklearnClassifier, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained model."""
        return model.predict(X_test), model.predict_proba(X_test)

    def _calculate_metrics(self, y_test: np.ndarray, predictions: np.ndarray, probas: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'roc_auc': roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
        }

    def _explain_model(self, model: autosklearn.classification.AutoSklearnClassifier, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """Explain the model using SHAP values."""
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
        shap_values = explainer.shap_values(X_test)
        return np.abs(shap_values).mean(axis=0).mean(axis=0)

    def _log_results(self, metrics: Dict[str, float]) -> None:
        """Log the results of the model training."""
        logger.info(f"[{self.name}] Model Accuracy: {metrics['accuracy']:.4f}, "
                    f"F1 Score: {metrics['f1_score']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")

    def _update_metrics(self, accuracy: float) -> None:
        """Update Prometheus metrics."""
        TASK_COUNTER.labels(agent=self.name).inc()
        AGENT_PERFORMANCE.labels(agent=self.name).set(accuracy)

# Unit tests
from main import DataPreprocessingAgent, ModelTrainingAgent
from main import CoreAgent
import unittest
import autosklearn.classification
import numpy as np
from sklearn import datasets
from prometheus_client import start_http_server
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class TestMicroAgents(unittest.TestCase):
    def setUp(self):
        self.config = {
            'performance_threshold': 0.8,
            'accuracy_weight': 0.7,
            'f1_score_weight': 0.3,
            'rl_timesteps': 100,
            'ga_population_size': 10,
            'ga_individual_size': 5,
            'ga_crossover_prob': 0.7,
            'ga_mutation_prob': 0.2,
            'ga_generations': 5,
            'docker_memory_limit': '500m',
            'max_workers': 4
        }
        self.core_agent = CoreAgent(self.config)

    def test_data_preprocessing_agent(self):
        agent = DataPreprocessingAgent("TestPreprocessor")
        data = np.random.rand(100, 5)
        result = agent.perform_task(data)
        self.assertEqual(result.shape, data.shape)
        self.assertAlmostEqual(result.mean(), 0, places=7)
        self.assertAlmostEqual(result.std(), 1, places=7)

    def test_model_training_agent(self):
        agent = ModelTrainingAgent("TestTrainer")
        iris = datasets.load_iris()
        result = agent.perform_task({'data': iris.data, 'target': iris.target})
        self.assertIsInstance(result[0], autosklearn.classification.AutoSklearnClassifier)
        self.assertGreater(result[1], 0)
        self.assertGreater(result[2], 0)

    def test_core_agent(self):
        self.core_agent.add_micro_agent("TestPreprocessor", DataPreprocessingAgent("TestPreprocessor"))
        self.core_agent.add_micro_agent("TestTrainer", ModelTrainingAgent("TestTrainer"))
        self.assertEqual(len(self.core_agent.micro_agents), 2)
        
        iris = datasets.load_iris()
        result = self.core_agent.perform_task_with_agent("TestPreprocessor", iris.data)
        self.assertIsNotNone(result)
        
        result = self.core_agent.perform_task_with_agent("TestTrainer", {'data': iris.data, 'target': iris.target})
        self.assertIsNotNone(result)

    def test_error_handling(self):
        with self.assertLogs(level='ERROR') as cm:
            self.core_agent.perform_task_with_agent("NonExistentAgent", None)
        self.assertIn("Agent 'NonExistentAgent' not found.", cm.output[0])

if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Configuration
    config = {
        'performance_threshold': 0.8,
        'accuracy_weight': 0.7,
        'f1_score_weight': 0.3,
        'rl_timesteps': 1000,
        'ga_population_size': 50,
        'ga_individual_size': 10,
        'ga_crossover_prob': 0.7,
        'ga_mutation_prob': 0.2,
        'ga_generations': 20,
        'docker_memory_limit': '1g',
        'max_workers': 4
    }

    # Start Prometheus metrics server
    start_http_server(8000)

    # Initialize CoreAgent
    core_agent = CoreAgent(config)

    # Add Micro-Agents
    core_agent.add_micro_agent("DataPreprocessor", DataPreprocessingAgent("DataPreprocessor"))
    core_agent.add_micro_agent("ModelTrainer", ModelTrainingAgent("ModelTrainer"))

    # Evaluate and improve agents
    core_agent.evaluate_and_improve()

    # Run a sample task
    iris_data = datasets.load_iris()
    preprocessed_data = core_agent.perform_task_with_agent("DataPreprocessor", iris_data.data)
    if preprocessed_data is not None:
        model, accuracy, f1_score = core_agent.perform_task_with_agent("ModelTrainer", {'data': preprocessed_data, 'target': iris_data.target})
        logger.info(f"Final model accuracy: {accuracy}, F1 score: {f1_score}")

    logger.info("AI system execution completed.")
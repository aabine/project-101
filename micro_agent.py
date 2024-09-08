import numpy as np
import pandas as pd
from typing import Any
import logging

# Assuming these are custom modules/classes
from micro_agent import MicroAgent

# Prometheus metrics (assuming they are defined elsewhere)
from prometheus_client import Counter

# Set up logging
logger = logging.getLogger(__name__)

# Prometheus metrics
TASK_COUNTER = Counter('task_counter', 'Number of tasks performed', ['agent'])

class MicroAgent:
    """
    Base class for all Micro-Agents in the system.
    
    Attributes:
        name (str): The name of the agent.
    """
    
    def __init__(self, name: str):
        """
        Initialize a new MicroAgent.
        
        Args:
            name (str): The name of the agent.
        """
        self.name = name

    def perform_task(self, input_data: Any) -> Any:
        """
        Perform the agent's specific task.
        
        Args:
            input_data (Any): The input data for the task.
        
        Returns:
            Any: The result of the task.
        
        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class DataPreprocessingAgent(MicroAgent):
    """
    Agent responsible for data preprocessing tasks.
    """
    
    def perform_task(self, input_data: Any) -> Any:
        """
        Preprocess the input data by normalizing it.
        
        Args:
            input_data (Any): The input data to preprocess.
        
        Returns:
            Any: The preprocessed data.
        
        Raises:
            ValueError: If the input data is not valid.
        """
        logger.info(f"[{self.name}] Performing data preprocessing...")
        try:
            self._validate_input(input_data)
            normalized_data = (input_data - input_data.mean(axis=0)) / input_data.std(axis=0)
            TASK_COUNTER.labels(agent=self.name).inc()
            return normalized_data
        except Exception as e:
            logger.error(f"[{self.name}] Error in data preprocessing: {str(e)}")
            raise

    def _validate_input(self, input_data: Any):
        """
        Validate the input data.
        
        Args:
            input_data (Any): The input data to validate.
        
        Raises:
            ValueError: If the input data is not valid.
        """
        if not isinstance(input_data, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input data must be a numpy array or pandas DataFrame")
        if input_data.size == 0:
            raise ValueError("Input data cannot be empty")
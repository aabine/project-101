from asyncio import as_completed
from asyncio.log import logger
from datetime import time
import numpy as np
import random
from deap import algorithms, base, creator, tools
from sklearn import datasets, autosklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from typing import Any, Dict, List

from CoreAgentBase import CoreAgentBase
from micro_agent import MicroAgent
from model_training_agent import ModelTrainingAgent

class AgentEvolutionManager(CoreAgentBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def setup_genetic_algorithm(self):
        toolbox = base.Toolbox()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, self.config['ga_individual_size'])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20.0, indpb=1.0/self.config['ga_individual_size'])
        toolbox.register("select", tools.selNSGA2)
        return toolbox

    def evaluate_individual(self, individual):
        temp_agent = self.generate_temp_agent(individual)
        iris = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
        try:
            start_time = time.time()
            result = temp_agent.perform_task({'data': X_train, 'target': y_train})
            end_time = time.time()
            accuracy = accuracy_score(y_test, result[0].predict(X_test))
            f1 = f1_score(y_test, result[0].predict(X_test), average='weighted')
            execution_time = end_time - start_time
            normalized_time = 1 - (execution_time / self.config['max_execution_time'])
            return accuracy, f1, normalized_time
        except Exception as e:
            logger.error(f"Error evaluating individual: {str(e)}")
            return 0, 0, 0

    def generate_temp_agent(self, individual):
        class TempAgent(MicroAgent):
            def __init__(self, name, genes):
                super().__init__(name)
                self.genes = genes

            def perform_task(self, input_data):
                X, y = input_data['data'], input_data['target']
                model = autosklearn.classification.AutoSklearnClassifier(
                    time_left_for_this_task=int(self.genes[0] * 100),
                    per_run_time_limit=int(self.genes[1] * 30),
                    ensemble_size=int(self.genes[2] * 50),
                    initial_configurations_via_metalearning=int(self.genes[3] * 25)
                )
                model.fit(X, y)
                return model, model.score(X, y)
        return TempAgent("TempAgent", individual)

    def evaluate_and_improve(self):
        try:
            futures = []
            for name, agent in self.micro_agents.items():
                if isinstance(agent, ModelTrainingAgent):
                    future = self.executor.submit(self._evaluate_agent, name, agent)
                    futures.append(future)
            for future in as_completed(futures):
                name, perf = future.result()
                if name not in self.history:
                    self.history[name] = []
                self.history[name].append(perf)
                if perf < self.config['performance_threshold']:
                    logger.warning(f"Agent '{name}' is underperforming. Generating a new agent...")
                    self.generate_new_agent(name)
            self.adjust_config()
        except Exception as e:
            logger.error(f"Error in evaluate_and_improve: {str(e)}")
            self._handle_error(e)

    def _evaluate_agent(self, name: str, agent: ModelTrainingAgent):
        _, accuracy, f1_score_value, _, _ = agent.perform_task(datasets.load_iris())
        perf = self.compute_reward(accuracy, f1_score_value, 0, 0)
        self.performance[name] = perf
        self.rl_model.learn(total_timesteps=self.config['rl_timesteps'])
        return name, perf

    def generate_new_agent(self, base_agent_name: str):
        if base_agent_name not in self.micro_agents:
            logger.error(f"Base agent '{base_agent_name}' not found.")
            return
        try:
            toolbox = self.setup_genetic_algorithm()
            population = toolbox.population(n=self.config['ga_population_size'])
            population, logbook = algorithms.eaSimple(population, toolbox, 
                                cxpb=self.config['ga_crossover_prob'], 
                                mutpb=self.config['ga_mutation_prob'], 
                                ngen=self.config['ga_generations'], 
                                stats=None, halloffame=None)
            new_agent_code = self.generate_agent_code(base_agent_name, population[0])
            self.execute_code_in_docker(new_agent_code)
        except Exception as e:
            logger.error(f"Error in generate_new_agent: {str(e)}")
            self._handle_error(e)

    def generate_agent_code(self, base_agent_name: str, individual: list) -> str:
        return f"""
class {base_agent_name}V2(MicroAgent):
    def perform_task(self, input_data: Any) -> Any:
        logger.info(f"[{{self.name}}] Enhanced agent is processing data...")
        processed_data = input_data * {individual[0]} + {individual[1]}
        return processed_data"""

    def adjust_config(self):
        for agent_name, perf_history in self.history.items():
            if len(perf_history) > self.config['history_window']:
                recent_perf = perf_history[-self.config['history_window']:]
                perf_trend = np.polyfit(range(len(recent_perf)), recent_perf, 1)[0]
                if perf_trend < 0:  # Performance is decreasing
                    self.config['rl_timesteps'] = min(self.config['rl_timesteps'] * 1.5, self.config['max_rl_timesteps'])
                    self.config['ga_generations'] = min(self.config['ga_generations'] * 1.5, self.config['max_ga_generations'])
                else:  # Performance is stable or increasing
                    self.config['rl_timesteps'] = max(self.config['rl_timesteps'] * 0.9, self.config['min_rl_timesteps'])
                    self.config['ga_generations'] = max(self.config['ga_generations'] * 0.9, self.config['min_ga_generations'])
        logger.info(f"Adjusted configuration: RL timesteps = {self.config['rl_timesteps']}, GA generations = {self.config['ga_generations']}")
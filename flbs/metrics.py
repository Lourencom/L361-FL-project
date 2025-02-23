from abc import ABC, abstractmethod
from typing import Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils import relative_to_absolute_path

class Metrics(ABC):
    """
    This class represents the metrics used to asses the quality of the estimate for 
    some variation of the critical batch size defined for centralized training.
    The original paper that proposed crticial batch size:
    https://arxiv.org/abs/1812.06162
    """

    did_calculate: bool = False

    def calculate(self) -> None:
        """
        Calculate metrics
        """
        self.best_accs = [max(acc) for acc in self.accuracies]
        best_acc_idx = np.argmax(self.best_accs)
        self.critical_batch_size = self.batch_sizes[best_acc_idx]

        # metric is like a relative error
        self.metric_val = abs(self.critical_batch_size - self.estimate) / self.critical_batch_size
        self.did_calculate = True


    def plot_metrics(self, save_dir: str = None) -> None:
        """ 
        Plot metrics
        """
        if not self.did_calculate:
            self.calculate()
        save_dir = relative_to_absolute_path(save_dir)
        self._plot(save_dir)
    
    @abstractmethod
    def _plot(self, save_dir: str = None) -> None:
        pass
    
    def _save_plot(self, save_dir: str, fig: plt.Figure) -> None:
        plot_name = f'{self.metric_name}_estimate_{self.estimate:.2f}_target_{self.critical_batch_size:.2f}.png'
        plot_path = os.path.join(save_dir, plot_name)
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        fig.savefig(plot_path, dpi=300)

    def reset(self) -> None:
        self.did_calculate = False

    
class CentralizedCriticalBatchSizeMetric(Metrics):
    def __init__(self, batch_sizes: List[int], accuracies: List[List[float]], batch_size_estimate: float):
        """
        Args:
            batch_sizes: List of batch sizes used in experiments
            accuracies: List of accuracies for each batch size
            batch_size_estimate: Batch size estimate
        """
        self.batch_sizes = batch_sizes
        self.accuracies = accuracies
        self.estimate = batch_size_estimate

        self.did_calculate = False
        self.metric_name = "centralized_critical_batch_size"

    def _plot(self, save_dir: str = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(self.batch_sizes, self.best_accs, label='Accuracy')
        ax.plot(self.batch_sizes, self.best_accs, linestyle='--', color='black')
        ax.axvline(x=self.estimate, color='red', linestyle='--', label='Estimated Critical Batch Size')
        ax.axvline(x=self.critical_batch_size, color='green', linestyle='--', label='Target Critical Batch Size')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Accuracy')
        
        plt.grid(True)
        ax.set_title(f'Critical BS estimate: {self.estimate}, Critical BS: {self.critical_batch_size}, Relative Error: {self.metric_val}')
        ax.legend()
        plt.show()

        if save_dir:
            self._save_plot(save_dir, fig)
    

class FLCriticalLocalBatchSizeMetric(Metrics):
    def __init__(self, batch_sizes: List[int], accuracies: List[List[float]], local_bs_estimate: float):
        self.batch_sizes = batch_sizes
        self.accuracies = accuracies
        self.estimate = local_bs_estimate

        self.did_calculate = False
        self.metric_name = "federated_critical_local_batch_size"

    def _plot(self, save_dir: str = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(self.batch_sizes, self.best_accs, label='Accuracy')
        ax.plot(self.batch_sizes, self.best_accs, linestyle='--', color='black')
        ax.axvline(x=self.estimate, color='red', linestyle='--', label='Estimated Critical Local BS')
        ax.axvline(x=self.critical_batch_size, color='green', linestyle='--', label='Target Critical BS')
        ax.set_xlabel('Local Batch Size')
        ax.set_ylabel('Accuracy')
        plt.grid(True)
        ax.set_title(f'Local BS estimate: {self.estimate}, Critical BS: {self.critical_batch_size}, Relative Error: {self.metric_val}')
        ax.legend()
        plt.show()

        if save_dir:
            self._save_plot(save_dir, fig)


class FLCriticalCohortSizeMetric(Metrics):
    def __init__(self, batch_sizes: List[int], accuracies: List[List[float]], cohort_size_estimate: float):
        self.batch_sizes = batch_sizes
        self.accuracies = accuracies
        self.estimate = cohort_size_estimate

        self.did_calculate = False
        self.metric_name = "federated_critical_cohort_size"
        
    def _plot(self, save_dir: str = None) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(self.batch_sizes, self.best_accs, label='Accuracy')
        ax.plot(self.batch_sizes, self.best_accs, linestyle='--', color='black')
        ax.axvline(x=self.estimate, color='red', linestyle='--', label='Estimated Critical Cohort Size')
        ax.axvline(x=self.critical_batch_size, color='green', linestyle='--', label='Target Critical Cohort Size')
        ax.set_xlabel('Cohort Size')
        ax.set_ylabel('Accuracy')
        plt.grid(True)
        ax.set_title(f'Cohort Size estimate: {self.estimate}, Critical Cohort Size: {self.critical_batch_size}, Relative Error: {self.metric_val}')
        ax.legend()
        plt.show()

        if save_dir:
            self._save_plot(save_dir, fig)


if __name__ == "__main__":
    batch_sizes = [10, 20, 30, 40, 50]
    accuracies = [[0.5, 0.6, 0.7], [0.6, 0.7, 0.8], [0.7, 0.8, 0.9], [0.8, 0.91, 0.89], [0.7, 0.8, 0.88]]
    batch_size_estimate = 35
    metrics = FLCriticalCohortSizeMetric(batch_sizes, accuracies, batch_size_estimate)
    metrics.plot_metrics(save_dir="plots")

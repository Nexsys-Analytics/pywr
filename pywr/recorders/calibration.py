"""
This module contains `Recorder` subclasses useful in the evaluation of model
 performance and calibration.

Several different evaluation metrics are implemented in this module. Many are
 well known to modellers. However an important reference for this implementation
 is `[1]_` which reviews the available metrics for watershed models.

..  [1] Moriasi, D.N., et al. (2007) Model Evaluation Guidelines for Systematic
    Quantification of Accuracy in Watershed Simulations. Transactions of the ASABE, 50, 885-900.
    http://dx.doi.org/10.13031/2013.23153


"""

from ._recorders import NumpyArrayNodeRecorder
import numpy as np
from pywr.dataframe_tools import load_dataframe
from pywr.dataframe_tools import align_and_resample_dataframe
import pandas as pd



class AbstractComparisonNodeRecorder(NumpyArrayNodeRecorder): 
    """Base class for all Recorders performing timeseries comparison of `Node` flows.""" 

    def __init__(self, model, node, observed, **kwargs): 
        super(AbstractComparisonNodeRecorder, self).__init__(model, node, **kwargs) 
        self.observed = load_dataframe(model, observed) 
        self._aligned_observed = None 

    def setup(self): 
        super(AbstractComparisonNodeRecorder, self).setup() 
        # Align the observed data to the model's timestep index
        self._aligned_observed = align_and_resample_dataframe( 
            self.observed, self.model.timestepper.datetime_index 
        ) 

AbstractComparisonNodeRecorder.register() 


class RootMeanSquaredErrorNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the RMSE between model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        return np.sqrt(np.mean((obs - mod) ** 2, axis=0))


class MeanAbsoluteErrorNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the MAE between model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        return np.mean(np.abs(obs - mod), axis=0)


class MeanSquareErrorNodeRecorder(AbstractComparisonNodeRecorder): 
    """Recorder calculates the mean squared error (MSE) between model and observed across scenarios.""" 

    def values(self): 
        # Simulated data from NumpyArrayNodeRecorder has shape: (time, scenarios)
        mod = self.data 
        
        # Ensure observed data is a 2D numpy array of shape (time, 1) for broadcasting
        if isinstance(self._aligned_observed, pd.DataFrame):
            obs = self._aligned_observed.iloc[:, 0].values.reshape(-1, 1)
        else:
            obs = self._aligned_observed.values.reshape(-1, 1)

        # Calculate MSE over the time axis (axis=0) for each scenario
        # Using np.nanmean safely ignores missing observed records (NaNs)
        mse = np.nanmean((mod - obs) ** 2, axis=0) 
        
        return mse 

    def to_dataframe(self):
        """Returns the MSE for each scenario as a pandas DataFrame."""
        mse_values = self.values()
        
        # Check if the model uses multiple scenarios
        if self.model.scenarios.combinations:
            # Create a MultiIndex from the scenario combinations
            index = pd.MultiIndex.from_tuples(
                [c.labels for c in self.model.scenarios.combinations],
                names=[s.name for s in self.model.scenarios.scenarios]
            )
        else:
            # Fallback for baseline runs without explicitly defined scenarios
            index = ["Baseline"]
            
        # Return a DataFrame with scenarios as the index and the node name as the column
        return pd.DataFrame({self.node.name: mse_values}, index=index)

MeanSquareErrorNodeRecorder.register()

class PercentBiasNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the percent bias between model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        return np.sum(obs - mod, axis=0) * 100 / np.sum(obs, axis=0)


class RMSEStandardDeviationRatioNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the RMSE-observations standard deviation ratio between model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        return np.sqrt(np.mean((obs - mod) ** 2, axis=0)) / np.std(obs, axis=0)


class NashSutcliffeEfficiencyNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the Nash-Sutcliffe efficiency model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        obs_mean = np.mean(obs, axis=0)
        mod_flat = mod.flatten()
        return 1.0 - np.sum((obs - mod_flat) ** 2, axis=0) / np.sum(
            (obs - obs_mean) ** 2, axis=0
        )
NashSutcliffeEfficiencyNodeRecorder.register()
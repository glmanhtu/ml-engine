from typing import Any, Dict, Union, Optional, TYPE_CHECKING

import mlflow
from mlflow import ActiveRun
from ml_engine.tracking.tracker import Tracker

import PIL.Image
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.figure
    import PIL
    import plotly


class MLFlowTracker(Tracker):
    def __init__(self,
                 name: str,
                 tracking_uri: str,
                 artifact_location: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None):

        mlflow.set_tracking_uri(tracking_uri)
        exp = mlflow.get_experiment_by_name(name)
        if not exp:
            mlflow.create_experiment(name, artifact_location, tags)
        exp = mlflow.set_experiment(name)
        self.exp_id = exp.experiment_id
        self.tracking_uri = tracking_uri
        self.run: Union[ActiveRun, None] = None
        self.client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    def start_tracking(self, run_id: Optional[str] = None,
                       run_name: Optional[str] = None, nested: bool = False, tags: Optional[Dict[str, Any]] = None,
                       description: Optional[str] = None, log_system_metrics: Optional[bool] = None):
        self.run = mlflow.start_run(run_id, self.exp_id, run_name, nested, tags, description, log_system_metrics)
        return self.run

    def stop_tracking(self):
        mlflow.end_run()

    def get_state_dict(self, artifact_path):
        state_dict_uri = mlflow.get_artifact_uri(artifact_path)
        return mlflow.pytorch.load_state_dict(state_dict_uri)

    def log_state_dict(self, state_dict, artifact_path):
        mlflow.pytorch.log_state_dict(state_dict, artifact_path)

    def log_metrics(self, metrics: Dict[str, float], step: int, synchronous: bool = True) -> None:
        mlflow.log_metrics(metrics, step, synchronous)

    def log_metric(self, key, value, step: Optional[int] = None, synchronous: bool = True) -> None:
        mlflow.log_metric(key, value, step, synchronous)

    def log_table(self, data: Union[Dict[str, Any], "pd.DataFrame"], artifact_file: str):
        mlflow.log_table(data, artifact_file)

    def log_image(self, image: Union["np.ndarray", "PIL.Image.Image"], artifact_file: str):
        mlflow.log_image(image, artifact_file)

    def log_figure(self, figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"], artifact_file: str,
                   save_kwargs: Optional[Dict[str, Any]] = None):
        mlflow.log_figure(figure, artifact_file, save_kwargs=save_kwargs)

    def log_params(self, params: Dict[str, Any], synchronous: bool = True):
        mlflow.log_params(params, synchronous)

    def log_param(self, key: str, value: Any, synchronous: bool = True):
        mlflow.log_param(key, value, synchronous)

    def get_param(self, key: str):
        data = self.run.data.to_dictionary()
        return data['params'][key]

    def get_metric(self, key: str):
        return self.client.get_metric_history(self.run.info.run_id, key)

    def log_artifact(self, local_file_path, artifact_path):
        mlflow.log_artifact(local_file_path, artifact_path)

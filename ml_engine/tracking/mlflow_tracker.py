import shutil
import tempfile
from typing import Any, Dict, Union, Optional, TYPE_CHECKING
import os
from urllib.parse import urlparse

import PIL.Image
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow import ActiveRun
import torch.distributed as dist

from ml_engine.tracking.tracker import Tracker
from ml_engine.utils import EmptyContext

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.figure
    import PIL
    import plotly


class MLFlowTracker(Tracker):
    def __init__(self,
                 name: str,
                 tracking_uri: str,
                 rank: Optional[int] = None,
                 artifact_location: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 synchronous=True,
                 enabled: bool = True,
                 save_artifact_to_disk: bool = False,
                 local_artifact_dir: str = ''):

        mlflow.set_tracking_uri(tracking_uri)
        self.rank = rank
        self.name = name
        self.artifact_location = artifact_location
        self.exp_tags = tags
        self.tracking_uri = tracking_uri
        self.run: Union[ActiveRun, None] = None
        self.client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        self.synchronous = synchronous
        self.enabled = enabled
        self.save_artifact_to_disk = save_artifact_to_disk
        self.local_artifact_dir = local_artifact_dir

    def start_tracking(self, run_id: Optional[str] = None,
                       run_name: Optional[str] = None, nested: bool = False, tags: Optional[Dict[str, Any]] = None,
                       description: Optional[str] = None, log_system_metrics: Optional[bool] = None):
        if not self.should_monitor():
            return EmptyContext()

        exp = mlflow.get_experiment_by_name(self.name)
        if not exp:
            mlflow.create_experiment(self.name, self.artifact_location, self.exp_tags)
        exp = mlflow.set_experiment(self.name)
        exp_id = exp.experiment_id
        self.run = mlflow.start_run(run_id, exp_id, run_name, nested, tags, description, log_system_metrics)
        return self.run

    def should_monitor(self):
        if not self.enabled:
            return False
        if self.rank is not None:
            if self.rank != dist.get_rank():
                return False
        return True

    def stop_tracking(self):
        mlflow.end_run()

    def get_state_dict(self, artifact_path):
        if os.path.exists(artifact_path):
            return torch.load(artifact_path, map_location='cpu')
        if self.save_artifact_to_disk:
            model_path = os.path.join(self.local_artifact_dir, artifact_path, 'checkpoint.pth')
            if os.path.exists(model_path):
                return torch.load(model_path)

        parsed_path = urlparse(artifact_path)
        if not parsed_path.scheme:
            artifact_path = mlflow.get_artifact_uri(artifact_path)
        return mlflow.pytorch.load_state_dict(artifact_path)

    def log_state_dict(self, state_dict, artifact_path):
        if not self.should_monitor():
            return
        if self.save_artifact_to_disk:
            model_dir = os.path.join(self.local_artifact_dir, artifact_path)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(model_dir, 'checkpoint.pth'))
        mlflow.pytorch.log_state_dict(state_dict, artifact_path)

    def log_metrics(self, metrics: Dict[str, float], step: int, synchronous: Union[bool, None] = None) -> None:
        if not self.should_monitor():
            return

        if synchronous is None:
            synchronous = self.synchronous
        mlflow.log_metrics(metrics, step, synchronous)

    def log_metric(self, key, value, step: Optional[int] = None, synchronous: Union[bool, None] = None) -> None:
        if not self.should_monitor():
            return

        if synchronous is None:
            synchronous = self.synchronous
        mlflow.log_metric(key, value, step, synchronous)

    def log_table(self, data: Union[Dict[str, Any], "pd.DataFrame"], artifact_file: str):
        if not self.should_monitor():
            return

        mlflow.log_table(data, artifact_file)

    def log_image(self, image: Union["np.ndarray", "PIL.Image.Image"], artifact_file: str):
        if not self.should_monitor():
            return

        mlflow.log_image(image, artifact_file)

    def log_figure(self, figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"], artifact_file: str,
                   save_kwargs: Optional[Dict[str, Any]] = None):
        if not self.should_monitor():
            return

        mlflow.log_figure(figure, artifact_file, save_kwargs=save_kwargs)

    def log_params(self, params: Dict[str, Any], synchronous: Union[bool, None] = None):
        if not self.should_monitor():
            return

        if synchronous is None:
            synchronous = self.synchronous
        mlflow.log_params(params, synchronous)

    def log_param(self, key: str, value: Any, synchronous: Union[bool, None] = None):
        if not self.should_monitor():
            return

        if synchronous is None:
            synchronous = self.synchronous
        mlflow.log_param(key, value, synchronous)

    def get_param(self, key: str):
        data = self.run.data.to_dictionary()
        return data['params'][key]

    def get_metric(self, key: str):
        return self.client.get_metric_history(self.run.info.run_id, key)

    def log_artifact(self, local_file_path, artifact_path):
        if not self.should_monitor():
            return

        if self.save_artifact_to_disk:
            artifact_dir = os.path.join(self.local_artifact_dir, artifact_path)
            os.makedirs(artifact_dir, exist_ok=True)
            shutil.copy2(local_file_path, artifact_dir)

        mlflow.log_artifact(local_file_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        if not self.should_monitor():
            return

        if self.save_artifact_to_disk:
            artifact_dir = os.path.join(self.local_artifact_dir, artifact_path)
            os.makedirs(artifact_dir, exist_ok=True)
            shutil.copytree(local_dir, artifact_dir, dirs_exist_ok=True)

        mlflow.log_artifacts(local_dir, artifact_path)

    def log_model(self, model, artifact_path: str, signature=None):
        if not self.should_monitor():
            return

        mlflow.pytorch.log_model(model, artifact_path, signature=signature)

    def log_table_as_csv(self, data: pd.DataFrame, artifact_path: str, filename: str) -> None:
        if not self.should_monitor():
            return

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, filename)
            data.to_csv(path)
            self.log_artifact(path, artifact_path)



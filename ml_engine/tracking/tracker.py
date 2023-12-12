from typing import Dict, Union, Any, Optional, TYPE_CHECKING

import PIL.Image
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.figure
    import PIL
    import plotly


class Tracker(object):

    def start_tracking(self,
                       run_id: Optional[str] = None,
                       run_name: Optional[str] = None,
                       nested: bool = False,
                       tags: Optional[Dict[str, Any]] = None,
                       description: Optional[str] = None,
                       log_system_metrics: Optional[bool] = None):
        raise NotImplementedError()

    def stop_tracking(self):
        raise NotImplementedError()

    def get_state_dict(self, artifact_path):
        raise NotImplementedError()

    def log_state_dict(self, state_dict, artifact_path):
        raise NotImplementedError()

    def log_metrics(self, metrics: Dict[str, float], step: int, synchronous: Union[bool, None] = None) -> None:
        raise NotImplementedError()

    def log_metric(self, key, value, step: Optional[int] = None, synchronous: Union[bool, None] = None) -> None:
        raise NotImplementedError()

    def log_table(self,
                  data: Union[Dict[str, Any], "pd.DataFrame"],
                  artifact_file: str):
        raise NotImplementedError()

    def log_image(self, image: Union["np.ndarray", "PIL.Image.Image"], artifact_file: str):
        raise NotImplementedError()

    def log_figure(self, figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
                   artifact_file: str, save_kwargs: Optional[Dict[str, Any]] = None):
        raise NotImplementedError()

    def log_params(self, params: Dict[str, Any], synchronous: Union[bool, None] = None):
        raise NotImplementedError()

    def log_param(self, key: str, value: Any, synchronous: Union[bool, None] = None):
        raise NotImplementedError()

    def get_param(self, key: str):
        raise NotImplementedError()

    def get_metric(self, key: str):
        raise NotImplementedError()

    def log_artifact(self, local_file_path, artifact_path):
        raise NotImplementedError()

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        raise NotImplementedError()

    def log_model(self, model, signature, artifact_path: str):
        raise NotImplementedError()

    def infer_signature(self, model, samples):
        raise NotImplementedError()

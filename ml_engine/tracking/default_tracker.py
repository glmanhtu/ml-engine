from typing import Optional, Any, Union, Dict
from ml_engine.tracking.tracker import Tracker


class DefaultTracker(Tracker):
    def start_tracking(self, run_id: Optional[str] = None, run_name: Optional[str] = None, nested: bool = False,
                       tags: Optional[Dict[str, Any]] = None, description: Optional[str] = None,
                       log_system_metrics: Optional[bool] = None):
        pass

    def stop_tracking(self):
        pass

    def get_state_dict(self, artifact_path):
        pass

    def log_state_dict(self, state_dict, artifact_path):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: int, synchronous: Union[bool, None] = None) -> None:
        pass

    def log_metric(self, key, value, step: Optional[int] = None, synchronous: Union[bool, None] = None) -> None:
        pass

    def log_table(self, data: Union[Dict[str, Any], "pd.DataFrame"], artifact_file: str):
        pass

    def log_image(self, image: Union["np.ndarray", "PIL.Image.Image"], artifact_file: str):
        pass

    def log_figure(self, figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"], artifact_file: str,
                   save_kwargs: Optional[Dict[str, Any]] = None):
        pass

    def log_params(self, params: Dict[str, Any], synchronous: Union[bool, None] = None):
        pass

    def log_param(self, key: str, value: Any, synchronous: Union[bool, None] = None):
        pass

    def get_param(self, key: str):
        pass

    def get_metric(self, key: str):
        pass

    def log_artifact(self, local_file_path, artifact_path):
        pass

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        pass

    def log_model(self, model, signature, artifact_path: str):
        pass

    def infer_signature(self, model, samples):
        pass
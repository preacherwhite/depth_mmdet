import torch
from mmengine.runner.checkpoint import load_from_local

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = load_from_local(path,map_location='cpu')

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

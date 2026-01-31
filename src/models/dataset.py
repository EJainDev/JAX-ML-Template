from operator import index
from typing import SupportsIndex
import grain.python as pygrain
import numpy as np

from ..config import *


class Datasource(pygrain.RandomAccessDataSource):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.array(X)
        self.y = np.array(y)

    def load_data(self) -> None:
        pass

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: SupportsIndex) -> dict[str, np.ndarray]:
        return {
            "features": self.X[index(idx)],
            "targets": self.y[index(idx)],
        }

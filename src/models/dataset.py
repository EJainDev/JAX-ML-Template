from operator import index
from typing import SupportsIndex
import grain.python as pygrain
import numpy as np


class Datasource(pygrain.RandomAccessDataSource):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.array(x)
        self.y = np.array(y)

    def load_data(self) -> None:
        pass

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: SupportsIndex) -> dict[str, np.ndarray]:
        return {
            "features": self.x[index(idx)],
            "targets": self.y[index(idx)],
        }

import torch

from typing import Tuple, List

from torchio.transforms import RescaleIntensity as TorchIORescaleIntensity
from torchio.transforms import ZNormalization as TorchIOZNormalization
from torchio.transforms import HistogramStandardization as TorchIOHistogramStandardization


class RescaleIntensity:
    def __init__(
            self,
            fields: List[str],
            out_min_max: Tuple[float, float],
            percentiles: Tuple[int, int] = (0, 100),
            mask_field: str = None,
            p: float = 1,
    ):
        self.fields = fields
        self.mask_field = mask_field
        self.tform = TorchIORescaleIntensity(out_min_max, percentiles, mask_field, p)

    def __call__(self, data: dict) -> dict:
        for field in self.fields:
            if self.mask_field is None:
                mask = torch.ones(*data[field].shape).bool()
            else:
                mask = torch.from_numpy(data[self.mask_field]).bool()

            data_tensor = self.tform.rescale(
                tensor=torch.from_numpy(data[field]),
                mask=mask,
                image_name=field
            )

            data[field] = data_tensor.numpy()

        return data


class ZNormalization:
    def __init__(
            self,
            fields: List[str],
            mask_field: str = None,
            p: float = 1,
    ):
        self.fields = fields
        self.mask_field = mask_field
        self.tform = TorchIOZNormalization(mask_field, p)

    def __call__(self, data: dict) -> dict:
        for field in self.fields:
            if self.mask_field is None:
                mask = torch.ones(*data[field].shape).bool()
            else:
                mask = torch.from_numpy(data[self.mask_field]).bool()

            data_tensor = self.tform.znorm(
                tensor=torch.from_numpy(data[field]),
                mask=mask
            )

            data[field] = data_tensor.numpy()

        return data


class HistogramStandardization:
    def __init__(
            self,
            fields: List[str],
            mask_field: str = None,
            p: float = 1,
    ):
        pass
import torch

from typing import Tuple, List

from torchio.transforms import RescaleIntensity as TorchIORescaleIntensity
from torchio.transforms import ZNormalization as TorchIOZNormalization
from torchio.transforms import HistogramStandardization as TorchIOHistogramStandardization

from torchio.transforms import CropOrPad as TorchIOCropOrPad
from torchio.transforms import Crop as TorchIOCrop
from torchio.transforms import Pad as TorchIOPad
from torchio.transforms import Resample as TorchIOResample
from torchio.transforms import ToCanonical as TorchIOToCanonical


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
    """
    Performs ZNormalization using TorchIO. This transformation performs the same functionality as
    torchio.transforms.ZNormalization but exposes an interface compatible with Eisen and its functionality
    uses the data processing paradigm employed in Eisen.
    """
    def __init__(
            self,
            fields: List[str],
            mask_field: str = None,
            p: float = 1,
    ):
        """
        Initializes an object of type eisen_extras.torchio.transforms.ZNormalization

        :param fields:
        :type fields: list of str
        :param mask_field: The field of the data dictionary containing a tensor to be used as mask. Can be None
        :type mask_field: str
        :param p: probability of applying this transform
        :type p: float
        """
        self.fields = fields
        self.mask_field = mask_field
        self.tform = TorchIOZNormalization(mask_field, p)

    def __call__(self, data: dict) -> dict:
        """
        Runs the transform on the data dictionary.

        :param data: dictionary containing data
        :type data: dict
        :return: dictionary containing transformed data
        """
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
            histogram_field: str,
            mask_field: str = None,
            p: float = 1,
    ):
        self.fields = fields
        self.histogram_field = histogram_field
        self.mask_field = mask_field

        self.tform = TorchIOHistogramStandardization(
            landmarks={},  # just for init
            masking_method=mask_field,
            p=p
        )

    def __call__(self, data):
        for field in self.fields:
            if self.mask_field is None:
                mask = torch.ones(*data[field].shape).bool()
            else:
                mask = torch.from_numpy(data[self.mask_field]).bool()

            data_tensor = self.tform.normalize(
                tensor=torch.from_numpy(data[field]),
                landmarks=data[self.histogram_field],
                mask=mask
            )

            data[field] = data_tensor.numpy()

        return data


class CropOrPad:
    def __init__(
            self,
            fields: List[str],
            target_shape: int,
            padding_mode: str,
            mask_field: str = None,
            p: float = 1,
    ):
        self.fields = fields
        self.mask_field = mask_field
        self.tform = TorchIOCropOrPad(target_shape, padding_mode, mask_field, p)

    def __call__(self, data: dict) -> dict:
        pass


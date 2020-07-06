import numpy as np
import torchio
import torch

from torchio.data import Subject, Image
from torchio.transforms import RescaleIntensity as TorchIORescaleIntensity
from torchio.transforms import ZNormalization as TorchIOZNormalization
from eisen_extras.torchio.transforms import RescaleIntensity
from eisen_extras.torchio.transforms import ZNormalization


class TestRescaleIntensity:
    def setup_class(self):
        self.data = {
            'data': np.random.rand(10, 10, 10).astype(dtype=np.float32) * 255,
            'mask': (np.random.rand(10, 10, 10) > 0.5).astype(dtype=bool)
        }

        self.subject = Subject(
            data=Image(tensor=self.data['data'], type=torchio.INTENSITY),  # 4D input required
            mask={'data': torch.from_numpy(self.data['mask'][np.newaxis])}
        )

    def test_min_max(self):
        tform_ref = TorchIORescaleIntensity(
            out_min_max=[0.0, 1.0]
        )

        tform_tst = RescaleIntensity(
            ['data'],
            out_min_max=[0.0, 1.0]
        )

        reference_result = tform_ref.apply_transform(self.subject)

        test_result = tform_tst(self.data)

        assert isinstance(reference_result['data'].tensor.numpy(), np.ndarray)
        assert isinstance(test_result['data'], np.ndarray)

        assert np.all(reference_result['data'].tensor.numpy() == test_result['data'])

    def test_percentiles(self):
        tform_ref = TorchIORescaleIntensity(
            out_min_max=[0.0, 1.0],
            percentiles=(0.05, 0.95)
        )

        tform_tst = RescaleIntensity(
            ['data'],
            out_min_max=[0.0, 1.0],
            percentiles=(0.05, 0.95)
        )

        reference_result = tform_ref.apply_transform(self.subject)

        test_result = tform_tst(self.data)

        assert isinstance(reference_result['data'].tensor.numpy(), np.ndarray)
        assert isinstance(test_result['data'], np.ndarray)

        assert np.all(reference_result['data'].tensor.numpy() == test_result['data'])

    def test_with_mask(self):
        tform_ref = TorchIORescaleIntensity(
            out_min_max=[0.0, 1.0],
            masking_method='mask'
        )

        tform_tst = RescaleIntensity(
            ['data'],
            out_min_max=[0.0, 1.0],
            mask_field='mask'
        )

        test_result = tform_tst(self.data)
        reference_result = tform_ref.apply_transform(self.subject)

        assert isinstance(reference_result['data'].tensor.numpy(), np.ndarray)
        assert isinstance(test_result['data'], np.ndarray)

        assert np.all(reference_result['data'].tensor.numpy() == test_result['data'])


class TestZNormalization:
    def setup_class(self):
        self.data = {
            'data': np.random.rand(10, 10, 10).astype(dtype=np.float32) * 255,
            'mask': (np.random.rand(10, 10, 10) > 0.5).astype(dtype=bool)
        }

        self.subject = Subject(
            data=Image(tensor=self.data['data'], type=torchio.INTENSITY),  # 4D input required
            mask={'data': torch.from_numpy(self.data['mask'][np.newaxis])}
        )

    def test_no_mask(self):
        tform_ref = TorchIOZNormalization()

        tform_tst = ZNormalization(
            ['data'],
        )

        test_result = tform_tst(self.data)
        reference_result = tform_ref.apply_transform(self.subject)

        assert isinstance(reference_result['data'].tensor.numpy(), np.ndarray)
        assert isinstance(test_result['data'], np.ndarray)

        assert np.all(reference_result['data'].tensor.numpy() == test_result['data'])

    def test_mask(self):
        tform_ref = TorchIOZNormalization(
            masking_method='mask'
        )

        tform_tst = ZNormalization(
            ['data'],
            mask_field='mask'
        )

        test_result = tform_tst(self.data)
        reference_result = tform_ref.apply_transform(self.subject)

        assert isinstance(reference_result['data'].tensor.numpy(), np.ndarray)
        assert isinstance(test_result['data'], np.ndarray)

        assert np.all(reference_result['data'].tensor.numpy() == test_result['data'])



from typing import Callable

import pytest
from optuna.samplers import BaseSampler
from optuna.testing.pytest_samplers import BasicSamplerTestCase

import rustuna
from rustuna.converter import ToOptunaSampler


class TestTpeSampler(BasicSamplerTestCase):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: ToOptunaSampler(rustuna.Sampler.tpe())

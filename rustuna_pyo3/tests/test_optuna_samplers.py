from typing import Callable

from optuna.samplers import BaseSampler
from optuna.testing.pytest_samplers import BasicSamplerTestCase
import pytest

import rustuna
from rustuna.converter import ToOptunaSampler


class TestTpeSampler(BasicSamplerTestCase):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: ToOptunaSampler(rustuna.Sampler.tpe())

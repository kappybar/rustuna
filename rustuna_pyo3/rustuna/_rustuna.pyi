import enum
from typing import Callable, Literal, Protocol, Sequence, TypedDict

CategoricalChoiceType = float | int | str | bool | None
DistributionDict = (
    FloatDistributionDict | IntDistributionDict | CategoricalDistributionDict
)

FloatDistributionDict = TypedDict(
    "FloatDistributionDict",
    {
        "type": Literal["FloatDistribution"],
        "low": float,
        "high": float,
        "log": bool,
        "step": float | None,
    },
)
IntDistributionDict = TypedDict(
    "IntDistributionDict",
    {
        "type": Literal["IntDistribution"],
        "low": int,
        "high": int,
        "log": bool,
        "step": int | None,
    },
)
CategoricalDistributionDict = TypedDict(
    "CategoricalDistributionDict",
    {
        "type": Literal["CategoricalDistribution"],
        "choices": list[CategoricalChoiceType],
    },
)

# Distribution
class Distribution:
    @classmethod
    def float(
        cls, low: float, high: float, log: bool = False, step: float | None = None
    ) -> Distribution: ...
    @classmethod
    def int(
        cls, low: int, high: int, log: bool = False, step: int | None = None
    ) -> Distribution: ...
    @classmethod
    def categorical(cls, choices: list[CategoricalChoiceType]) -> Distribution: ...
    def to_dict(self) -> DistributionDict: ...

# Trial
class Trial:
    number: int

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        step: float | None = None,
        log: bool = False,
    ) -> float: ...
    def suggest_int(
        self, name: str, low: int, high: int, step: int | None = None, log: bool = False
    ) -> int: ...
    def suggest_categorical(
        self, name: str, choices: list[CategoricalChoiceType]
    ) -> CategoricalChoiceType: ...
    def set_user_attr(self, key: str, value: str) -> None: ...

class TrialState(enum.IntEnum):
    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    WAITING = 3
    FAIL = 4

    def is_finished(self) -> bool: ...

class PersistedTrial:
    def __init__(
        self,
        study_id: int,
        number: int,
        state: TrialState,
        values: list[float] | None = None,
        internal_params: dict[str, float] | None = None,
        distributions: dict[str, Distribution] | None = None,
        user_attrs: dict[str, str] | None = None,
        system_attrs: dict[str, str] | None = None,
    ) -> None: ...
    @property
    def study_id(self) -> int: ...
    @property
    def number(self) -> int: ...
    @property
    def state(self) -> TrialState: ...
    @property
    def values(self) -> list[float] | None: ...
    @property
    def distributions(self) -> dict[str, Distribution]: ...
    @property
    def user_attrs(self) -> dict[str, str]: ...
    @property
    def system_attrs(self) -> dict[str, str]: ...
    @property
    def internal_params(self) -> dict[str, float]: ...
    @property
    def params(self) -> dict[str, CategoricalChoiceType]: ...

# Study
ObjectiveFuncType = Callable[[Trial], float | tuple[float, ...]]

def create_study(
    *,
    study_name: str | None = None,
    storage: Storage | StorageProtocol | None = None,
    sampler: Sampler | SamplerProtocol | None = None,
    direction: Literal["minimize"] | Literal["maximize"] | None = None,
    directions: list[Literal["minimize"] | Literal["maximize"]] | None = None,
) -> Study: ...
def load_study(
    *,
    study_name: str | None = None,
    storage: Storage | StorageProtocol | None = None,
    sampler: Sampler | SamplerProtocol | None = None,
) -> Study: ...
def get_param_importance(study: Study) -> list[list[float]]: ...

class Study:
    def __init__(
        self,
        study_id: int,
        name: str,
        directions: list[StudyDirection],
        storage: Storage,
        sampler: Sampler,
    ) -> None: ...
    def ask(self) -> Trial: ...
    def tell(
        self,
        number: int,
        values: float | None = None,
        state: TrialState | None = None,
    ) -> Trial: ...
    def optimize(self, objective: ObjectiveFuncType, n_trials: int) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def directions(self) -> list[StudyDirection]: ...
    @property
    def best_trial(self) -> PersistedTrial: ...
    @property
    def trials(self) -> list[PersistedTrial]: ...
    @property
    def best_trials(self) -> list[PersistedTrial]: ...

class StudyDirection(enum.IntEnum):
    MINIMIZE = 0
    MAXIMIZE = 1

class PersistedStudy:
    def __init__(
        self,
        id: int,
        name: str,
        directions: list[StudyDirection],
        user_attrs: dict[str, str] | None = None,
        system_attrs: dict[str, str] | None = None,
    ) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def directions(self) -> list[StudyDirection]: ...
    @property
    def user_attrs(self) -> dict[str, str]: ...
    @property
    def system_attrs(self) -> dict[str, str]: ...

## Storage
class StorageProtocol(Protocol):
    @property
    def is_distributed(self) -> bool: ...
    def create_new_study(
        self, study_name: str, directions: list[StudyDirection]
    ) -> PersistedStudy: ...
    def create_new_trial(self, study_id: int) -> PersistedTrial: ...
    def set_trial_param(
        self,
        study_id: int,
        trial_number: int,
        name: str,
        distribution: Distribution,
        value: float,
    ) -> None: ...
    def set_trial_state_values(
        self,
        study_id: int,
        trial_number: int,
        state: TrialState,
        values: None | list[float] = None,
    ) -> None: ...
    def get_studies(self) -> list[PersistedStudy]: ...
    def get_study(self, study_id: int) -> PersistedStudy: ...
    def get_trials(self, study_id: int) -> list[PersistedTrial]: ...
    def get_trial(self, study_id: int, trial_number: int) -> PersistedTrial: ...
    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_system_attrs(
        self, study_id: int, trial_number: int, attrs: dict[str, str]
    ) -> None: ...
    def set_trial_user_attrs(
        self, study_id: int, trial_number: int, attrs: dict[str, str]
    ) -> None: ...

class Storage:
    @classmethod
    def in_memory(cls) -> Storage: ...
    def create_new_study(
        self, study_name: str, directions: list[StudyDirection]
    ) -> PersistedStudy: ...
    def create_new_trial(self, study_id: int) -> PersistedTrial: ...
    def set_trial_param(
        self,
        study_id: int,
        trial_number: int,
        name: str,
        distribution: Distribution,
        value: float,
    ) -> None: ...
    def set_category_labels(
        self,
        study_id: int,
        param_name: str,
        labels: list[None | bool | int | float | str],
    ) -> None: ...
    def get_category_labels(
        self,
        study_id: int,
        param_name: str,
    ) -> list[None | bool | int | float | str]: ...
    def set_trial_state_values(
        self,
        study_id: int,
        trial_number: int,
        state: TrialState,
        values: None | list[float] = None,
    ) -> None: ...
    def get_studies(self) -> list[PersistedStudy]: ...
    def get_study(self, study_id: int) -> PersistedStudy: ...
    def get_trials(self, study_id: int) -> list[PersistedTrial]: ...
    def get_trial(self, study_id: int, trial_number: int) -> PersistedTrial: ...
    def set_study_system_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_study_user_attrs(self, study_id: int, attrs: dict[str, str]) -> None: ...
    def set_trial_system_attrs(
        self, study_id: int, trial_number: int, attrs: dict[str, str]
    ) -> None: ...
    def set_trial_user_attrs(
        self, study_id: int, trial_number: int, attrs: dict[str, str]
    ) -> None: ...

# Sampler
class SamplerContext:
    study_id: int
    trial_number: int
    directions: list[StudyDirection]

    def __init__(
        self,
        study_id: int,
        trial_number: int,
        directions: list[StudyDirection],
    ) -> None: ...

class SamplerProtocol(Protocol):
    @property
    def support_joint_sampling(self) -> bool: ...
    def sample_joint(
        self,
        ctx: SamplerContext,
        storage: Storage,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]: ...
    def sample_independent(
        self,
        ctx: SamplerContext,
        storage: Storage,
        name: str,
        distribution: Distribution,
    ) -> float: ...

class Sampler:
    @classmethod
    def tpe(cls, seed: int | None = None) -> Sampler: ...
    @classmethod
    def random(cls, seed: int | None = None) -> Sampler: ...
    @classmethod
    def nsgaii(
        cls,
        seed: int | None = None,
        population_size: int = 50,
        mutation_prob: float | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
    ) -> Sampler: ...
    @property
    def support_joint_sampling(self) -> bool: ...
    def sample_joint(
        self,
        ctx: SamplerContext,
        storage: Storage,
        search_space: dict[str, Distribution],
    ) -> dict[str, float]: ...
    def sample_independent(
        self,
        ctx: SamplerContext,
        storage: Storage,
        name: str,
        distribution: Distribution,
    ) -> float: ...

# Private APIs for rustuna.optuna package.
def _get_param_importance_from_list(
    features: list[list[float]],
    targets: list[float],
    n_trees: int,
) -> list[float]: ...

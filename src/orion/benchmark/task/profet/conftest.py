import pytest
from .profet_task import MetaModelTrainingConfig

@pytest.fixture(scope="session")
def profet_train_config():
    # TODO: Figure out a good set of values that makes the training of the meta-model faster.
    quick_train_config = MetaModelTrainingConfig(
        num_burnin_steps=100,
    )
    return quick_train_config 

"""Common fixtures for functional tests"""

# Import fixtures from command.conftest
# to be used in test_storage_resource
# Need to also import parent fixtures to make fixtures work
from commands.conftest import (
    one_experiment,
    pkl_experiments,
    pkl_experiments_and_benchmarks,
    testing_helpers,
    three_experiments_branch_same_name,
    three_experiments_branch_same_name_trials,
    three_experiments_branch_same_name_trials_benchmarks,
    two_experiments_same_name,
)

# 'Use' imported fixtures here, to avoid being considered as unused imports by formatting tools
assert one_experiment
assert two_experiments_same_name
assert three_experiments_branch_same_name
assert three_experiments_branch_same_name_trials
assert three_experiments_branch_same_name_trials_benchmarks
assert pkl_experiments
assert pkl_experiments_and_benchmarks
assert testing_helpers

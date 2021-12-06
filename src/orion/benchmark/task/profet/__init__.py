"""Surrogate (simulated) tasks created using the Profet algorithm.

For a detailed description of Profet, see original paper at https://arxiv.org/abs/1905.12982 or
souce code at https://github.com/EmuKit/emukit/tree/main/emukit/examples/profet

Klein, Aaron, Zhenwen Dai, Frank Hutter, Neil Lawrence, and Javier Gonzalez. "Meta-surrogate
benchmarking for hyperparameter optimization." Advances in Neural Information Processing Systems 32
(2019): 6270-6280.
"""
from .profet_task import ProfetTask
from .svm import SvmTask, SvmTaskHParams
from .fcnet import FcNetTask, FcNetTaskHParams
from .forrester import ForresterTask, ForresterTaskHParams
from .xgboost import XgBoostTask, XgBoostTaskHParams

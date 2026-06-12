from .base import TrainProblem
from .deep_mnist_mlp import DeepMNISTMLPProblem
from .mf_input import MatrixFactorizationInputProblem
from .matrix_sensing import MatrixSensingProblem
from .mlp_digits import SmallMLPDigitsProblem
from .mnist_convnet import MNISTConvNetProblem
from .mnist_mlp import MNISTMLPProblem
from .mnist_patch_classifier import MNISTPatchClassifierProblem

__all__ = [
    "TrainProblem",
    "DeepMNISTMLPProblem",
    "MatrixFactorizationInputProblem",
    "MatrixSensingProblem",
    "SmallMLPDigitsProblem",
    "MNISTConvNetProblem",
    "MNISTMLPProblem",
    "MNISTPatchClassifierProblem",
]

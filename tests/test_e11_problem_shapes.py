import torch

from e11_condition_geometry.config import ProblemSpec
from e11_condition_geometry.runner import build_problem


def test_problem_layer_counts_and_activation_definitions():
    dtype = torch.float64
    device = torch.device("cpu")
    specs = [
        ProblemSpec(family="MatrixFactorizationInput", setting="mf", steps=1, d=12, rank=3, kappa=10.0, num_factors=10),
        ProblemSpec(family="MatrixSensing", setting="ms", steps=1, d=12, rank=3, kappa=10.0),
        ProblemSpec(family="SmallMLPDigits", setting="mlp", steps=1, d=64, rank=10, kappa=1.0, hidden_dim=8, num_samples=64),
        ProblemSpec(family="MNISTMLP", setting="mnist", steps=1, input_dim=784, hidden_dim=8, output_dim=10, num_samples=32),
        ProblemSpec(
            family="DeepMNISTMLP",
            setting="deep mnist",
            steps=1,
            input_dim=784,
            hidden_dim=8,
            output_dim=10,
            num_samples=32,
            num_factors=4,
        ),
        ProblemSpec(
            family="MNISTPatchClassifier",
            setting="mnist patch",
            steps=1,
            d=4,
            rank=5,
            hidden_dim=8,
            output_dim=10,
            num_samples=32,
            batch_size=8,
        ),
        ProblemSpec(
            family="MNISTConvNet",
            setting="mnist conv",
            steps=1,
            d=4,
            rank=5,
            hidden_dim=8,
            output_dim=10,
            num_samples=32,
            batch_size=8,
        ),
    ]
    expected_layers = {
        "MatrixFactorizationInput": 10,
        "MatrixSensing": 1,
        "SmallMLPDigits": 2,
        "MNISTMLP": 2,
        "DeepMNISTMLP": 4,
        "MNISTPatchClassifier": 2,
        "MNISTConvNet": 2,
    }
    expected_a = {
        "MatrixFactorizationInput": "downstream_activation_product",
        "MatrixSensing": "measurement_operator_proxy",
        "SmallMLPDigits": "full_layer_input_activation",
        "MNISTMLP": "full_layer_input_activation",
        "DeepMNISTMLP": "full_layer_input_activation",
        "MNISTPatchClassifier": "full_patch_and_classifier_activation",
        "MNISTConvNet": "full_conv_patch_and_classifier_activation",
    }
    for spec in specs:
        problem = build_problem(spec, seed=0, device=device, dtype=dtype)
        assert len(problem.parameters()) == expected_layers[spec.family]
        assert len(problem.activation_matrices()) == expected_layers[spec.family]
        assert problem.diagnostic_a_definition == expected_a[spec.family]


def test_mlp_training_batch_is_separate_from_full_activation_diagnostics():
    dtype = torch.float64
    device = torch.device("cpu")
    spec = ProblemSpec(
        family="SmallMLPDigits",
        setting="mlp batch",
        steps=1,
        d=64,
        rank=10,
        kappa=1.0,
        hidden_dim=8,
        num_samples=64,
        batch_size=16,
    )
    problem = build_problem(spec, seed=0, device=device, dtype=dtype)

    problem.set_train_step(0)
    assert problem.logits().shape[0] == 16
    assert problem.loss().ndim == 0

    activations = problem.activation_matrices()
    assert activations[0].shape[0] == 64
    assert activations[1].shape[0] == 64


def test_mnist_patch_training_batch_is_separate_from_full_activation_diagnostics():
    dtype = torch.float64
    device = torch.device("cpu")
    spec = ProblemSpec(
        family="MNISTPatchClassifier",
        setting="mnist patch batch",
        steps=1,
        d=4,
        rank=5,
        hidden_dim=8,
        output_dim=10,
        num_samples=32,
        batch_size=8,
    )
    problem = build_problem(spec, seed=0, device=device, dtype=dtype)

    problem.set_train_step(0)
    assert problem.logits().shape[0] == 8
    assert problem.loss().ndim == 0

    activations = problem.activation_matrices()
    assert activations[0].shape[0] > 32
    assert activations[0].shape[1] == 25
    assert activations[1].shape == (32, 8)


def test_mnist_conv_training_batch_is_separate_from_full_activation_diagnostics():
    dtype = torch.float64
    device = torch.device("cpu")
    spec = ProblemSpec(
        family="MNISTConvNet",
        setting="mnist conv batch",
        steps=1,
        d=4,
        rank=5,
        hidden_dim=8,
        output_dim=10,
        num_samples=32,
        batch_size=8,
    )
    problem = build_problem(spec, seed=0, device=device, dtype=dtype)

    problem.set_train_step(0)
    assert problem.parameters()[0].shape == (8, 1, 5, 5)
    assert problem.logits().shape[0] == 8
    assert problem.loss().ndim == 0

    activations = problem.activation_matrices()
    assert activations[0].shape[0] > 32
    assert activations[0].shape[1] == 25
    assert activations[1].shape == (32, 8)

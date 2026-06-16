import torch

from e11_condition_geometry.config import ProblemSpec
from e11_condition_geometry.cifar100_resnet_tail import (
    Cifar100ResNetOneStepConfig,
    build_cifar_resnet_model,
    dataset_num_classes,
)
from e11_condition_geometry.runner import build_problem
from scripts.e11_run_cifar100_resnet_layer_jvp_checkpoint_prediction import parse_class_list


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


def test_synthetic_problems_use_noisy_minibatches_with_full_diagnostics():
    dtype = torch.float64
    device = torch.device("cpu")
    mf_spec = ProblemSpec(
        family="MatrixFactorizationInput",
        setting="mf noisy minibatch",
        steps=1,
        d=12,
        rank=3,
        kappa=10.0,
        num_factors=10,
        batch_size=16,
        noise_std=1e-2,
    )
    mf = build_problem(mf_spec, seed=0, device=device, dtype=dtype)
    mf.set_train_step(0)
    assert mf.loss().ndim == 0
    assert mf._train_indices.numel() == 16
    assert mf.activation_matrices()[0].shape[1] == 120
    assert not torch.allclose(mf.clean_target_output, mf.target_output)

    ms_spec = ProblemSpec(
        family="MatrixSensing",
        setting="ms noisy minibatch",
        steps=1,
        d=12,
        rank=3,
        kappa=10.0,
        batch_size=16,
        noise_std=1e-2,
    )
    ms = build_problem(ms_spec, seed=0, device=device, dtype=dtype)
    ms.set_train_step(0)
    assert ms.loss().ndim == 0
    assert ms._train_indices.numel() == 16
    assert ms.activation_matrices()[0].shape[0] == 72
    assert not torch.allclose(ms.clean_observations, ms.observations)


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


def test_cifar_resnet_heldout_architecture_and_data_config_shapes():
    config = Cifar100ResNetOneStepConfig(
        dataset_name="CIFAR10",
        model_arch="resnet34",
        head_classes=tuple(range(5)),
        tail_classes=tuple(range(5, 10)),
    )

    model = build_cifar_resnet_model(config, device=torch.device("cpu"), dtype=torch.float32)

    assert dataset_num_classes(config.dataset_name) == 10
    assert model.fc.out_features == 10
    assert model.conv1.kernel_size == (3, 3)
    assert parse_class_list("0,1,2,3,4") == tuple(range(5))


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

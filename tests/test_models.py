import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from minesweeper_ml.training import build_cnn_model


class FakeTensor:
    def __init__(self, shape, layers=None):
        self.shape = shape
        self.layers = list(layers or [])


class FakeLayer:
    def __init__(self, kind, **config):
        self.kind = kind
        self.config = config

    def __call__(self, tensor):
        return FakeTensor(tensor.shape, tensor.layers + [self])


class FakeFunctionalModel:
    def __init__(self, inputs, outputs):
        self.kind = "functional"
        self.inputs = inputs
        self.outputs = outputs
        self.layers = (
            outputs.layers
            if isinstance(outputs, FakeTensor)
            else next(iter(outputs.values())).layers
        )
        self.compile_options = None

    def compile(self, **options):
        self.compile_options = options


class FakeSequentialModel:
    def __init__(self, layers):
        self.kind = "sequential"
        self.layers = layers
        self.compile_options = None

    def compile(self, **options):
        self.compile_options = options


def fake_keras_modules():
    layer_module = ModuleType("tensorflow.keras.layers")
    layer_module.Input = lambda shape: FakeTensor(shape)
    layer_module.Conv2D = lambda filters, kernel_size, **kwargs: FakeLayer(
        "Conv2D",
        filters=filters,
        kernel_size=kernel_size,
        **kwargs,
    )
    layer_module.Dense = lambda units, **kwargs: FakeLayer("Dense", units=units, **kwargs)
    layer_module.Flatten = lambda: FakeLayer("Flatten")
    layer_module.MaxPooling2D = lambda pool_size: FakeLayer(
        "MaxPooling2D",
        pool_size=pool_size,
    )

    keras_module = ModuleType("tensorflow.keras")
    keras_module.layers = layer_module
    fake_keras = SimpleNamespace(
        Input=layer_module.Input,
        layers=layer_module,
        Model=FakeFunctionalModel,
        Sequential=FakeSequentialModel,
        metrics=SimpleNamespace(
            BinaryAccuracy=lambda name: SimpleNamespace(name=name),
        ),
    )

    tensorflow_module = ModuleType("tensorflow")
    tensorflow_module.keras = fake_keras
    return {
        "tensorflow": tensorflow_module,
        "tensorflow.keras": keras_module,
        "tensorflow.keras.layers": layer_module,
    }


class ModelBuilderTest(unittest.TestCase):
    def test_cnn_is_a_fully_convolutional_spatial_safety_map(self):
        with patch.dict(sys.modules, fake_keras_modules()):
            model = build_cnn_model(width=5, height=4)

        self.assertEqual("functional", model.kind)
        self.assertEqual(10, model.inputs.shape[2])
        self.assertEqual(
            ["Conv2D", "Conv2D", "Conv2D", "Conv2D"],
            [layer.kind for layer in model.layers],
        )
        self.assertEqual(1, model.layers[-1].config["filters"])
        self.assertEqual("sigmoid", model.layers[-1].config["activation"])
        self.assertEqual("binary_crossentropy", model.compile_options["loss"])
        self.assertEqual(
            "masked_accuracy",
            model.compile_options["weighted_metrics"][0].name,
        )

    def test_multitask_cnn_has_named_safety_and_value_heads(self):
        with patch.dict(sys.modules, fake_keras_modules()):
            model = build_cnn_model(
                width=5,
                height=4,
                value_head=True,
            )

        self.assertEqual({"safety", "value"}, set(model.outputs))
        self.assertEqual(
            "binary_crossentropy",
            model.compile_options["loss"]["safety"],
        )
        self.assertEqual(
            "binary_crossentropy",
            model.compile_options["loss"]["value"],
        )

    def test_paper_architecture_uses_wider_five_by_five_layers(self):
        with patch.dict(sys.modules, fake_keras_modules()):
            model = build_cnn_model(
                width=10,
                height=10,
                architecture="paper_5x5",
            )

        spatial_layers = model.layers[:-1]
        self.assertTrue(
            all(layer.config["kernel_size"] == (5, 5) for layer in spatial_layers)
        )
        self.assertGreater(
            max(layer.config["filters"] for layer in spatial_layers),
            64,
        )

    def test_unknown_architecture_is_rejected_without_tensorflow(self):
        with self.assertRaises(ValueError):
            build_cnn_model(5, 5, architecture="unknown")

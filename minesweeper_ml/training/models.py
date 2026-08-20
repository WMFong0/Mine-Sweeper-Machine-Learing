from __future__ import annotations


def build_dense_model(input_output_dim: int):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_output_dim,)),
            keras.layers.Dense(input_output_dim, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(input_output_dim, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


CNN_ARCHITECTURES = {
    "local_3x3": ((32, 64, 32), (3, 3)),
    "paper_5x5": ((64, 128, 64), (5, 5)),
}


def build_cnn_model(
    width: int,
    height: int,
    *,
    architecture: str = "local_3x3",
    value_head: bool = False,
    value_loss_weight: float = 0.5,
):
    if architecture not in CNN_ARCHITECTURES:
        raise ValueError(f"Unknown CNN architecture: {architecture}")
    if value_loss_weight <= 0.0:
        raise ValueError("value_loss_weight must be greater than zero.")
    from tensorflow import keras

    inputs = keras.Input(shape=(None, None, 10))
    filters, kernel_size = CNN_ARCHITECTURES[architecture]
    hidden = inputs
    for filter_count in filters:
        hidden = keras.layers.Conv2D(
            filter_count,
            kernel_size,
            padding="same",
            activation="relu",
        )(hidden)
    safety = keras.layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid",
        name="safety",
    )(hidden)
    outputs = (
        {
            "safety": safety,
            "value": keras.layers.Conv2D(
                1,
                (1, 1),
                activation="sigmoid",
                name="value",
            )(hidden),
        }
        if value_head
        else safety
    )
    model = keras.Model(inputs=inputs, outputs=outputs)
    compile_options = (
        {
            "optimizer": "adam",
            "loss": {
                "safety": "binary_crossentropy",
                "value": "binary_crossentropy",
            },
            "loss_weights": {
                "safety": 1.0,
                "value": value_loss_weight,
            },
            "weighted_metrics": {
                "safety": [
                    keras.metrics.BinaryAccuracy(
                        name="masked_accuracy"
                    )
                ],
                "value": [
                    keras.metrics.BinaryAccuracy(
                        name="masked_value_accuracy"
                    )
                ],
            },
        }
        if value_head
        else {
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "weighted_metrics": [
                keras.metrics.BinaryAccuracy(name="masked_accuracy"),
            ],
        }
    )
    model.compile(**compile_options)
    return model

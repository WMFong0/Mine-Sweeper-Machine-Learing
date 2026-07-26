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


def build_cnn_model(width: int, height: int):
    from tensorflow import keras

    inputs = keras.Input(shape=(height, width, 10))
    hidden = keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
    )(inputs)
    hidden = keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
    )(hidden)
    hidden = keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
    )(hidden)
    outputs = keras.layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid",
    )(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        weighted_metrics=[
            keras.metrics.BinaryAccuracy(name="masked_accuracy"),
        ],
    )
    return model

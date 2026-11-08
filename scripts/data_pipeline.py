
import tensorflow as tf
import pandas as pd

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(SEED)

# Training augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
], name="augmentation")


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    # Resize to a fixed resolution
    image = tf.image.resize(image, IMG_SIZE)

    # Keep pixel values in 0-255 range initially
    image = tf.cast(image, tf.float32)

    return image, label


def build_dataset(csv_path, training=False):
    df = pd.read_csv(csv_path)

    paths = df["filepath"].astype(str).tolist()
    labels = df["label"].astype("float32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    if training:
        dataset = dataset.map(
            lambda x, y: (
                augmentation(x, training=True), y
            ),
            num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


if __name__ == "__main__":
    train_ds = build_dataset(
        "data/splits/train.csv",
        training=True
    )

    val_ds = build_dataset(
        "data/splits/val.csv"
    )

    test_ds = build_dataset(
        "data/splits/test.csv"
    )

    print("Training batches:", len(train_ds))
    print("Validation batches:", len(val_ds))
    print("Testing batches:", len(test_ds))

    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)

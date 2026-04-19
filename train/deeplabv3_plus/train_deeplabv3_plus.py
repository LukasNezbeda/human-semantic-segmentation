"""Train DeepLabV3+ for human semantic segmentation.

Supports:
  - Person segmentation dataset with train/test split.
  - PennFudan dataset using k-fold cross validation.

Example:
    python train/deeplabv3_plus/train_deeplabv3_plus.py --dataset penn_fudan
"""

import argparse
import os
import sys

# Add parent directory to path to enable imports
# Allows to reach the models and metrics module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging

# Pokud je zabraná karta 0, použijeme kartu 1 pro trénování
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from glob import glob
from typing import Sequence

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ( # type: ignore
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.metrics import Precision, Recall # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from metrics.metrics import combined_loss, dice_coef, dice_loss, iou
from models.deeplabv3_plus import deeplabv3_plus

""" Global parameters """
H = 512
W = 512
PERSON_SEGMENTATION_DATASET = "person_segmentation"
PENN_FUDAN_DATASET = "penn_fudan"
PENN_FUDAN_ROOT_DEFAULT = os.path.join("data", "penn_fudan", "new_data")

""" Directory creation """
def create_dir(path: str) -> None:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def shuffling(x: Sequence[str], y: Sequence[str]) -> tuple[list[str], list[str]]:
    """Shuffle paired lists with a fixed seed.

    Args:
        x: Image paths.
        y: Mask paths.

    Returns:
        Tuple of shuffled lists.
    """
    # Sounds "None" is not iterable
    x, y = shuffle(x, y, random_state=42) # type: ignore
    return list(x), list(y)


def load_data(path: str) -> tuple[list[str], list[str]]:
    """Load paired image and mask paths from a fold directory.

    Args:
        path: Fold directory containing image/ and mask/ subfolders.

    Returns:
        Tuple of (image_paths, mask_paths).
    """
    x = sorted(glob(os.path.join(path, "image", "*.png")))
    y = sorted(glob(os.path.join(path, "mask", "*.png")))

    return x, y


def read_image(path: bytes) -> np.ndarray:
    """Read and normalize an RGB image.

    Args:
        path: Image path as bytes (TensorFlow input).

    Returns:
        Normalized float32 image array.
    """
    path = path.decode() # Convert bytes to string # type: ignore
    x = cv2.imread(path, cv2.IMREAD_COLOR) # type: ignore
    x = x/255.0 # type: ignore
    x = x.astype(np.float32)
    return x


# Normalizing depends on mask format, binary mask works for values (0 to 255) which does require it
def read_mask(path: bytes) -> np.ndarray:
    """Read a grayscale mask and expand the channel dimension.

    Args:
        path: Mask path as bytes (TensorFlow input).

    Returns:
        Float32 mask array with shape (H, W, 1).
    """
    path = path.decode() # Convert bytes to string # type: ignore
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # type: ignore
    y = y.astype(np.float32) # type: ignore
    y = np.expand_dims(y, axis=-1) # Add channel dimension
    return y


def tf_parse(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """TensorFlow-compatible parsing wrapper.

    Args:
        x: Image path tensor.
        y: Mask path tensor.

    Returns:
        Tuple of image and mask tensors with fixed shapes.
    """
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    # read_image and read_mask are outside of tensorflow environment
    # tf.numpy_function is needed to wrap a python function and use it as Tensorflow op
    # May sound that it is not iterable
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


# Formatting dataset to be tensorflow compatible
def tf_dataset(x: Sequence[str], y: Sequence[str], batch: int = 2) -> tf.data.Dataset:
    """Build a TensorFlow dataset pipeline.

    Args:
        x: Image paths.
        y: Mask paths.
        batch: Batch size.

    Returns:
        TensorFlow dataset.
    """
    datasset = tf.data.Dataset.from_tensor_slices((x, y))
    datasset = datasset.map(tf_parse)
    dataset = datasset.batch(batch)
    dataset = dataset.prefetch(10)

    return dataset


def get_project_root() -> str:
    """Return the repository root directory.

    Returns:
        Absolute path to the project root.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_dataset_root(project_root: str, dataset_root: str) -> str:
    """Resolve dataset root from a relative or absolute path.

    Args:
        project_root: Project root directory.
        dataset_root: User-provided dataset root.

    Returns:
        Absolute dataset root.
    """
    if os.path.isabs(dataset_root):
        return dataset_root
    return os.path.join(project_root, dataset_root)


def list_penn_fudan_folds(penn_root: str) -> list[str]:
    """List PennFudan fold directories sorted by fold index.

    Args:
        penn_root: Root directory containing fold_*/ folders.

    Returns:
        Sorted list of fold directories.
    """
    fold_entries: list[tuple[int, str]] = []
    for name in os.listdir(penn_root):
        if not name.startswith("fold_"):
            continue
        fold_path = os.path.join(penn_root, name)
        if not os.path.isdir(fold_path):
            continue
        try:
            fold_index = int(name.split("_", 1)[1])
        except ValueError:
            continue
        fold_entries.append((fold_index, fold_path))

    fold_entries.sort(key=lambda item: item[0])
    return [path for _, path in fold_entries]


def build_kfold_split(
    fold_dirs: Sequence[str],
    val_fold_index: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Build train and validation lists from k-fold directories.

    Args:
        fold_dirs: List of fold directories.
        val_fold_index: Index of the fold to use for validation.

    Returns:
        Tuple of (train_x, train_y, val_x, val_y).
    """
    train_x: list[str] = []
    train_y: list[str] = []
    val_x: list[str] = []
    val_y: list[str] = []

    for index, fold_dir in enumerate(fold_dirs):
        fold_x, fold_y = load_data(fold_dir)
        if index == val_fold_index:
            val_x, val_y = fold_x, fold_y
        else:
            train_x.extend(fold_x)
            train_y.extend(fold_y)

    return train_x, train_y, val_x, val_y


def build_callbacks(
    model_path: str,
    csv_path: str,
    tensor_logs: str,
) -> list[tf.keras.callbacks.Callback]:
    """Create Keras callbacks for training.

    Args:
        model_path: Output path for the best model.
        csv_path: Output path for training logs.
        tensor_logs: TensorBoard logs directory.

    Returns:
        List of callbacks.
    """
    return [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(csv_path),
        # TensorBoard(log_dir=tensor_logs),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=False),
        # TODO: Callback pro vizualizaci vstupu, predikce a ground truth v každé epoše
    ]


def train_single_run(
    train_x: Sequence[str],
    train_y: Sequence[str],
    val_x: Sequence[str],
    val_y: Sequence[str],
    batch_size: int,
    lr: float,
    num_epochs: int,
    model_path: str,
    csv_path: str,
    tensor_logs: str,
) -> None:
    """Train DeepLabV3+ once with the provided datasets.

    Args:
        train_x: Training image paths.
        train_y: Training mask paths.
        val_x: Validation image paths.
        val_y: Validation mask paths.
        batch_size: Batch size.
        lr: Learning rate.
        num_epochs: Number of epochs.
        model_path: Output model path.
        csv_path: Output CSV log path.
        tensor_logs: TensorBoard log directory.
    """
    train_dataset = tf_dataset(train_x, train_y, batch_size)
    val_dataset = tf_dataset(val_x, val_y, batch_size)

    # Reset state between folds to keep memory usage stable.
    tf.keras.backend.clear_session()

    model = deeplabv3_plus((H, W, 3))
    model.compile(
        loss=combined_loss,
        optimizer=Adam(lr),
        metrics=[dice_coef, iou, Recall(), Precision()],
    )

    callbacks = build_callbacks(model_path, csv_path, tensor_logs)
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional list of arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train DeepLabV3+ on person_segmentation or PennFudan (k-fold).",
    )
    parser.add_argument(
        "--dataset",
        choices=[PERSON_SEGMENTATION_DATASET, PENN_FUDAN_DATASET],
        default=PERSON_SEGMENTATION_DATASET,
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--penn-root",
        default=PENN_FUDAN_ROOT_DEFAULT,
        help="PennFudan k-fold root (contains fold_*/).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training pipeline.

    Args:
        argv: Optional list of CLI arguments.

    Returns:
        Exit code.
    """
    args = parse_args(argv)

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    # """ Directory for storing files """
    # create_dir("files")

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4
    num_epochs = 20

    project_root = get_project_root()

    if args.dataset == PERSON_SEGMENTATION_DATASET:
        """ Dataset"""
        dataset_path = os.path.join(project_root, "data", "person_segmentation", "new_data")
        print(f"Dataset path: {dataset_path}")

        train_path = os.path.join(dataset_path, "train")
        val_path = os.path.join(dataset_path, "test")

        # Load and shuffle training data
        train_x, train_y = load_data(train_path)
        train_x, train_y = shuffling(train_x, train_y)

        # No need to shuffle validation data
        val_x, val_y = load_data(val_path)

        # Sounds that it cannot be sized
        print(f"Training samples: {len(train_x)} | {len(train_y)}")
        print(f"Validation samples: {len(val_x)} | {len(val_y)}")

        model_path = os.path.join(project_root, "runs", "deeplabv3_plus.h5")
        csv_path = os.path.join(project_root, "runs", "training_log.csv")

        tensor_logs = os.path.join(project_root, "runs", "tensor_logs")
        create_dir(tensor_logs)

        train_single_run(
            train_x,
            train_y,
            val_x,
            val_y,
            batch_size,
            lr,
            num_epochs,
            model_path,
            csv_path,
            tensor_logs,
        )
        return 0

    penn_root = resolve_dataset_root(project_root, args.penn_root)
    if not os.path.isdir(penn_root):
        print(f"PennFudan root not found: {penn_root}")
        return 1

    fold_dirs = list_penn_fudan_folds(penn_root)
    if not fold_dirs:
        print(f"No fold directories found under: {penn_root}")
        return 1

    print(f"PennFudan root: {penn_root}")
    print(f"Found {len(fold_dirs)} folds")

    for fold_index in range(len(fold_dirs)):
        train_x, train_y, val_x, val_y = build_kfold_split(fold_dirs, fold_index)
        train_x, train_y = shuffling(train_x, train_y)

        print(
            f"Fold {fold_index}: train={len(train_x)} | {len(train_y)}, "
            f"val={len(val_x)} | {len(val_y)}"
        )

        fold_output_dir = os.path.join(
            project_root,
            "runs",
            "deeplabv3_plus",
            f"fold_{fold_index}",
        )
        create_dir(fold_output_dir)

        tensor_logs = os.path.join(fold_output_dir, "tensor_logs")
        create_dir(tensor_logs)

        model_path = os.path.join(fold_output_dir, "deeplabv3_plus.h5")
        csv_path = os.path.join(fold_output_dir, "training_log.csv")

        train_single_run(
            train_x,
            train_y,
            val_x,
            val_y,
            batch_size,
            lr,
            num_epochs,
            model_path,
            csv_path,
            tensor_logs,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
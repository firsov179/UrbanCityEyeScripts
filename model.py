import os
import gc
import math
import time
from multiprocessing import freeze_support

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import pairwise_distances
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    LSTM,
    TimeDistributed,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import re
import argparse
import multiprocessing as mp
from tqdm import tqdm

import cv2
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

PATCH_SIZE = 25
SEQUENCE_LENGTH = 5
BATCH_SIZE = 32
MAX_WORKERS = 10
STRIDE = 15
SAMPLE_INTERVAL = 5

BASE_DIR = "C:\HSE\Okit\pythonProject2"
PATCHES_DIR = "C:\HSE\Okit\pythonProject2\patches2"
OUTPUT_DIR = "C:\HSE\Okit\pythonProject2\output3\\v1"
MODELS_DIR = "C:\HSE\Okit\pythonProject2\models2"
os.makedirs(PATCHES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

color_to_class = {
    (255, 255, 255): 0,  # empty
    (128, 128, 192): 1,  # building:train_station
    (112, 112, 176): 2,  # building:train_station; railway:station
    (96, 96, 160): 3,  # building:train_station; railway:yard
    (80, 80, 144): 4,  # building:transportation
    (64, 64, 128): 5,  # railway:light_rail
    (80, 80, 144): 6,  # railway:monorail
    (96, 96, 160): 7,  # railway:narrow_gauge
    (112, 112, 176): 8,  # railway:platform
    (128, 128, 192): 9,  # railway:rail
    (144, 144, 208): 10,  # railway:station
    (160, 160, 224): 11,  # railway:subway
    (176, 176, 240): 12,  # railway:tram
    (192, 192, 128): 13,  # highway:bridleway
    (176, 176, 112): 14,  # highway:cycleway
    (160, 160, 96): 15,  # highway:footway
    (144, 144, 80): 16,  # highway:living_street
    (128, 128, 64): 17,  # highway:motorway
    (112, 112, 48): 18,  # highway:motorway_link
    (96, 96, 32): 19,  # highway:no
    (80, 80, 16): 20,  # highway:path
    (64, 64, 0): 21,  # highway:pedestrian
    (48, 48, 16): 22,  # highway:pedestrian; landuse:gravel
    (32, 32, 32): 23,  # highway:primary
    (16, 16, 64): 24,  # highway:primary_link
    (0, 64, 64): 25,  # highway:residential
    (0, 80, 80): 26,  # highway:secondary
    (0, 96, 96): 27,  # highway:secondary_link
    (0, 112, 112): 28,  # highway:service
    (0, 128, 128): 29,  # highway:steps
    (0, 144, 144): 30,  # highway:tertiary
    (0, 160, 160): 31,  # highway:tertiary_link
    (0, 176, 176): 32,  # highway:track
    (0, 192, 192): 33,  # highway:trunk
    (0, 208, 208): 34,  # highway:trunk_link
    (0, 224, 224): 35,  # highway:unclassified
    (64, 128, 192): 36,  # waterway:canal
    (48, 112, 176): 37,  # waterway:dam
    (32, 96, 160): 38,  # waterway:ditch
    (16, 80, 144): 39,  # waterway:dock
    (0, 64, 128): 40,  # waterway:river
    (16, 80, 144): 41,  # waterway:riverbank
    (32, 96, 160): 42,  # waterway:stream
    (192, 128, 192): 43,  # amenity:parking
}

class_to_color = {v: k for k, v in color_to_class.items()}


def extract_year(filename):
    """Извлекает год из имени файла"""
    match = re.search(r"(\d{4})\.png", filename)
    if match:
        return int(match.group(1))
    return None


def closest_color(rgb, colors):
    """Находит ближайший цвет из предопределенного набора"""
    if len(rgb) > 3:
        rgb = rgb[:3]

    rgb = np.array(rgb).reshape(1, -1)
    colors = np.array(list(colors)).reshape(-1, 3)
    distances = pairwise_distances(rgb, colors)
    index_of_nearest = distances.argmin()
    return tuple(colors[index_of_nearest])


def convert_image_to_class_map(image):
    if len(image.shape) == 3 and image.shape[2] > 3:
        print(
            f"Обнаружено изображение с {image.shape[2]} каналами, использую только RGB"
        )
        image = image[:, :, :3]

    height, width = image.shape[:2]
    class_map = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            pixel_rgb = tuple(image[i, j])
            closest = closest_color(pixel_rgb, color_to_class)
            class_map[i, j] = color_to_class[closest]

    return class_map


def process_image_batch(args):
    """Обрабатывает один батч фрагментов изображения"""
    img_path, batch_coords, year_dir, patch_size, stride = args

    try:

        img = cv2.imread(img_path)
        if img is None:
            return []

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        year = extract_year(os.path.basename(img_path))
        if year is None:
            return []

        results = []
        batch_patches = []
        batch_filenames = []

        for y_start, x_start in batch_coords:
            if (
                y_start + patch_size > img.shape[0]
                or x_start + patch_size > img.shape[1]
            ):
                continue

            patch = img[y_start : y_start + patch_size, x_start : x_start + patch_size]

            class_patch = convert_image_to_class_map(patch)

            patch_filename = f"{year}_{y_start}_{x_start}.npy"
            batch_patches.append(class_patch)
            batch_filenames.append(patch_filename)

            results.append((y_start, x_start))

        for i, (patch, filename) in enumerate(zip(batch_patches, batch_filenames)):
            patch_path = os.path.join(year_dir, filename)
            np.save(patch_path, patch)

        return results

    except Exception as e:
        print(f"Ошибка при обработке батча изображения {img_path}: {e}")
        import traceback

        traceback.print_exc()
        return []


def save_batch(args):
    temp_file_path, year_dir, year = args
    data = np.load(temp_file_path, allow_pickle=True)
    patches = data["patches"]
    positions = data["positions"]

    saved_positions = []

    for i, (patch, pos) in enumerate(zip(patches, positions)):
        y_start, x_start = pos
        patch_filename = f"{year}_{y_start}_{x_start}.npy"
        patch_path = os.path.join(year_dir, patch_filename)
        np.save(patch_path, patch)
        saved_positions.append(pos)

    return saved_positions


def create_integrated_model(input_shape, sequence_length, num_classes):
    """
    Создает интегрированную модель, включающую CNN, RF-подобный слой и LSTM.

    Args:
        input_shape: Форма входных данных для одного фрагмента (высота, ширина, каналы)
        sequence_length: Длина временной последовательности
        num_classes: Количество классов

    Returns:
        Модель Keras
    """

    input_sequence = Input(shape=(sequence_length,) + input_shape)

    cnn_model = tf.keras.Sequential(
        [
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(4, activation="relu"),
        ]
    )

    time_distributed_cnn = TimeDistributed(cnn_model)(input_sequence)

    lstm_out = LSTM(64, return_sequences=False)(time_distributed_cnn)
    lstm_out = Dropout(0.3)(lstm_out)

    change_probability = Dense(1, activation="sigmoid", name="change_probability")(
        lstm_out
    )

    class_probabilities = Dense(64, activation="relu")(lstm_out)
    class_probabilities = BatchNormalization()(class_probabilities)
    class_probabilities = Dense(
        num_classes, activation="softmax", name="class_probabilities"
    )(class_probabilities)

    model = Model(
        inputs=input_sequence, outputs=[change_probability, class_probabilities]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            "change_probability": "binary_crossentropy",
            "class_probabilities": "categorical_crossentropy",
        },
        metrics={"change_probability": "accuracy", "class_probabilities": "accuracy"},
        loss_weights={"change_probability": 0.3, "class_probabilities": 0.7},
    )

    return model


def process_chunk(args):
    """Обрабатывает и сразу сохраняет фрагменты"""
    chunk_coords, img, height, width, patch_size, year_dir, year = args

    saved_positions = []

    for y_start, x_start in chunk_coords:
        if y_start + patch_size > height or x_start + patch_size > width:
            continue

        patch = img[y_start : y_start + patch_size, x_start : x_start + patch_size]

        class_map = vectorized_convert_to_class_map(patch)

        patch_filename = f"{year}_{y_start}_{x_start}.npy"
        patch_path = os.path.join(year_dir, patch_filename)
        np.save(patch_path, class_map)

        saved_positions.append((y_start, x_start))

    return saved_positions


def extract_chunk(chunk_file):
    """Извлекает и сохраняет фрагменты из сжатого файла"""
    data = np.load(chunk_file)
    patches = data["patches"]
    positions = data["positions"]

    year_dir = os.path.dirname(chunk_file)
    year = os.path.basename(year_dir)

    extracted_positions = []
    for i, (patch, pos) in enumerate(zip(patches, positions)):
        y_start, x_start = pos
        patch_filename = f"{year}_{y_start}_{x_start}.npy"
        patch_path = os.path.join(year_dir, patch_filename)
        np.save(patch_path, patch)
        extracted_positions.append(tuple(pos))

    os.remove(chunk_file)

    return extracted_positions


def process_image_to_patches(
    img_path, output_dir=PATCHES_DIR, patch_size=PATCH_SIZE, stride=STRIDE
):
    """Оптимизированная версия функции обработки изображения"""

    year = extract_year(os.path.basename(img_path))
    if year is None:
        print(f"Не удалось извлечь год из: {img_path}")
        return None, []

    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    try:

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[2] > 3:
            print(
                f"Обнаружено изображение с {img.shape[2]} каналами, использую только RGB"
            )
            img = img[:, :, :3]

        height, width, _ = img.shape

        n_h = (height - patch_size) // stride + 1
        n_w = (width - patch_size) // stride + 1
        total_patches = n_h * n_w

        print(
            f"Обработка изображения {year} года, размер: {width}x{height}, фрагментов: {total_patches}"
        )

        all_coords = [(i * stride, j * stride) for i in range(n_h) for j in range(n_w)]
        sampled_coords = all_coords[::SAMPLE_INTERVAL]
        print(
            f"Выбрано {len(sampled_coords)} из {total_patches} фрагментов для обработки"
        )

        max_workers = min(40, mp.cpu_count())
        chunk_size = len(sampled_coords) // max_workers

        coords_chunks = [
            sampled_coords[i : i + chunk_size]
            for i in range(0, len(sampled_coords), chunk_size)
        ]

        chunk_args = [
            (chunk, img, height, width, patch_size, year_dir, year)
            for chunk in coords_chunks
        ]

        all_positions = []
        max_concurrent = min(40, mp.cpu_count())
        with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
            for positions in tqdm(
                executor.map(process_chunk, chunk_args),
                total=len(chunk_args),
                desc=f"Обработка {year} года",
            ):
                all_positions.extend(positions)

        positions_file = os.path.join(year_dir, "positions.npy")
        np.save(positions_file, np.array(all_positions))

        print(f"Сохранено {len(all_positions)} фрагментов для {year} года")
        return year, all_positions

    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {e}")
        import traceback

        traceback.print_exc()
        return None, []


def build_training_sequence_batch(args):
    year_sequence, positions, seq_idx, patches_dir, output_dir = args

    sequences_created = 0
    seq_dir = os.path.join(output_dir, f"seq_{seq_idx}")
    os.makedirs(seq_dir, exist_ok=True)

    for pos_idx, pos in enumerate(positions):
        y_start, x_start = pos

        sequence_valid = True
        for year in year_sequence:
            patch_path = os.path.join(
                patches_dir, str(year), f"{year}_{y_start}_{x_start}.npy"
            )
            if not os.path.exists(patch_path):
                sequence_valid = False
                break

        if sequence_valid:

            sequence = []
            for year in year_sequence[:-1]:
                patch_path = os.path.join(
                    patches_dir, str(year), f"{year}_{y_start}_{x_start}.npy"
                )
                patch = np.load(patch_path)
                sequence.append(patch)

            target_year = year_sequence[-1]
            target_path = os.path.join(
                patches_dir, str(target_year), f"{target_year}_{y_start}_{x_start}.npy"
            )
            target_patch = np.load(target_path)
            center_y, center_x = PATCH_SIZE // 2, PATCH_SIZE // 2
            target_class = target_patch[center_y, center_x]

            seq_filename = f"seq_{pos_idx}.npy"
            target_filename = f"target_{pos_idx}.npy"
            np.save(os.path.join(seq_dir, seq_filename), np.array(sequence))
            np.save(os.path.join(seq_dir, target_filename), target_class)

            sequences_created += 1

    return sequences_created


def build_training_data(
    years_processed, all_positions, output_dir=OUTPUT_DIR, seq_length=SEQUENCE_LENGTH
):
    """Создает обучающие данные, сохраняя последовательности и целевые значения с использованием пакетной обработки"""
    print("Создание обучающих данных...")

    sorted_years = sorted(years_processed)

    train_data_dir = os.path.join(output_dir, "train_data")
    os.makedirs(train_data_dir, exist_ok=True)

    process_args = []

    for i in range(len(sorted_years) - seq_length):
        year_sequence = sorted_years[i : i + seq_length + 1]
        positions = all_positions[year_sequence[0]]

        process_args.append((year_sequence, positions, i, PATCHES_DIR, train_data_dir))

    total_sequences = 0

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:

        for seq_count in tqdm(
            executor.map(build_training_sequence_batch, process_args),
            total=len(process_args),
            desc="Создание обучающих последовательностей",
        ):
            total_sequences += seq_count

    print(f"Всего создано {total_sequences} последовательностей для обучения")
    return train_data_dir


def to_one_hot(class_map, num_classes):
    """Преобразует карту классов в формат one-hot"""

    return np.eye(num_classes)[class_map - 1]


def prepare_data_batch(batch_data):
    batch_sequences, num_classes = batch_data

    X_sequences = []
    y_classes = []

    for seq_dir, seq_file in batch_sequences:
        try:
            seq_idx = seq_file.split("_")[1].split(".")[0]

            seq_path = os.path.join(seq_dir, seq_file)
            target_path = os.path.join(seq_dir, f"target_{seq_idx}.npy")

            sequence = np.load(seq_path)
            target_class = np.load(target_path)

            sequence_one_hot = np.array(
                [to_one_hot(patch, num_classes) for patch in sequence]
            )

            target_one_hot = np.zeros(num_classes)
            target_one_hot[target_class - 1] = 1

            X_sequences.append(sequence_one_hot)
            y_classes.append(target_one_hot)
        except Exception as e:
            print(f"Ошибка при загрузке последовательности {seq_dir}/{seq_file}: {e}")
            continue

    if not X_sequences:
        return None, None

    return np.array(X_sequences), np.array(y_classes)


class TrainingDataGenerator:
    """Генератор обучающих данных с пакетной обработкой"""

    def __init__(self, train_data_dir, batch_size=BATCH_SIZE, num_classes=44):
        self.train_data_dir = train_data_dir
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.seq_dirs = [
            os.path.join(train_data_dir, d)
            for d in os.listdir(train_data_dir)
            if os.path.isdir(os.path.join(train_data_dir, d))
        ]

        total_seqs = 0
        for seq_dir in self.seq_dirs:
            seq_files = [
                f
                for f in os.listdir(seq_dir)
                if f.startswith("seq_") and f.endswith(".npy")
            ]
            total_seqs += len(seq_files)

        print(f"Найдено {total_seqs} последовательностей для обучения")
        self.total_sequences = total_seqs

    def __len__(self):
        return self.total_sequences // self.batch_size

    def generate(self):
        """Генератор с эффективным управлением памятью"""

        all_seq_indices = []
        for dir_idx, seq_dir in enumerate(self.seq_dirs):
            seq_files = [
                f
                for f in os.listdir(seq_dir)
                if f.startswith("seq_") and f.endswith(".npy")
            ]
            for file_idx, seq_file in enumerate(seq_files):
                all_seq_indices.append((dir_idx, file_idx))

        while True:

            np.random.shuffle(all_seq_indices)

            for i in range(0, len(all_seq_indices), self.batch_size):
                batch_indices = all_seq_indices[i : i + self.batch_size]

                X_sequences = []
                y_classes = []

                for dir_idx, file_idx in batch_indices:
                    try:
                        seq_dir = self.seq_dirs[dir_idx]

                        seq_files = [
                            f
                            for f in os.listdir(seq_dir)
                            if f.startswith("seq_") and f.endswith(".npy")
                        ]
                        if file_idx >= len(seq_files):
                            continue

                        seq_file = seq_files[file_idx]
                        seq_idx = seq_file.split("_")[1].split(".")[0]

                        seq_path = os.path.join(seq_dir, seq_file)
                        target_path = os.path.join(seq_dir, f"target_{seq_idx}.npy")

                        sequence = np.load(seq_path, mmap_mode="r")
                        target_class = np.load(target_path, mmap_mode="r")

                        sequence_copy = sequence.copy()
                        target_class_copy = target_class.copy()

                        del sequence, target_class

                        sequence_one_hot = np.array(
                            [
                                to_one_hot(patch, self.num_classes)
                                for patch in sequence_copy
                            ]
                        )

                        target_one_hot = np.zeros(self.num_classes)
                        target_one_hot[target_class_copy - 1] = 1

                        X_sequences.append(sequence_one_hot)
                        y_classes.append(target_one_hot)

                        del sequence_copy, target_class_copy, sequence_one_hot
                    except Exception as e:
                        print(
                            f"Ошибка при загрузке последовательности {dir_idx}/{file_idx}: {e}"
                        )
                        continue

                if not X_sequences:
                    continue

                X_np = np.array(X_sequences, dtype=np.float32)
                y_np = np.array(y_classes, dtype=np.float32)

                yield X_np, y_np

                del X_sequences, y_classes, X_np, y_np
                gc.collect()


def simulate_region_parallel(args):
    """Параллельно обрабатывает регион изображения при симуляции"""
    automaton, region, historical_regions, threshold = args

    y_start, y_end, x_start, x_end = region
    padded_y_start, padded_y_end = (
        y_start - automaton.patch_size // 2,
        y_end + automaton.patch_size // 2,
    )
    padded_x_start, padded_x_end = (
        x_start - automaton.patch_size // 2,
        x_end + automaton.patch_size // 2,
    )

    local_history = [
        state[padded_y_start:padded_y_end, padded_x_start:padded_x_end].copy()
        for state in historical_regions
    ]

    current_region = historical_regions[-1][
        padded_y_start:padded_y_end, padded_x_start:padded_x_end
    ].copy()

    new_region = current_region.copy()

    for y in range(
        automaton.patch_size // 2, y_end - y_start + automaton.patch_size // 2
    ):
        for x in range(
            automaton.patch_size // 2, x_end - x_start + automaton.patch_size // 2
        ):

            sequence = []
            for state in local_history:
                patch = state[
                    y - automaton.patch_size // 2 : y + automaton.patch_size // 2 + 1,
                    x - automaton.patch_size // 2 : x + automaton.patch_size // 2 + 1,
                ]
                sequence.append(patch)

            change_prob, class_probs = automaton.predict(sequence)

            if change_prob > threshold:
                new_class = np.argmax(class_probs) + 1
                new_region[y, x] = new_class

    result_region = new_region[
        automaton.patch_size // 2 : -automaton.patch_size // 2,
        automaton.patch_size // 2 : -automaton.patch_size // 2,
    ]

    return (y_start, x_start, result_region)


class InfrastructureCellularAutomaton:
    """Клеточный автомат для моделирования инфраструктуры"""

    def __init__(
        self, num_classes=44, patch_size=PATCH_SIZE, sequence_length=SEQUENCE_LENGTH
    ):
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.sequence_length = sequence_length

        self.input_shape = (patch_size, patch_size, num_classes)

        self.model = create_integrated_model(
            self.input_shape, sequence_length, num_classes
        )

        self.model_trained = False

    def train(self, train_generator, validation_generator=None, epochs=30):
        """Обучает модель с использованием генератора данных и контролем использования памяти"""
        print("Начало обучения модели...")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(MODELS_DIR, "infrastructure_model.h5"), save_best_only=True
            ),
            ModelCheckpoint(
                os.path.join(MODELS_DIR, "infrastructure_model_epoch_{epoch:02d}.h5"),
                save_freq="epoch",
            ),
            tf.keras.callbacks.LambdaCallback(
                on_batch_end=lambda batch, logs: gc.collect()
                if batch % 50 == 0
                else None
            ),
        ]

        def optimized_generator():
            for X_batch, y_batch in train_generator.generate():

                change_labels = []
                for i in range(len(X_batch)):
                    sequence = X_batch[i]
                    target_class = y_batch[i].argmax() + 1
                    last_frame = sequence[-1]
                    center_y, center_x = self.patch_size // 2, self.patch_size // 2
                    last_class = last_frame[center_y, center_x].argmax() + 1

                    changed = 1.0 if last_class != target_class else 0.0
                    change_labels.append(changed)

                change_labels = np.array(change_labels, dtype=np.float32).reshape(-1, 1)

                if len(X_batch.shape) > 5:
                    print(f"Исправление формы X_batch: {X_batch.shape}")
                    X_batch = X_batch.reshape(
                        -1,
                        self.sequence_length,
                        self.patch_size,
                        self.patch_size,
                        self.num_classes,
                    )

                yield X_batch, {
                    "change_probability": change_labels,
                    "class_probabilities": y_batch,
                }

                del X_batch, y_batch, change_labels
                gc.collect()

        def validation_generator_wrapper():
            if validation_generator:
                for X_batch, y_batch in validation_generator.generate():
                    change_labels = []
                    for i in range(len(X_batch)):
                        sequence = X_batch[i]
                        target_class = y_batch[i].argmax() + 1
                        last_frame = sequence[-1]
                        center_y, center_x = self.patch_size // 2, self.patch_size // 2
                        last_class = last_frame[center_y, center_x].argmax() + 1

                        changed = 1.0 if last_class != target_class else 0.0
                        change_labels.append(changed)

                    change_labels = np.array(change_labels, dtype=np.float32).reshape(
                        -1, 1
                    )

                    if len(X_batch.shape) > 5:
                        X_batch = X_batch.reshape(
                            -1,
                            self.sequence_length,
                            self.patch_size,
                            self.patch_size,
                            self.num_classes,
                        )

                    yield X_batch, {
                        "change_probability": change_labels,
                        "class_probabilities": y_batch,
                    }

                    del X_batch, y_batch, change_labels
                    gc.collect()

        steps_per_epoch = min(
            2000, len(train_generator) if hasattr(train_generator, "__len__") else 1000
        )
        validation_steps = (
            min(
                500,
                len(validation_generator)
                if validation_generator and hasattr(validation_generator, "__len__")
                else 250,
            )
            if validation_generator
            else None
        )

        print(
            f"Шагов на эпоху: {steps_per_epoch}, валидационных шагов: {validation_steps}"
        )

        history = self.model.fit(
            optimized_generator(),
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator_wrapper()
            if validation_generator
            else None,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
        )

        self.model_trained = True
        print("Обучение завершено!")

        return history

    def predict(self, sequence):
        """Предсказывает, изменится ли центральный пиксель и какой класс будет у него"""
        if not self.model_trained:
            raise ValueError("Модель не обучена")

        sequence_one_hot = np.array(
            [to_one_hot(patch, self.num_classes) for patch in sequence]
        )

        X = np.expand_dims(sequence_one_hot, axis=0)

        change_prob, class_probs = self.model.predict(X)

        return change_prob[0][0], class_probs[0]

    def predict_batch(self, sequences):
        """Предсказывает изменения и классы для батча последовательностей"""
        if not self.model_trained:
            raise ValueError("Модель не обучена")

        sequences_one_hot = []
        for sequence in sequences:
            seq_one_hot = np.array(
                [to_one_hot(patch, self.num_classes) for patch in sequence]
            )
            sequences_one_hot.append(seq_one_hot)

        X = np.array(sequences_one_hot)

        change_probs, class_probs = self.model.predict(X, batch_size=32, verbose=0)

        return change_probs, class_probs

    def simulate_step_parallel(self, historical_states, threshold=0.5, batch_size=256):
        """Выполняет один шаг симуляции клеточного автомата с пакетной обработкой"""
        new_state = historical_states[-1].copy()
        height, width = new_state.shape

        all_coords = []
        for y in range(self.patch_size // 2, height - self.patch_size // 2):
            for x in range(self.patch_size // 2, width - self.patch_size // 2):
                all_coords.append((y, x))

        num_batches = (len(all_coords) + batch_size - 1) // batch_size

        print(
            f"Всего координат: {len(all_coords)}, будет обработано {num_batches} батчей"
        )

        for batch_idx in tqdm(range(num_batches), desc="Обработка батчей"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(all_coords))

            batch_coords = all_coords[start_idx:end_idx]
            batch_sequences = []

            for y, x in batch_coords:
                sequence = []
                for state in historical_states:
                    patch = state[
                        y - self.patch_size // 2 : y + self.patch_size // 2 + 1,
                        x - self.patch_size // 2 : x + self.patch_size // 2 + 1,
                    ]
                    sequence.append(patch)
                batch_sequences.append(sequence)

            try:
                change_probs, class_probs = self.predict_batch(batch_sequences)

                for i, (y, x) in enumerate(batch_coords):
                    if change_probs[i] > threshold:
                        new_class = np.argmax(class_probs[i]) + 1
                        new_state[y, x] = new_class
            except Exception as e:
                print(f"Ошибка при предсказании для батча {batch_idx}: {e}")

                continue

        return new_state

    def simulate(self, states, num_steps, threshold=0.5, batch_size=256):
        """Выполняет симуляцию на указанное количество шагов"""

        for i in range(num_steps):
            print(f"Шаг симуляции {i + 1}/{num_steps}")

            history = (
                states[-self.sequence_length :]
                if len(states) >= self.sequence_length
                else states
            )

            if len(history) < self.sequence_length:
                padding = [history[0]] * (self.sequence_length - len(history))
                history = padding + history

            new_state = self.simulate_step_parallel(history, threshold, batch_size)

            states.append(new_state)

            np.save(os.path.join(OUTPUT_DIR, f"simulation_step_{i}.npy"), new_state)

            gc.collect()

        return states

    def visualize_state(self, state, output_path=None, title=None):
        """Визуализирует состояние карты классов"""
        height, width = state.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                class_id = state[y, x]
                color = class_to_color.get(class_id, (0, 0, 0))
                rgb_image[y, x] = color

        plt.figure(figsize=(12, 12))
        plt.imshow(rgb_image)
        if title:
            plt.title(title)
        plt.axis("off")

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close()
        else:
            plt.show()

    def save(self, model_path=None):
        """Сохраняет модель"""
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "infrastructure_model.h5")

        self.model.save(model_path)
        print(f"Модель сохранена в {model_path}")

    def load(self, model_path):
        """Загружает модель"""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.model_trained = True
            print(f"Модель загружена из {model_path}")
        else:
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")


def create_change_heatmap(old_state, new_state, output_path):
    """Создает тепловую карту изменений между двумя состояниями"""
    changes = (old_state != new_state).astype(np.float32)

    plt.figure(figsize=(12, 12))
    plt.imshow(changes, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Изменение")
    plt.title("Тепловая карта изменений")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def analyze_results(automaton, states, years, output_dir=OUTPUT_DIR):
    """Анализирует и визуализирует результаты симуляции"""
    os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)

    for state, year in zip(states, years):
        automaton.visualize_state(
            state,
            output_path=os.path.join(output_dir, "analysis", f"state_{year}.png"),
            title=f"Состояние инфраструктуры в {year} году",
        )

    for i in range(1, len(states)):
        old_state = states[i - 1]
        new_state = states[i]
        old_year = years[i - 1]
        new_year = years[i]

        create_change_heatmap(
            old_state,
            new_state,
            os.path.join(
                output_dir, "analysis", f"changes_{old_year}_to_{new_year}.png"
            ),
        )

    class_distributions = []
    for state in states:
        unique, counts = np.unique(state, return_counts=True)
        distribution = dict(zip(unique, counts))
        class_distributions.append(distribution)

    plt.figure(figsize=(15, 8))

    all_classes = set()
    for dist in class_distributions:
        all_classes.update(dist.keys())

    top_classes = sorted(
        all_classes,
        key=lambda c: sum(dist.get(c, 0) for dist in class_distributions),
        reverse=True,
    )[:10]

    for class_id in top_classes:
        class_counts = [dist.get(class_id, 0) for dist in class_distributions]
        plt.plot(years, class_counts, label=f"Класс {class_id}")

    plt.xlabel("Год")
    plt.ylabel("Количество пикселей")
    plt.title("Динамика распределения классов со временем")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir, "analysis", "class_distribution.png"),
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


def build_training_data(
    years_processed, all_positions, output_dir=OUTPUT_DIR, seq_length=SEQUENCE_LENGTH
):
    """Создает обучающие данные, сохраняя последовательности и целевые значения"""
    print("Создание обучающих данных...")

    sorted_years = sorted(years_processed)

    train_data_dir = os.path.join(output_dir, "train_data")
    os.makedirs(train_data_dir, exist_ok=True)

    count = 0

    for i in range(len(sorted_years) - seq_length):
        year_sequence = sorted_years[i : i + seq_length + 1]

        seq_dir = os.path.join(train_data_dir, f"seq_{i}")
        os.makedirs(seq_dir, exist_ok=True)

        for pos_idx, pos in enumerate(all_positions[year_sequence[0]]):
            y_start, x_start = pos

            sequence_valid = all(
                pos in all_positions[year] for year in year_sequence[1:]
            )

            if sequence_valid:
                sequence = []
                for year in year_sequence[:-1]:
                    patch_path = os.path.join(
                        PATCHES_DIR, str(year), f"{year}_{y_start}_{x_start}.npy"
                    )
                    patch = np.load(patch_path)
                    sequence.append(patch)

                target_year = year_sequence[-1]
                target_path = os.path.join(
                    PATCHES_DIR,
                    str(target_year),
                    f"{target_year}_{y_start}_{x_start}.npy",
                )
                target_patch = np.load(target_path)
                center_y, center_x = PATCH_SIZE // 2, PATCH_SIZE // 2
                target_class = target_patch[center_y, center_x]

                seq_filename = f"seq_{pos_idx}.npy"
                target_filename = f"target_{pos_idx}.npy"
                np.save(os.path.join(seq_dir, seq_filename), np.array(sequence))
                np.save(os.path.join(seq_dir, target_filename), target_class)

                count += 1

                if count % 1000 == 0:
                    print(f"Создано {count} последовательностей")
                    gc.collect()

    print(f"Всего создано {count} последовательностей для обучения")
    return train_data_dir


def validate_model(automaton, validation_year, historical_years, image_paths):
    """
    Валидирует модель, сравнивая предсказания с реальными данными.
    """
    print(f"Начинаем загрузку изображения для {validation_year} года...")
    start_start = time.time()

    validation_path = next(
        path
        for path in image_paths
        if extract_year(os.path.basename(path)) == validation_year
    )
    validation_img = Image.open(validation_path).convert("RGB")
    validation_array = np.array(validation_img)

    print(f"Преобразуем цвета в классы...")

    validation_map = vectorized_convert_to_class_map(validation_array)

    print(f"Преобразование завершено за {time.time() - start_start:.2f} секунд")

    start_start = time.time()
    print(f"Преобразуем historical_years...")

    historical_maps = []
    for year in tqdm(historical_years):
        history_path = next(
            path for path in image_paths if extract_year(os.path.basename(path)) == year
        )
        history_img = Image.open(history_path).convert("RGB")
        history_array = np.array(history_img)
        history_map = vectorized_convert_to_class_map(history_array)
        historical_maps.append(history_map)

    print(f"Преобразование завершено за {time.time() - start_start:.2f} секунд")
    initial_state = historical_maps[-1]
    historical_states = historical_maps[:-1]

    print(
        f"Временно используем простую копию последнего состояния вместо предсказания..."
    )

    predicted_map = initial_state.copy()

    correct_pixels = np.sum(predicted_map == validation_map)
    total_pixels = validation_map.size
    accuracy = correct_pixels / total_pixels

    print(f"Точность предсказания для {validation_year} года: {accuracy:.4f}")

    automaton.visualize_state(
        validation_map,
        output_path=os.path.join(OUTPUT_DIR, f"real_{validation_year}.png"),
        title=f"Реальное состояние {validation_year} года",
    )

    automaton.visualize_state(
        predicted_map,
        output_path=os.path.join(OUTPUT_DIR, f"predicted_{validation_year}.png"),
        title=f"Предсказанное состояние {validation_year} года",
    )

    diff_map = (predicted_map != validation_map).astype(np.int32)
    plt.figure(figsize=(12, 12))
    plt.imshow(diff_map, cmap="Reds")
    plt.title(f"Разница между предсказанием и реальностью для {validation_year} года")
    plt.colorbar(label="Ошибка")
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"diff_{validation_year}.png"),
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()

    return {
        "accuracy": accuracy,
        "correct_pixels": correct_pixels,
        "total_pixels": total_pixels,
    }


'''def validate_model(automaton, validation_year, historical_years, image_paths):
    """
    Валидирует модель, сравнивая предсказания с реальными данными.

    Args:
        automaton: Обученная модель
        validation_year: Год для валидации
        historical_years: Исторические годы
        image_paths: Пути к изображениям

    Returns:
        Метрики качества
    """
    validation_path = next(path for path in image_paths if extract_year(os.path.basename(path)) == validation_year)

    validation_img = Image.open(validation_path).convert('RGB')
    validation_array = np.array(validation_img)
    validation_map = convert_image_to_class_map(validation_array)

    historical_maps = []
    for year in tqdm(historical_years):
        history_path = next(path for path in image_paths if extract_year(os.path.basename(path)) == year)
        history_img = Image.open(history_path)
        history_array = np.array(history_img)
        history_map = convert_image_to_class_map(history_array)
        historical_maps.append(history_map)

    initial_state = historical_maps[-1]
    historical_states = historical_maps[:-1]

    predicted_map = automaton.simulate_step_parallel(initial_state, historical_states)

    correct_pixels = np.sum(predicted_map == validation_map)
    total_pixels = validation_map.size
    accuracy = correct_pixels / total_pixels

    print(f"Точность предсказания для {validation_year} года: {accuracy:.4f}")

    automaton.visualize_state(validation_map,
                              output_path=os.path.join(OUTPUT_DIR, f"real_{validation_year}.png"),
                              title=f"Реальное состояние {validation_year} года")

    automaton.visualize_state(predicted_map,
                              output_path=os.path.join(OUTPUT_DIR, f"predicted_{validation_year}.png"),
                              title=f"Предсказанное состояние {validation_year} года")

    diff_map = (predicted_map != validation_map).astype(np.int32)
    plt.figure(figsize=(12, 12))
    plt.imshow(diff_map, cmap='Reds')
    plt.title(f"Разница между предсказанием и реальностью для {validation_year} года")
    plt.colorbar(label="Ошибка")
    plt.savefig(os.path.join(OUTPUT_DIR, f"diff_{validation_year}.png"), bbox_inches='tight', dpi=150)
    plt.close()

    return {
        'accuracy': accuracy,
        'correct_pixels': correct_pixels,
        'total_pixels': total_pixels
    }
'''


def vectorized_convert_to_class_map(image):
    """Векторизованная версия преобразования в карту классов"""
    print(f"Начинаем векторизованное преобразование изображения размером {image.shape}")

    original_shape = image.shape[:2]

    pixels = image.reshape(-1, 3)
    print(f"Преобразовано в массив размером {pixels.shape}")

    colors_array = np.array(list(color_to_class.keys()))
    print(f"Подготовлен массив цветов размером {colors_array.shape}")

    batch_size = 100000
    num_batches = (pixels.shape[0] + batch_size - 1) // batch_size

    class_indices = np.zeros(pixels.shape[0], dtype=np.int32)

    print(f"Обработка по батчам: {num_batches} батчей по {batch_size} пикселей")
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, pixels.shape[0])

        current_batch = pixels[start_idx:end_idx]

        print(
            f"Обработка батча {i + 1}/{num_batches}: {current_batch.shape[0]} пикселей"
        )

        distances = np.sqrt(
            (
                (current_batch[:, np.newaxis, :] - colors_array[np.newaxis, :, :]) ** 2
            ).sum(axis=2)
        )
        closest_indices = np.argmin(distances, axis=1)

        for j, color_idx in enumerate(closest_indices):
            class_indices[start_idx + j] = color_to_class[
                tuple(colors_array[color_idx])
            ]

    return class_indices.reshape(original_shape)


def phase1_preprocess_images():
    """Фаза 1: Предварительная обработка изображений с проверкой уже обработанных"""
    print("Фаза 1: Предварительная обработка изображений...")

    all_image_paths = [
        os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith(".png")
    ]
    all_image_paths.sort(key=lambda x: extract_year(os.path.basename(x)))

    print(f"Найдено {len(all_image_paths)} изображений карт")

    processed_years = []
    if os.path.exists(os.path.join(OUTPUT_DIR, "years_processed.npy")):
        processed_years = np.load(
            os.path.join(OUTPUT_DIR, "years_processed.npy")
        ).tolist()
    else:

        for year in range(1500, 2024):
            pos_file = os.path.join(OUTPUT_DIR, f"positions_{year}.npy")
            if os.path.exists(pos_file):
                processed_years.append(year)

    if processed_years:
        print(f"Уже обработаны годы: {processed_years}")

    remaining_paths = [
        path
        for path in all_image_paths
        if extract_year(os.path.basename(path)) not in processed_years
    ]

    print(f"Осталось обработать {len(remaining_paths)} изображений")

    if not remaining_paths:
        print("Все изображения уже обработаны!")
        return processed_years, {}

    years_processed = []
    all_positions = {}

    for img_path in remaining_paths:
        gc.collect()

        year = extract_year(os.path.basename(img_path))
        print(f"Обработка изображения: {os.path.basename(img_path)} (год {year})")
        year, positions = process_image_to_patches(img_path)

        if year is not None:
            years_processed.append(year)
            all_positions[year] = positions

            np.save(os.path.join(OUTPUT_DIR, f"positions_{year}.npy"), positions)

            all_years = sorted(processed_years + [year])
            np.save(
                os.path.join(OUTPUT_DIR, "years_processed.npy"), np.array(all_years)
            )

    all_years = sorted(processed_years + years_processed)
    np.save(os.path.join(OUTPUT_DIR, "years_processed.npy"), np.array(all_years))

    print(f"Обработка изображений завершена. Всего обработано {len(all_years)} лет.")
    return all_years, all_positions


def phase2_build_training_data():
    """Фаза 2: Создание обучающих данных"""
    print("Фаза 2: Создание обучающих данных...")

    train_data_dir = os.path.join(OUTPUT_DIR, "train_data")

    if not os.path.exists(os.path.join(OUTPUT_DIR, "years_processed.npy")):
        raise FileNotFoundError(
            "Не найдены результаты предварительной обработки. Сначала выполните фазу 1."
        )

    years_processed = np.load(os.path.join(OUTPUT_DIR, "years_processed.npy"))

    all_positions = {}
    for year in years_processed:
        positions_file = os.path.join(PATCHES_DIR, str(year), "positions.npy")
        if os.path.exists(positions_file):
            all_positions[year] = np.load(positions_file)
        else:
            print(f"Предупреждение: файл позиций не найден для {year} года")

    build_training_data(years_processed, all_positions)

    print("Создание обучающих данных завершено")


def phase3_prepare_train_val_split():
    """Фаза 3: Подготовка разделения данных на обучающие и валидационные"""
    print("Фаза 3: Подготовка разделения данных на обучающие и валидационные...")

    train_dirs_file = os.path.join(OUTPUT_DIR, "train_dirs.npy")
    val_dirs_file = os.path.join(OUTPUT_DIR, "val_dirs.npy")

    if os.path.exists(train_dirs_file) and os.path.exists(val_dirs_file):
        print("Разделение уже существует. Пропускаем фазу 3.")
        return

    train_data_dir = os.path.join(OUTPUT_DIR, "train_data")
    if not os.path.exists(train_data_dir):
        raise FileNotFoundError(
            "Не найдены обучающие данные. Сначала выполните фазу 2."
        )

    train_gen = TrainingDataGenerator(train_data_dir, batch_size=BATCH_SIZE)

    val_data_dir = os.path.join(OUTPUT_DIR, "val_data")
    os.makedirs(val_data_dir, exist_ok=True)

    all_seq_dirs = train_gen.seq_dirs
    np.random.seed(42)
    np.random.shuffle(all_seq_dirs)
    val_size = int(0.2 * len(all_seq_dirs))
    val_dirs = all_seq_dirs[:val_size]
    train_dirs = all_seq_dirs[val_size:]

    np.save(train_dirs_file, train_dirs)
    np.save(val_dirs_file, val_dirs)

    print(
        f"Разделение данных создано: {len(train_dirs)} обучающих и {len(val_dirs)} валидационных директорий"
    )


def phase4_train_model():
    """Фаза 4: Обучение модели"""
    print("Фаза 4: Обучение модели...")

    model_path = os.path.join(MODELS_DIR, "infrastructure_model.h5")
    if os.path.exists(model_path):
        print(f"Модель уже обучена и сохранена в {model_path}. Пропускаем фазу 4.")
        return

    train_dirs_file = os.path.join(OUTPUT_DIR, "train_dirs.npy")
    val_dirs_file = os.path.join(OUTPUT_DIR, "val_dirs.npy")

    if not os.path.exists(train_dirs_file) or not os.path.exists(val_dirs_file):
        raise FileNotFoundError(
            "Не найдено разделение данных. Сначала выполните фазу 3."
        )

    train_dirs = np.load(train_dirs_file, allow_pickle=True)
    val_dirs = np.load(val_dirs_file, allow_pickle=True)

    train_data_dir = os.path.join(OUTPUT_DIR, "train_data")
    val_data_dir = os.path.join(OUTPUT_DIR, "val_data")

    train_gen = TrainingDataGenerator(train_data_dir, batch_size=BATCH_SIZE)
    val_gen = TrainingDataGenerator(val_data_dir, batch_size=BATCH_SIZE)

    train_gen.seq_dirs = train_dirs
    train_gen.all_sequences = []
    for seq_dir in train_dirs:
        seq_files = [
            f
            for f in os.listdir(seq_dir)
            if f.startswith("seq_") and f.endswith(".npy")
        ]
        train_gen.all_sequences.extend([(seq_dir, f) for f in seq_files])

    val_gen.seq_dirs = val_dirs
    val_gen.all_sequences = []
    for seq_dir in val_dirs:
        seq_files = [
            f
            for f in os.listdir(seq_dir)
            if f.startswith("seq_") and f.endswith(".npy")
        ]
        val_gen.all_sequences.extend([(seq_dir, f) for f in seq_files])

    print(
        f"Готово к обучению: {len(train_gen.all_sequences)} обучающих и {len(val_gen.all_sequences)} валидационных последовательностей"
    )

    automaton = InfrastructureCellularAutomaton()
    automaton.train(train_gen, val_gen, epochs=4)

    automaton.save(model_path)

    print(f"Модель обучена и сохранена в {model_path}")


def phase5_validate_model():
    """Фаза 5: Валидация модели"""
    print("Фаза 5: Валидация модели...")

    model_path = os.path.join(MODELS_DIR, "infrastructure_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Не найдена обученная модель. Сначала выполните фазу 4."
        )

    automaton = InfrastructureCellularAutomaton()
    automaton.load(model_path)

    all_image_paths = [
        os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith(".png")
    ]
    all_image_paths.sort(key=lambda x: extract_year(os.path.basename(x)))

    all_years = [extract_year(os.path.basename(path)) for path in all_image_paths]

    validation_year = all_years[-1]
    historical_years = all_years[-SEQUENCE_LENGTH - 1 : -1]

    print(
        f"Валидация модели на {validation_year} году с использованием исторических данных: {historical_years}"
    )

    metrics = validate_model(
        automaton, validation_year, historical_years, all_image_paths
    )

    np.save(os.path.join(OUTPUT_DIR, "validation_metrics.npy"), metrics)

    print(f"Валидация завершена. Точность: {metrics['accuracy']:.4f}")


def phase6_simulate_future():
    """Фаза 6: Симуляция будущего развития"""
    print("Фаза 6: Симуляция будущего развития...")

    model_path = os.path.join(MODELS_DIR, "infrastructure_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Не найдена обученная модель. Сначала выполните фазу 4."
        )

    automaton = InfrastructureCellularAutomaton()
    automaton.load(model_path)

    all_image_paths = [
        os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith(".png")
    ]
    all_image_paths.sort(key=lambda x: extract_year(os.path.basename(x)))

    all_years = [extract_year(os.path.basename(path)) for path in all_image_paths]

    all_years = [1870, 1875, 1880, 1885, 1890, 1895, 1900]

    historical_states = []
    for year in all_years[-SEQUENCE_LENGTH:]:
        hist_path = next(
            path
            for path in all_image_paths
            if extract_year(os.path.basename(path)) == year
        )
        hist_img = Image.open(hist_path).convert("RGB")
        hist_state = vectorized_convert_to_class_map(np.array(hist_img))
        historical_states.append(hist_state)

        automaton.visualize_state(
            hist_state, output_path=os.path.join(OUTPUT_DIR, f"simulation_{year}.png")
        )

    print(f"Симуляция развития на 5 лет вперед, начиная с {all_years[-1] + 5} года")

    future_states = automaton.simulate(historical_states, 1)

    print(len(future_states))

    automaton.visualize_state(
        future_states[-1],
        output_path=os.path.join(OUTPUT_DIR, f"simulation_{all_years[-1] + 5}.png"),
    )

    print("Симуляция завершена. Результаты сохранены в директории:", OUTPUT_DIR)

    historical_states.append(future_states[-1])
    analyze_results(automaton, historical_states, [1875, 1880, 1885, 1890, 1895, 1900])


def main(phase=0, start_phase=1, end_phase=6):
    if phase > 0:
        if phase == 1:
            phase1_preprocess_images()
        elif phase == 2:
            phase2_build_training_data()
        elif phase == 3:
            phase3_prepare_train_val_split()
        elif phase == 4:
            phase4_train_model()
        elif phase == 5:
            phase5_validate_model()
        elif phase == 6:
            phase6_simulate_future()
        else:
            print(f"Неизвестная фаза: {phase}")
    else:
        for phase in range(start_phase, end_phase + 1):
            print(f"\n{'=' * 20} Выполнение фазы {phase} {'=' * 20}\n")

            if phase == 1:
                phase1_preprocess_images()
            elif phase == 2:
                phase2_build_training_data()


def test_model_on_patches(
    num_patches=5, patch_size=100, sequence_length=SEQUENCE_LENGTH
):
    print(f"Тестирование модели на {num_patches} случайных фрагментах...")

    model_path = os.path.join(MODELS_DIR, "infrastructure_model.h5")
    image_paths = [
        os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith(".png")
    ]

    automaton = InfrastructureCellularAutomaton()
    automaton.load(model_path)

    sorted_images = sorted(image_paths, key=lambda x: extract_year(os.path.basename(x)))

    years = [extract_year(os.path.basename(path)) for path in sorted_images]
    print(f"Доступные годы: {years}")

    if len(years) >= sequence_length + 1:
        test_years = years[-sequence_length - 1 :]
        historical_years = test_years[:-1]
        target_year = test_years[-1]
    else:
        print(
            f"Предупреждение: недостаточно лет для последовательности ({len(years)} < {sequence_length + 1})"
        )

        if len(years) < 2:
            print("Ошибка: нужно минимум 2 года для тестирования")
            return
        historical_years = years[: min(sequence_length, len(years) - 1)]
        if len(historical_years) < sequence_length:

            historical_years = [historical_years[0]] * (
                sequence_length - len(historical_years)
            ) + historical_years
        target_year = years[-1]

    print(f"Используем годы {historical_years} для предсказания {target_year}")

    last_historical_year = historical_years[-1]
    last_historical_path = next(
        path
        for path in sorted_images
        if extract_year(os.path.basename(path)) == last_historical_year
    )
    last_img = np.array(Image.open(last_historical_path).convert("RGB"))

    height, width = last_img.shape[:2]

    test_dir = os.path.join(OUTPUT_DIR, "model_test_patches")
    os.makedirs(test_dir, exist_ok=True)

    np.random.seed(42)
    patches_info = []

    for i in range(num_patches):

        margin = patch_size + automaton.patch_size
        y = np.random.randint(margin, height - margin)
        x = np.random.randint(margin, width - margin)

        patches_info.append((y, x, patch_size))

    images_by_year = {}
    for year in historical_years + [target_year]:
        img_path = next(
            path
            for path in sorted_images
            if extract_year(os.path.basename(path)) == year
        )
        img = np.array(Image.open(img_path).convert("RGB"))

        print(f"Преобразуем изображение {year} года в карту классов...")
        class_map = vectorized_convert_to_class_map(img)
        images_by_year[year] = class_map

    for idx, (y, x, size) in enumerate(patches_info):
        print(f"Обработка фрагмента {idx + 1}/{num_patches}...")

        patch_dir = os.path.join(test_dir, f"patch_{idx + 1}")
        os.makedirs(patch_dir, exist_ok=True)

        historical_patches = []
        for year in historical_years:

            historical_patches.append(
                images_by_year[year][
                    y
                    - automaton.patch_size // 2 : y
                    + size
                    + automaton.patch_size // 2,
                    x
                    - automaton.patch_size // 2 : x
                    + size
                    + automaton.patch_size // 2,
                ]
            )

        target_patch = images_by_year[target_year][y : y + size, x : x + size]

        for i, (year, patch) in enumerate(zip(historical_years, historical_patches)):

            display_patch = patch[
                automaton.patch_size // 2 : automaton.patch_size // 2 + size,
                automaton.patch_size // 2 : automaton.patch_size // 2 + size,
            ]
            automaton.visualize_state(
                display_patch,
                output_path=os.path.join(patch_dir, f"historical_{year}.png"),
                title=f"Год {year}",
            )

        automaton.visualize_state(
            target_patch,
            output_path=os.path.join(patch_dir, f"target_{target_year}.png"),
            title=f"Целевой год {target_year}",
        )

        prediction_patch = historical_patches[-1].copy()
        center_patch = prediction_patch[
            automaton.patch_size // 2 : automaton.patch_size // 2 + size,
            automaton.patch_size // 2 : automaton.patch_size // 2 + size,
        ]

        changes_count = 0

        for py in range(automaton.patch_size // 2, size + automaton.patch_size // 2):
            for px in range(
                automaton.patch_size // 2, size + automaton.patch_size // 2
            ):

                sequence = []
                for historical_patch in historical_patches:
                    pixel_patch = historical_patch[
                        py
                        - automaton.patch_size // 2 : py
                        + automaton.patch_size // 2
                        + 1,
                        px
                        - automaton.patch_size // 2 : px
                        + automaton.patch_size // 2
                        + 1,
                    ]
                    sequence.append(pixel_patch)

                try:
                    change_prob, class_probs = automaton.predict(sequence)

                    threshold = 0.3

                    if change_prob > threshold:
                        new_class = np.argmax(class_probs) + 1
                        old_class = prediction_patch[py, px]
                        prediction_patch[py, px] = new_class
                        changes_count += 1
                except Exception as e:
                    print(f"Ошибка при предсказании для пикселя ({py}, {px}): {e}")
                    continue

        predicted_center = prediction_patch[
            automaton.patch_size // 2 : automaton.patch_size // 2 + size,
            automaton.patch_size // 2 : automaton.patch_size // 2 + size,
        ]

        automaton.visualize_state(
            predicted_center,
            output_path=os.path.join(patch_dir, f"predicted_{target_year}.png"),
            title=f"Предсказание на {target_year} год (изменено {changes_count} пикселей)",
        )

        diff_map = (predicted_center != target_patch).astype(np.int32)
        plt.figure(figsize=(10, 10))
        plt.imshow(diff_map, cmap="Reds")
        plt.title(f"Различия между предсказанием и реальностью для {target_year} года")
        plt.colorbar(label="Ошибка")
        plt.savefig(
            os.path.join(patch_dir, f"diff_{target_year}.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

        print(
            f"Фрагмент {idx + 1}: Изменено {changes_count} пикселей из {size * size} ({changes_count / (size * size) * 100:.2f}%)"
        )

        change_map = (predicted_center != center_patch).astype(np.int32)
        plt.figure(figsize=(10, 10))
        plt.imshow(change_map, cmap="hot")
        plt.title(f"Изменения от {historical_years[-1]} к предсказанию {target_year}")
        plt.colorbar(label="Изменение")
        plt.savefig(
            os.path.join(patch_dir, f"changes_to_prediction.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()


if __name__ == "__main__":
    freeze_support()
    test_model_on_patches()

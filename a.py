
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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Dropout, BatchNormalization
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

BASE_DIR = 'C:\HSE\Okit\pythonProject2'
PATCHES_DIR = 'C:\HSE\Okit\pythonProject2\output'
OUTPUT_DIR = 'C:\HSE\Okit\pythonProject2\output'
MODELS_DIR = 'C:\HSE\Okit\pythonProject2\models'
os.makedirs(PATCHES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

color_to_class = {
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
    match = re.search(r'(\d{4})\.png', filename)
    if match:
        return int(match.group(1))
    return None


def closest_color(rgb, colors):
    """Находит ближайший цвет из предопределенного набора"""
    rgb = np.array(rgb).reshape(1, -1)
    colors = np.array(list(colors)).reshape(-1, 3)
    distances = pairwise_distances(rgb, colors)
    index_of_nearest = distances.argmin()
    return tuple(colors[index_of_nearest])


def convert_image_to_class_map(image):
    """Преобразует RGB изображение в карту классов"""
    height, width, _ = image.shape
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
        # Загружаем изображение только один раз для всего батча
        img = cv2.imread(img_path)
        if img is None:
            return []

        # Конвертируем из BGR в RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Извлекаем год
        year = extract_year(os.path.basename(img_path))
        if year is None:
            return []

        results = []
        batch_patches = []
        batch_filenames = []

        for y_start, x_start in batch_coords:
            if (y_start + patch_size > img.shape[0] or
                    x_start + patch_size > img.shape[1]):
                continue

            # Вырезаем фрагмент
            patch = img[y_start:y_start + patch_size, x_start:x_start + patch_size]

            # Преобразуем в классы
            class_patch = convert_image_to_class_map(patch)

            # Сохраняем для пакетной записи
            patch_filename = f"{year}_{y_start}_{x_start}.npy"
            batch_patches.append(class_patch)
            batch_filenames.append(patch_filename)

            # Добавляем позицию в результат
            results.append((y_start, x_start))

        # Пакетная запись на диск - все фрагменты сразу
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
    patches = data['patches']
    positions = data['positions']

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
    # Входной слой для последовательности фрагментов
    input_sequence = Input(shape=(sequence_length,) + input_shape)

    # CNN для извлечения признаков из каждого фрагмента последовательности
    cnn_model = tf.keras.Sequential([
        Conv2D(16, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(4, activation='relu')
    ])

    # TimeDistributed обертка применяет CNN к каждому фрагменту последовательности
    time_distributed_cnn = TimeDistributed(cnn_model)(input_sequence)

    # LSTM для анализа временной последовательности
    lstm_out = LSTM(64, return_sequences=False)(time_distributed_cnn)
    lstm_out = Dropout(0.3)(lstm_out)

    # Слой для определения, изменится ли пиксель (RF-подобный)
    # Мы не можем напрямую использовать Random Forest в Keras,
    # но можем симулировать его с помощью полносвязного слоя
    change_probability = Dense(1, activation='sigmoid', name='change_probability')(lstm_out)

    # Слой для предсказания класса
    class_probabilities = Dense(64, activation='relu')(lstm_out)
    class_probabilities = BatchNormalization()(class_probabilities)
    class_probabilities = Dense(num_classes, activation='softmax', name='class_probabilities')(class_probabilities)

    # Создаем модель с несколькими выходами
    model = Model(inputs=input_sequence, outputs=[change_probability, class_probabilities])

    # Компилируем модель с разными функциями потерь
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'change_probability': 'binary_crossentropy',
            'class_probabilities': 'categorical_crossentropy'
        },
        metrics={
            'change_probability': 'accuracy',
            'class_probabilities': 'accuracy'
        },
        loss_weights={
            'change_probability': 0.3,  # Меньший вес для определения изменения
            'class_probabilities': 0.7  # Больший вес для предсказания класса
        }
    )

    return model

# Глобальная функция для векторизованного преобразования в карту классов
def vectorized_convert_to_class_map(patch, colors_array, color_to_class_dict):
    """Векторизованная версия преобразования в карту классов"""
    patch_flat = patch.reshape(-1, 3)
    distances = np.sqrt(np.sum((patch_flat[:, np.newaxis, :] - colors_array[np.newaxis, :, :]) ** 2, axis=2))
    closest_indices = np.argmin(distances, axis=1)
    closest_colors = colors_array[closest_indices]

    class_ids = np.zeros(len(patch_flat), dtype=np.int32)
    for i, color in enumerate(closest_colors):
        color_tuple = tuple(color)
        class_ids[i] = color_to_class_dict[color_tuple]

    return class_ids.reshape(patch.shape[0], patch.shape[1])

# Глобальная функция для обработки пакета координат
def process_chunk(args):
    """Обрабатывает и сразу сохраняет фрагменты"""
    chunk_coords, img, height, width, patch_size, year_dir, year, colors_array, color_to_class_dict = args

    saved_positions = []

    for y_start, x_start in chunk_coords:
        if y_start + patch_size > height or x_start + patch_size > width:
            continue

        # Извлекаем фрагмент
        patch = img[y_start:y_start + patch_size, x_start:x_start + patch_size]

        # Преобразуем фрагмент в карту классов (векторизовано)
        class_map = vectorized_convert_to_class_map(patch, colors_array, color_to_class_dict)

        # Сразу сохраняем каждый патч (без промежуточного NPZ)
        patch_filename = f"{year}_{y_start}_{x_start}.npy"
        patch_path = os.path.join(year_dir, patch_filename)
        np.save(patch_path, class_map)

        saved_positions.append((y_start, x_start))

    return saved_positions

# Глобальная функция для извлечения пакета
def extract_chunk(chunk_file):
    """Извлекает и сохраняет фрагменты из сжатого файла"""
    data = np.load(chunk_file)
    patches = data['patches']
    positions = data['positions']

    # Получаем директорию и год из пути к файлу
    year_dir = os.path.dirname(chunk_file)
    year = os.path.basename(year_dir)

    extracted_positions = []
    for i, (patch, pos) in enumerate(zip(patches, positions)):
        y_start, x_start = pos
        patch_filename = f"{year}_{y_start}_{x_start}.npy"
        patch_path = os.path.join(year_dir, patch_filename)
        np.save(patch_path, patch)
        extracted_positions.append(tuple(pos))

    # Удаляем временный файл после обработки
    os.remove(chunk_file)

    return extracted_positions


def process_image_to_patches(img_path, output_dir=PATCHES_DIR, patch_size=PATCH_SIZE, stride=STRIDE):
    """Оптимизированная версия функции обработки изображения"""
    # Извлекаем год из имени файла
    year = extract_year(os.path.basename(img_path))
    if year is None:
        print(f"Не удалось извлечь год из: {img_path}")
        return None, []

    # Создаем директорию для фрагментов этого года
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    try:
        # Загружаем изображение один раз в основном процессе
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")

        # Конвертируем из BGR в RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        # Вычисляем количество фрагментов
        n_h = (height - patch_size) // stride + 1
        n_w = (width - patch_size) // stride + 1
        total_patches = n_h * n_w

        print(f"Обработка изображения {year} года, размер: {width}x{height}, фрагментов: {total_patches}")

        # Выборка координат с большим шагом (увеличиваем SAMPLE_INTERVAL)
        all_coords = [(i * stride, j * stride) for i in range(n_h) for j in range(n_w)]
        sampled_coords = all_coords[::SAMPLE_INTERVAL]
        print(f"Выбрано {len(sampled_coords)} из {total_patches} фрагментов для обработки")

        # Преобразуем словарь цветов в более эффективную структуру для поиска
        colors_array = np.array(list(color_to_class.keys()))

        # Ограничиваем количество процессов
        max_workers = min(40, mp.cpu_count())
        chunk_size = len(sampled_coords) // max_workers

        # Разделяем координаты на примерно равные части
        coords_chunks = [sampled_coords[i:i + chunk_size] for i in range(0, len(sampled_coords), chunk_size)]

        # Подготовка аргументов для обработки чанков
        chunk_args = [(chunk, img, height, width, patch_size, year_dir, year, colors_array, color_to_class) for chunk in coords_chunks]

        # Запускаем параллельную обработку
        all_positions = []
        max_concurrent = min(40, mp.cpu_count())
        with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
            for positions in tqdm(executor.map(process_chunk, chunk_args),
                                  total=len(chunk_args),
                                  desc=f"Обработка {year} года"):
                all_positions.extend(positions)

        # Сохраняем позиции фрагментов
        positions_file = os.path.join(year_dir, "positions.npy")
        np.save(positions_file, np.array(all_positions))

        print(f"Сохранено {len(all_positions)} фрагментов для {year} года")
        return year, all_positions

    except Exception as e:
        print(f"Ошибка при обработке {img_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, []

# Функция для создания обучающих данных с пакетной обработкой
def build_training_sequence_batch(args):
    year_sequence, positions, seq_idx, patches_dir, output_dir = args

    sequences_created = 0
    seq_dir = os.path.join(output_dir, f"seq_{seq_idx}")
    os.makedirs(seq_dir, exist_ok=True)

    for pos_idx, pos in enumerate(positions):
        y_start, x_start = pos

        # Проверяем наличие всех файлов в последовательности
        sequence_valid = True
        for year in year_sequence:
            patch_path = os.path.join(patches_dir, str(year), f"{year}_{y_start}_{x_start}.npy")
            if not os.path.exists(patch_path):
                sequence_valid = False
                break

        if sequence_valid:
            # Загружаем все патчи последовательности сразу
            sequence = []
            for year in year_sequence[:-1]:
                patch_path = os.path.join(patches_dir, str(year), f"{year}_{y_start}_{x_start}.npy")
                patch = np.load(patch_path)
                sequence.append(patch)

            # Загружаем целевой патч
            target_year = year_sequence[-1]
            target_path = os.path.join(patches_dir, str(target_year), f"{target_year}_{y_start}_{x_start}.npy")
            target_patch = np.load(target_path)
            center_y, center_x = PATCH_SIZE // 2, PATCH_SIZE // 2
            target_class = target_patch[center_y, center_x]

            # Сохраняем в батчевом режиме
            seq_filename = f"seq_{pos_idx}.npy"
            target_filename = f"target_{pos_idx}.npy"
            np.save(os.path.join(seq_dir, seq_filename), np.array(sequence))
            np.save(os.path.join(seq_dir, target_filename), target_class)

            sequences_created += 1

    return sequences_created


def build_training_data(years_processed, all_positions, output_dir=OUTPUT_DIR, seq_length=SEQUENCE_LENGTH):
    """Создает обучающие данные, сохраняя последовательности и целевые значения с использованием пакетной обработки"""
    print("Создание обучающих данных...")

    sorted_years = sorted(years_processed)

    train_data_dir = os.path.join(output_dir, "train_data")
    os.makedirs(train_data_dir, exist_ok=True)

    # Подготовка аргументов для параллельной обработки
    process_args = []

    for i in range(len(sorted_years) - seq_length):
        year_sequence = sorted_years[i:i + seq_length + 1]
        positions = all_positions[year_sequence[0]]

        process_args.append((year_sequence, positions, i, PATCHES_DIR, train_data_dir))

    # Запускаем параллельную обработку
    total_sequences = 0

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Используем tqdm для отображения прогресса
        for seq_count in tqdm(executor.map(build_training_sequence_batch, process_args),
                              total=len(process_args),
                              desc="Создание обучающих последовательностей"):
            total_sequences += seq_count

    print(f"Всего создано {total_sequences} последовательностей для обучения")
    return train_data_dir

def to_one_hot(class_map, num_classes):
    """Преобразует карту классов в формат one-hot"""
    # Внимание: классы начинаются с 1, поэтому отнимаем 1
    return np.eye(num_classes)[class_map - 1]

# Функция для пакетного создания данных генератора
def prepare_data_batch(batch_data):
    batch_sequences, num_classes = batch_data

    X_sequences = []
    y_classes = []

    for seq_dir, seq_file in batch_sequences:
        try:
            seq_idx = seq_file.split('_')[1].split('.')[0]

            seq_path = os.path.join(seq_dir, seq_file)
            target_path = os.path.join(seq_dir, f"target_{seq_idx}.npy")

            sequence = np.load(seq_path)
            target_class = np.load(target_path)

            # Преобразуем в one-hot
            sequence_one_hot = np.array([to_one_hot(patch, num_classes) for patch in sequence])

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

    def __init__(self, train_data_dir, batch_size=BATCH_SIZE, num_classes=43):
        self.train_data_dir = train_data_dir
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.seq_dirs = [os.path.join(train_data_dir, d) for d in os.listdir(train_data_dir)
                         if os.path.isdir(os.path.join(train_data_dir, d))]

        self.all_sequences = []
        for seq_dir in self.seq_dirs:
            seq_files = [f for f in os.listdir(seq_dir) if f.startswith('seq_') and f.endswith('.npy')]
            self.all_sequences.extend([(seq_dir, f) for f in seq_files])

        print(f"Найдено {len(self.all_sequences)} последовательностей для обучения")

    def __len__(self):
        return len(self.all_sequences) // self.batch_size

    def generate(self):
        """Генератор батчей для обучения с использованием multiprocessing"""
        while True:
            np.random.shuffle(self.all_sequences)

            # Обрабатываем данные пакетами для улучшения производительности
            with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 4)) as executor:
                futures = []

                for i in range(0, len(self.all_sequences), self.batch_size):
                    batch_sequences = self.all_sequences[i:i + self.batch_size]

                    # Запускаем процессы для подготовки батча данных
                    future = executor.submit(prepare_data_batch, (batch_sequences, self.num_classes))
                    futures.append(future)

                # Получаем результаты по мере их готовности
                for future in futures:
                    X_np, y_np = future.result()

                    if X_np is not None and y_np is not None:
                        yield X_np, y_np

                        # Очистка памяти
                        del X_np, y_np
                        gc.collect()


class InfrastructureCellularAutomaton:
    """Клеточный автомат для моделирования инфраструктуры"""

    def __init__(self, num_classes=43, patch_size=PATCH_SIZE, sequence_length=SEQUENCE_LENGTH):
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.sequence_length = sequence_length

        self.input_shape = (patch_size, patch_size, num_classes)

        self.model = create_integrated_model(self.input_shape, sequence_length, num_classes)

        self.model_trained = False

    def train(self, train_generator, validation_generator=None, epochs=30):
        """Обучает модель с использованием генератора данных"""
        print("Начало обучения модели...")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(os.path.join(MODELS_DIR, 'infrastructure_model.h5'), save_best_only=True)
        ]

        def train_generator_wrapper():
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

                # Убедимся, что форма корректна перед возвратом
                X_batch_fixed = np.array(X_batch)  # (batch_size, seq_length, height, width, channels)

                # Проверим и исправим форму, если нужно
                if len(X_batch_fixed.shape) > 5:
                    X_batch_fixed = X_batch_fixed.reshape(-1, self.sequence_length, self.patch_size, self.patch_size,
                                                          self.num_classes)

                yield X_batch_fixed, {
                    'change_probability': np.array(change_labels).reshape(-1, 1),
                    'class_probabilities': y_batch
                }

        def val_generator_wrapper():
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

                    yield X_batch, {'change_probability': np.array(change_labels).reshape(-1, 1),
                                    'class_probabilities': y_batch}
            else:
                yield None, None

        train_gen = tf.data.Dataset.from_generator(
            train_generator_wrapper,
            output_types=(tf.float32, {'change_probability': tf.float32, 'class_probabilities': tf.float32}),
            output_shapes=(
                tf.TensorShape([None, self.sequence_length, self.patch_size, self.patch_size, self.num_classes]),
                {
                    'change_probability': tf.TensorShape([None, 1]),
                    'class_probabilities': tf.TensorShape([None, self.num_classes])
                }
            )
        ).prefetch(tf.data.AUTOTUNE)

        val_gen = None
        if validation_generator:
            val_gen = tf.data.Dataset.from_generator(
                val_generator_wrapper,
                output_types=(tf.float32, {'change_probability': tf.float32, 'class_probabilities': tf.float32}),
                output_shapes=(
                    tf.TensorShape([None, self.sequence_length, self.patch_size, self.patch_size, self.num_classes]),
                    {'change_probability': tf.TensorShape([None, 1]),
                     'class_probabilities': tf.TensorShape([None, self.num_classes])}
                )
            ).batch(validation_generator.batch_size)

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks
        )

        self.model_trained = True
        print("Обучение завершено!")

        return history

    def predict(self, sequence):
        """Предсказывает, изменится ли центральный пиксель и какой класс будет у него"""
        if not self.model_trained:
            raise ValueError("Модель не обучена")

        sequence_one_hot = np.array([to_one_hot(patch, self.num_classes) for patch in sequence])

        X = np.expand_dims(sequence_one_hot, axis=0)

        change_prob, class_probs = self.model.predict(X)

        return change_prob[0][0], class_probs[0]

    def simulate_region_parallel(args):
        """Параллельно обрабатывает регион изображения при симуляции"""
        automaton, region, historical_regions, threshold = args

        y_start, y_end, x_start, x_end = region
        padded_y_start, padded_y_end = y_start - automaton.patch_size // 2, y_end + automaton.patch_size // 2
        padded_x_start, padded_x_end = x_start - automaton.patch_size // 2, x_end + automaton.patch_size // 2

        # Создаем локальные копии исторических данных
        local_history = [state[padded_y_start:padded_y_end, padded_x_start:padded_x_end].copy()
                         for state in historical_regions]

        # Получаем текущее состояние региона
        current_region = historical_regions[-1][padded_y_start:padded_y_end, padded_x_start:padded_x_end].copy()

        # Создаем новое состояние
        new_region = current_region.copy()

        # Обрабатываем только центральную часть (без паддинга)
        for y in tqdm(range(automaton.patch_size // 2, y_end - y_start + automaton.patch_size // 2)):
            for x in range(automaton.patch_size // 2, x_end - x_start + automaton.patch_size // 2):
                # Получаем последовательность патчей
                sequence = []
                for state in local_history:
                    patch = state[y - automaton.patch_size // 2:y + automaton.patch_size // 2 + 1,
                            x - automaton.patch_size // 2:x + automaton.patch_size // 2 + 1]
                    sequence.append(patch)

                # Предсказываем изменение
                change_prob, class_probs = automaton.predict(sequence)

                if change_prob > threshold:
                    new_class = np.argmax(class_probs) + 1
                    new_region[y, x] = new_class

        # Возвращаем только обработанный регион без паддинга
        result_region = new_region[automaton.patch_size // 2:-automaton.patch_size // 2,
                        automaton.patch_size // 2:-automaton.patch_size // 2]

        return (y_start, x_start, result_region)

    def simulate_step_parallel(automaton, current_state, historical_states, threshold=0.5):
        """Выполняет один шаг симуляции клеточного автомата с параллельной обработкой"""
        height, width = current_state.shape
        new_state = current_state.copy()

        # Разбиваем изображение на регионы для параллельной обработки
        # Используем перекрытие, чтобы избежать краевых эффектов
        region_size = 100  # Размер региона, можно настроить

        regions = []
        for y in tqdm(range(0, height, region_size)):
            for x in range(0, width, region_size):
                y_end = min(y + region_size, height)
                x_end = min(x + region_size, width)
                regions.append((y, y_end, x, x_end))

        # Подготавливаем аргументы для параллельной обработки
        process_args = [(automaton, region, historical_states + [current_state], threshold)
                        for region in regions]

        # Запускаем параллельную обработку
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for y_start, x_start, region_result in tqdm(
                    executor.map(automaton.simulate_region_parallel, process_args),
                    total=len(regions),
                    desc="Обработка регионов"
            ):
                # Обновляем соответствующую часть нового состояния
                y_end = min(y_start + region_size, height)
                x_end = min(x_start + region_size, width)
                new_state[y_start:y_end, x_start:x_end] = region_result

        return new_state

    def simulate(self, initial_state, historical_states, num_steps, threshold=0.5):
        """Выполняет симуляцию на указанное количество шагов"""
        states = historical_states + [initial_state]
        current_state = initial_state

        for i in range(num_steps):
            print(f"Шаг симуляции {i + 1}/{num_steps}")

            history = states[-self.sequence_length:] if len(states) >= self.sequence_length else states

            if len(history) < self.sequence_length:
                padding = [history[0]] * (self.sequence_length - len(history))
                history = padding + history

            history = history[:-1]

            new_state = self.simulate_step_parallel(current_state, history, threshold)

            current_state = new_state
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
        plt.axis('off')

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()

    def save(self, model_path=None):
        """Сохраняет модель"""
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, 'infrastructure_model.h5')

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
    plt.imshow(changes, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Изменение')
    plt.title('Тепловая карта изменений')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def analyze_results(automaton, states, years, output_dir=OUTPUT_DIR):
    """Анализирует и визуализирует результаты симуляции"""
    os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)

    for state, year in zip(states, years):
        automaton.visualize_state(state,
                                  output_path=os.path.join(output_dir, "analysis", f"state_{year}.png"),
                                  title=f"Состояние инфраструктуры в {year} году")

    for i in range(1, len(states)):
        old_state = states[i - 1]
        new_state = states[i]
        old_year = years[i - 1]
        new_year = years[i]

        create_change_heatmap(old_state, new_state,
                              os.path.join(output_dir, "analysis", f"changes_{old_year}_to_{new_year}.png"))

    class_distributions = []
    for state in states:
        unique, counts = np.unique(state, return_counts=True)
        distribution = dict(zip(unique, counts))
        class_distributions.append(distribution)

    plt.figure(figsize=(15, 8))

    all_classes = set()
    for dist in class_distributions:
        all_classes.update(dist.keys())

    top_classes = sorted(all_classes, key=lambda c: sum(dist.get(c, 0) for dist in class_distributions), reverse=True)[
                  :10]

    for class_id in top_classes:
        class_counts = [dist.get(class_id, 0) for dist in class_distributions]
        plt.plot(years, class_counts, label=f'Класс {class_id}')

    plt.xlabel('Год')
    plt.ylabel('Количество пикселей')
    plt.title('Динамика распределения классов со временем')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "analysis", "class_distribution.png"), bbox_inches='tight', dpi=150)
    plt.close()


def preprocess_images(image_paths):
    """Обрабатывает все изображения, сохраняя фрагменты на диск"""
    sorted_paths = sorted(image_paths, key=lambda x: extract_year(os.path.basename(x)))

    print(f"Найдено {len(sorted_paths)} изображений карт")

    years_processed = []
    all_positions = {}

    for img_path in sorted_paths:
        gc.collect()

        print(f"Обработка изображения: {os.path.basename(img_path)}")
        year, positions = process_image_to_patches(img_path)

        if year is not None:
            years_processed.append(year)
            all_positions[year] = positions

            # Сохраняем промежуточные результаты после каждого изображения
            np.save(os.path.join(OUTPUT_DIR, "years_processed.npy"), np.array(years_processed))
            np.save(os.path.join(OUTPUT_DIR, f"positions_{year}.npy"), positions)

            # Даем время на сбор мусора и охлаждение
            time.sleep(1)

    return years_processed, all_positions


def build_training_data(years_processed, all_positions, output_dir=OUTPUT_DIR, seq_length=SEQUENCE_LENGTH):
    """Создает обучающие данные, сохраняя последовательности и целевые значения"""
    print("Создание обучающих данных...")

    sorted_years = sorted(years_processed)

    train_data_dir = os.path.join(output_dir, "train_data")
    os.makedirs(train_data_dir, exist_ok=True)

    count = 0

    for i in range(len(sorted_years) - seq_length):
        year_sequence = sorted_years[i:i + seq_length + 1]

        seq_dir = os.path.join(train_data_dir, f"seq_{i}")
        os.makedirs(seq_dir, exist_ok=True)

        for pos_idx, pos in enumerate(all_positions[year_sequence[0]]):
            y_start, x_start = pos

            sequence_valid = all(pos in all_positions[year] for year in year_sequence[1:])

            if sequence_valid:
                sequence = []
                for year in year_sequence[:-1]:
                    patch_path = os.path.join(PATCHES_DIR, str(year), f"{year}_{y_start}_{x_start}.npy")
                    patch = np.load(patch_path)
                    sequence.append(patch)

                target_year = year_sequence[-1]
                target_path = os.path.join(PATCHES_DIR, str(target_year), f"{target_year}_{y_start}_{x_start}.npy")
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

    Args:
        automaton: Обученная модель
        validation_year: Год для валидации
        historical_years: Исторические годы
        image_paths: Пути к изображениям

    Returns:
        Метрики качества
    """
    validation_path = next(path for path in image_paths if extract_year(os.path.basename(path)) == validation_year)
    validation_img = Image.open(validation_path)
    validation_array = np.array(validation_img)
    validation_map = convert_image_to_class_map(validation_array)

    historical_maps = []
    for year in historical_years:
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


def phase1_preprocess_images():
    """Фаза 1: Предварительная обработка изображений с проверкой уже обработанных"""
    print("Фаза 1: Предварительная обработка изображений...")

    all_image_paths = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith('.png')]
    all_image_paths.sort(key=lambda x: extract_year(os.path.basename(x)))

    print(f"Найдено {len(all_image_paths)} изображений карт")

    # Проверяем, какие годы уже обработаны
    processed_years = []
    if os.path.exists(os.path.join(OUTPUT_DIR, "years_processed.npy")):
        processed_years = np.load(os.path.join(OUTPUT_DIR, "years_processed.npy")).tolist()
    else:
        # Проверяем файлы positions_*.npy
        for year in range(1500, 2024):
            pos_file = os.path.join(OUTPUT_DIR, f"positions_{year}.npy")
            if os.path.exists(pos_file):
                processed_years.append(year)

    if processed_years:
        print(f"Уже обработаны годы: {processed_years}")

    # Фильтруем только необработанные изображения
    remaining_paths = [path for path in all_image_paths
                      if extract_year(os.path.basename(path)) not in processed_years]

    print(f"Осталось обработать {len(remaining_paths)} изображений")

    # Если все уже обработано, завершаем функцию
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

            # Сохраняем промежуточные результаты
            np.save(os.path.join(OUTPUT_DIR, f"positions_{year}.npy"), positions)

            # Добавляем новый год к общему списку и сразу сохраняем
            all_years = sorted(processed_years + [year])
            np.save(os.path.join(OUTPUT_DIR, "years_processed.npy"), np.array(all_years))


    # Финальное сохранение всех годов
    all_years = sorted(processed_years + years_processed)
    np.save(os.path.join(OUTPUT_DIR, "years_processed.npy"), np.array(all_years))

    print(f"Обработка изображений завершена. Всего обработано {len(all_years)} лет.")
    return all_years, all_positions


def phase2_build_training_data():
    """Фаза 2: Создание обучающих данных"""
    print("Фаза 2: Создание обучающих данных...")

    train_data_dir = os.path.join(OUTPUT_DIR, "train_data")

    if not os.path.exists(os.path.join(OUTPUT_DIR, "years_processed.npy")):
        raise FileNotFoundError("Не найдены результаты предварительной обработки. Сначала выполните фазу 1.")

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
        raise FileNotFoundError("Не найдены обучающие данные. Сначала выполните фазу 2.")

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

    print(f"Разделение данных создано: {len(train_dirs)} обучающих и {len(val_dirs)} валидационных директорий")


def phase4_train_model():
    """Фаза 4: Обучение модели"""
    print("Фаза 4: Обучение модели...")

    model_path = os.path.join(MODELS_DIR, 'infrastructure_model.h5')
    if os.path.exists(model_path):
        print(f"Модель уже обучена и сохранена в {model_path}. Пропускаем фазу 4.")
        return

    train_dirs_file = os.path.join(OUTPUT_DIR, "train_dirs.npy")
    val_dirs_file = os.path.join(OUTPUT_DIR, "val_dirs.npy")

    if not os.path.exists(train_dirs_file) or not os.path.exists(val_dirs_file):
        raise FileNotFoundError("Не найдено разделение данных. Сначала выполните фазу 3.")

    train_dirs = np.load(train_dirs_file, allow_pickle=True)
    val_dirs = np.load(val_dirs_file, allow_pickle=True)

    train_data_dir = os.path.join(OUTPUT_DIR, "train_data")
    val_data_dir = os.path.join(OUTPUT_DIR, "val_data")

    train_gen = TrainingDataGenerator(train_data_dir, batch_size=BATCH_SIZE)
    val_gen = TrainingDataGenerator(val_data_dir, batch_size=BATCH_SIZE)

    train_gen.seq_dirs = train_dirs
    train_gen.all_sequences = []
    for seq_dir in train_dirs:
        seq_files = [f for f in os.listdir(seq_dir) if f.startswith('seq_') and f.endswith('.npy')]
        train_gen.all_sequences.extend([(seq_dir, f) for f in seq_files])

    val_gen.seq_dirs = val_dirs
    val_gen.all_sequences = []
    for seq_dir in val_dirs:
        seq_files = [f for f in os.listdir(seq_dir) if f.startswith('seq_') and f.endswith('.npy')]
        val_gen.all_sequences.extend([(seq_dir, f) for f in seq_files])

    print(
        f"Готово к обучению: {len(train_gen.all_sequences)} обучающих и {len(val_gen.all_sequences)} валидационных последовательностей")

    automaton = InfrastructureCellularAutomaton()
    automaton.train(train_gen, val_gen, epochs=30)

    automaton.save(model_path)

    print(f"Модель обучена и сохранена в {model_path}")


def phase5_validate_model():
    """Фаза 5: Валидация модели"""
    print("Фаза 5: Валидация модели...")

    model_path = os.path.join(MODELS_DIR, 'infrastructure_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Не найдена обученная модель. Сначала выполните фазу 4.")

    automaton = InfrastructureCellularAutomaton()
    automaton.load(model_path)

    all_image_paths = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith('.png')]
    all_image_paths.sort(key=lambda x: extract_year(os.path.basename(x)))

    all_years = [extract_year(os.path.basename(path)) for path in all_image_paths]

    validation_year = all_years[-1]
    historical_years = all_years[-SEQUENCE_LENGTH - 1:-1]

    print(f"Валидация модели на {validation_year} году с использованием исторических данных: {historical_years}")

    metrics = validate_model(automaton, validation_year, historical_years, all_image_paths)

    np.save(os.path.join(OUTPUT_DIR, "validation_metrics.npy"), metrics)

    print(f"Валидация завершена. Точность: {metrics['accuracy']:.4f}")


def phase6_simulate_future():
    """Фаза 6: Симуляция будущего развития"""
    print("Фаза 6: Симуляция будущего развития...")

    model_path = os.path.join(MODELS_DIR, 'infrastructure_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Не найдена обученная модель. Сначала выполните фазу 4.")

    automaton = InfrastructureCellularAutomaton()
    automaton.load(model_path)

    all_image_paths = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith('.png')]
    all_image_paths.sort(key=lambda x: extract_year(os.path.basename(x)))

    all_years = [extract_year(os.path.basename(path)) for path in all_image_paths]

    last_year = all_years[-1]
    last_img_path = next(path for path in all_image_paths if extract_year(os.path.basename(path)) == last_year)
    last_img = Image.open(last_img_path)
    last_state = convert_image_to_class_map(np.array(last_img))

    historical_states = []
    for year in all_years[-SEQUENCE_LENGTH - 1:-1]:
        hist_path = next(path for path in all_image_paths if extract_year(os.path.basename(path)) == year)
        hist_img = Image.open(hist_path)
        hist_state = convert_image_to_class_map(np.array(hist_img))
        historical_states.append(hist_state)

    num_future_years = 5

    print(f"Симуляция развития на {num_future_years} лет вперед, начиная с {last_year} года")

    future_states = automaton.simulate(last_state, historical_states, num_future_years)

    simulated_years = []
    for i in range(len(future_states)):
        if i < len(historical_states) + 1:
            year = last_year - len(historical_states) + i
        else:
            year = last_year + (i - len(historical_states))

        simulated_years.append(year)

        automaton.visualize_state(future_states[i],
                                  output_path=os.path.join(OUTPUT_DIR, f"simulation_{year}.png"),
                                  title=f"Инфраструктура в {year} году")

    analyze_results(automaton, future_states, simulated_years)

    print("Симуляция завершена. Результаты сохранены в директории:", OUTPUT_DIR)


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
            elif phase == 3:
                phase3_prepare_train_val_split()
            elif phase == 4:
                phase4_train_model()
            elif phase == 5:
                phase5_validate_model()
            elif phase == 6:
                phase6_simulate_future()



def rebuild_years_processed():
    """Восстанавливает список обработанных годов из существующих файлов positions"""
    processed_years = []

    for year in range(1500, 2024):
        pos_file = os.path.join(OUTPUT_DIR, f"positions_{year}.npy")
        if os.path.exists(pos_file):
            processed_years.append(year)

    if processed_years:
        # Сортируем годы для корректной обработки
        processed_years.sort()
        np.save(os.path.join(OUTPUT_DIR, "years_processed.npy"), np.array(processed_years))
        print(f"Восстановлен список из {len(processed_years)} обработанных лет: от {min(processed_years)} до {max(processed_years)}")
    else:
        print("Не найдено данных об обработанных годах")

    return processed_years

# Восстанавливаем годы и запускаем обычную фазу 1
if __name__ == '__main__':
    freeze_support()
    main(4)

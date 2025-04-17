import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gc
import math
import time
from multiprocessing import freeze_support

import numpy as np
from scipy.spatial import KDTree
import tensorflow as tf
from PIL import Image
from sklearn.metrics import pairwise_distances
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Dropout, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import re
import argparse
import multiprocessing as mp
from tqdm import tqdm

import cv2
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

PATCH_SIZE = 25
SEQUENCE_LENGTH = 5
BATCH_SIZE = 32
MAX_WORKERS = 10
SAMPLE_INTERVAL = 5
NUM_CLASSES = 44

BASE_DIR = 'C:\HSE\Okit\diplom3'
PATCHES_DIR = 'C:\HSE\Okit\diplom3\patches'
OUTPUT_DIR = 'C:\HSE\Okit\diplom3\output'
MODELS_DIR = 'C:\HSE\Okit\diplom3\models'
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
    match = re.search(r'(\d{4})\.png', filename)
    if match:
        return int(match.group(1))
    return None


def create_kdtree():
    color_array = np.array(list(color_to_class.keys()))
    return KDTree(color_array), color_array


kdtree, color_list = create_kdtree()


def closest_colors_batch(rgb_batch):
    distances, indices_of_nearest = kdtree.query(rgb_batch)
    return indices_of_nearest


def convert_partial(image_slice):
    h, w, _ = image_slice.shape
    image_slice_flat = image_slice.reshape((-1, 3))
    indices = closest_colors_batch(image_slice_flat)
    class_map_slice = np.array([color_to_class[tuple(color_list[index])] for index in indices])
    return class_map_slice.reshape((h, w))


def convert_image_to_class_map(image):
    if len(image.shape) == 3 and image.shape[2] > 3:
        print(f"Обнаружено изображение с {image.shape[2]} каналами, использую только RGB")
        image = image[:, :, :3]

    height, width = image.shape[:2]
    class_map = np.zeros((height, width), dtype=np.int32)

    # Разделяем изображение по горизонтали
    slice_height = height // 8
    slices = [(image[start_row:start_row + slice_height], start_row) for start_row in range(0, height, slice_height)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(convert_partial, image_slice): start_row for image_slice, start_row in slices}
        for future in as_completed(futures):
            class_map_slice = future.result()
            start_row = futures[future]
            class_map[start_row:start_row + class_map_slice.shape[0], :] = class_map_slice

    return class_map

def convert_class_map_to_image(class_map):
    # Получаем размер изображения из карты классов
    height, width = class_map.shape

    # Создаём пустое изображение с тремя каналами (RGB)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Векторизованное преобразование
    for class_value, color in class_to_color.items():
        image[class_map == class_value] = color

    return image

def process_chunk(args):
    """Обрабатывает последовательности карт и сохраняет их в единый файл для каждой группы"""
    chunk_coords, imgs, height, width, patch_size, years = args

    saved_positions = []
    changed_count = 0
    
    sequences_dir = os.path.join(OUTPUT_DIR, "sequences")
    os.makedirs(sequences_dir, exist_ok=True)
    
    chunk_sequences = []
    chunk_targets = []
    chunk_metadata = []
    
    for center_y, center_x in chunk_coords:
        # Вычисляем верхний левый угол патча так, чтобы (center_y, center_x) был центром
        half_patch = patch_size // 2
        y_start = center_y - half_patch
        x_start = center_x - half_patch
        
        # Проверка границ изображения
        if (y_start < 0 or y_start + patch_size > height or 
            x_start < 0 or x_start + patch_size > width):
            continue
        
        # Проверка для всех изображений
        valid_sequence = True
        for img in imgs:
            if (y_start + patch_size > img.shape[0] or x_start + patch_size > img.shape[1]):
                valid_sequence = False
                break
        
        if not valid_sequence:
            continue
            
        # Извлекаем последовательность патчей
        sequence = []
        for i in range(len(years) - 1):  # Все кроме последней карты
            patch = imgs[i][y_start:y_start + patch_size, x_start:x_start + patch_size]
            sequence.append(patch)
        
        # Получаем целевое значение и проверяем изменение
        target_value = imgs[-1][center_y, center_x]
        initial_value = imgs[0][center_y, center_x]
        
        # Определяем, изменился ли пиксель
        if initial_value != target_value:
            changed_count += 1
        
        # Добавляем данные
        chunk_sequences.append(np.array(sequence))
        chunk_targets.append(target_value)
        chunk_metadata.append({
            'y': center_y,
            'x': center_x,
            'years': years[:-1],
            'target_year': years[-1],
            'changed': initial_value != target_value,
            'initial_value': int(initial_value),
            'target_value': int(target_value)
        })
        
        saved_positions.append((center_y, center_x))
    
    # Статистика для отслеживания
    total = len(chunk_coords)
    if total > 0:
        changed_percent = (changed_count / total) * 100
        print(f"Чанк: всего {total}, изменены {changed_count} ({changed_percent:.2f}%)")
    
    # Сохраняем результаты
    if chunk_sequences:
        chunk_id = hash(tuple(sorted([(p[0], p[1]) for p in saved_positions]))) % 10000
        chunk_filename = f"sequence_chunk_{chunk_id}.npz"
        np.savez_compressed(
            os.path.join(sequences_dir, chunk_filename),
            sequences=np.array(chunk_sequences),
            targets=np.array(chunk_targets),
            metadata=np.array(chunk_metadata, dtype=object),
        )
        print(f"Сохранен чанк {chunk_id} с {len(chunk_sequences)} последовательностями")

    return saved_positions



def process_maps_to_data(maps_paths, years, output_dir=PATCHES_DIR, patch_size=PATCH_SIZE):

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Загружаем изображение один раз в основном процессе
    imgs = [np.load(maps_path) for maps_path in maps_paths]

    print(f"Все изображения с {years[0]} года по {years[-1]}, успешно загружены")


    height, width = imgs[0].shape

    print(f"Обработка изображения {years[0]} года, размер: {width}x{height}")

    first_image = imgs[0]
    last_image = imgs[-1]

    diff = (first_image != last_image).astype(int)

    not_equal_positions = np.where(diff != 0)
    not_equal_coords = list(zip(not_equal_positions[0], not_equal_positions[1]))
    num_to_select = len(not_equal_coords)

    print(f"Выбрано {num_to_select} изменившихся клеток")

    equal_positions = np.where(diff == 0)

    selected_indices = np.random.choice(len(equal_positions[0]), size=num_to_select, replace=False)
    selected_equal_coords = list(
        zip(equal_positions[0][selected_indices], equal_positions[1][selected_indices]))


    final_coords = not_equal_coords + selected_equal_coords

    print(f"Выбрано {len(final_coords)} из {height * width} фрагментов для обработки")

    max_workers = min(40, mp.cpu_count())
    chunk_size = len(final_coords) // max_workers

    coords_chunks = [final_coords[i:i + chunk_size] for i in range(0, len(final_coords), chunk_size)]

    chunk_args = [(chunk, imgs, height, width, patch_size, years) for chunk in coords_chunks]

    # Запускаем параллельную обработку
    all_positions = []
    max_concurrent = min(40, mp.cpu_count())
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        for positions in tqdm(executor.map(process_chunk, chunk_args),
                              total=len(chunk_args),
                              desc=f"Обработка {years[0]} - {years[-1]} годов"):
            all_positions.extend(positions)

    # Сохраняем позиции фрагментов
    positions_file = os.path.join(os.path.join(data_dir, f"positdaions.npy"))
    np.save(positions_file, np.array(all_positions))

    print(f"Сохранено {len(all_positions)} фрагментов для {years[0]} года")
    return years[0], all_positions


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


def to_one_hot(class_map, num_classes):
    """Преобразует карту классов в формат one-hot"""
    return np.eye(num_classes)[class_map]


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


def vectorized_convert_to_class_map(image):
    """Векторизованная версия преобразования в карту классов"""
    print(f"Начинаем векторизованное преобразование изображения размером {image.shape}")

    # Сохраняем оригинальную форму
    original_shape = image.shape[:2]

    # Преобразуем в одномерный массив RGB-значений
    pixels = image.reshape(-1, 3)
    print(f"Преобразовано в массив размером {pixels.shape}")

    # Подготавливаем массив цветов из словаря
    colors_array = np.array(list(color_to_class.keys()))
    print(f"Подготовлен массив цветов размером {colors_array.shape}")

    # Обрабатываем по частям, чтобы избежать проблем с памятью
    batch_size = 100000  # Настройте этот параметр в зависимости от доступной памяти
    num_batches = (pixels.shape[0] + batch_size - 1) // batch_size

    class_indices = np.zeros(pixels.shape[0], dtype=np.int32)

    print(f"Обработка по батчам: {num_batches} батчей по {batch_size} пикселей")
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, pixels.shape[0])

        current_batch = pixels[start_idx:end_idx]

        print(f"Обработка батча {i + 1}/{num_batches}: {current_batch.shape[0]} пикселей")

        # Вычисляем расстояния для текущего батча
        distances = np.sqrt(((current_batch[:, np.newaxis, :] - colors_array[np.newaxis, :, :]) ** 2).sum(axis=2))
        closest_indices = np.argmin(distances, axis=1)

        # Преобразуем индексы цветов в индексы классов
        for j, color_idx in enumerate(closest_indices):
            class_indices[start_idx + j] = color_to_class[tuple(colors_array[color_idx])]

    # Возвращаем массив классов в исходной форме изображения
    return class_indices.reshape(original_shape)


def phase0_imeges_to_maps():
    """Фаза 0: Предварительная обработка изображений в матрицы"""
    print("Фаза 0: Предварительная обработка изображений в матрицы...")

    os.makedirs(os.path.join(OUTPUT_DIR, f"maps/"), exist_ok=True)

    all_image_paths = [os.path.join(OUT, f) for f in os.listdir(BASE_DIR) if f.endswith('.png')]
    all_image_paths.sort(key=lambda x: extract_year(os.path.basename(x)))

    print(f"Найдено {len(all_image_paths)} изображений карт")

    # Проверяем, какие годы уже обработаны
    processed_years = []
    if os.path.exists(os.path.join(OUTPUT_DIR, "years_processed_img.npy")):
        processed_years = np.load(os.path.join(OUTPUT_DIR, "years_processed_img.npy"), allow_pickle=True).tolist()
    else:
        # Проверяем файлы positions_*.npy
        for year in range(1680, 1950):
            pos_file = os.path.join(OUTPUT_DIR, f"maps/{year}.npy")
            if os.path.exists(pos_file):
                processed_years.append(year)

    if processed_years:
        print(f"Уже обработаны годы: {processed_years}")

    remaining_paths = [path for path in all_image_paths
                       if extract_year(os.path.basename(path)) not in processed_years]

    print(f"Осталось обработать {len(remaining_paths)} изображений")

    # Если все уже обработано, завершаем функцию
    if not remaining_paths:
        print("Все изображения уже обработаны!")
        return processed_years, {}

    for remaining_path in remaining_paths:
        gc.collect()
        year = extract_year(os.path.basename(remaining_path))

        print(f"Обработка изображений: для года {year}")

        img = cv2.imread(remaining_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {remaining_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[2] > 3:
            img = img[:, :, :3]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = convert_image_to_class_map(img)
        print(f"Обработка изображений: для года {year} завершена")

        save_path = os.path.join(OUTPUT_DIR, f"maps/{year}.npy")
        np.save(save_path, img)
        print(f"Результат сохранен в файл: {save_path}")

        processed_years.append(year)
        np.save(os.path.join(OUTPUT_DIR, "years_processed_img.npy"), np.array(processed_years))

    print(f"Обработка изображений завершена. Всего обработано {len(processed_years)} лет.")



def phase1_preprocess_images():
    """Фаза 1: Предварительная обработка изображений с проверкой уже обработанных"""
    print("Фаза 1: Предварительная обработка изображений...")

    os.makedirs(os.path.join(OUTPUT_DIR, f"data/"), exist_ok=True)

    processed_maps = []
    if os.path.exists(os.path.join(OUTPUT_DIR, "years_processed_img.npy")):
        processed_maps = np.load(os.path.join(OUTPUT_DIR, "years_processed_img.npy"), allow_pickle=True).tolist()
    if  processed_maps == []:
        print("Не найдены результаты фазы 0.")
        return

    processed_years = []
    if os.path.exists(os.path.join(OUTPUT_DIR, "years_processed.npy")):
        processed_years = np.load(os.path.join(OUTPUT_DIR, "years_processed.npy"), allow_pickle=True).tolist()
    else:
        for year in range(1680, 1950):
            pos_file = os.path.join(OUTPUT_DIR, f"data/{year}.npy")
            if os.path.exists(pos_file):
                processed_years.append(year)

    all_maps_paths = [os.path.join(OUTPUT_DIR, 'maps/', f) for f in os.listdir(os.path.join(OUTPUT_DIR, 'maps/')) if f.endswith('.npy')]

    if processed_years:
        print(f"Уже обработаны годы: {processed_years}")

    remaining_paths = [path for path in all_maps_paths
                       if extract_year(os.path.basename(path)) not in processed_years]

    print(f"Осталось обработать {len(remaining_paths)} изображений")

    if not remaining_paths:
        print("Все матрицы уже обработаны!")
        return processed_years, {}

    for i in range(len(remaining_paths) - SEQUENCE_LENGTH):
        gc.collect()

        maps_paths = remaining_paths[i:i + SEQUENCE_LENGTH + 1]
        years = [extract_year(os.path.basename(path)) for path in maps_paths]

        print(f"Обработка изображений: для годов {years[0]} целевой {years[-1]})")
        process_maps_to_data(maps_paths, years)

        processed_years.append(years[0])
        np.save(os.path.join(OUTPUT_DIR, "years_processed.npy"), np.array(processed_years))

    print(f"Обработка изображений завершена. Всего обработано {len(processed_years)} лет.")


def phase2_build_training_dataset():
    """Фаза 2: Создание объединенного набора данных для обучения и валидации"""
    print("Фаза 2: Создание объединенного датасета...")
    
    train_dataset_file = os.path.join(OUTPUT_DIR, "train_dataset.npz")
    val_dataset_file = os.path.join(OUTPUT_DIR, "val_dataset.npz")
    
    if os.path.exists(train_dataset_file) and os.path.exists(val_dataset_file):
        print("Датасеты уже существуют. Пропускаем фазу 2.")
        return train_dataset_file, val_dataset_file
    
    # Путь к директории с последовательностями
    sequences_dir = os.path.join(OUTPUT_DIR, "sequences")
    
    if not os.path.exists(sequences_dir):
        raise FileNotFoundError(f"Директория {sequences_dir} не найдена. Сначала выполните фазу 1.")
    
    # Находим все файлы с последовательностями
    chunk_files = [f for f in os.listdir(sequences_dir) if f.startswith("sequence_chunk_") and f.endswith(".npz")]
    
    if not chunk_files:
        raise FileNotFoundError(f"В директории {sequences_dir} не найдены файлы с последовательностями.")
    
    print(f"Найдено {len(chunk_files)} файлов с последовательностями")
    
    # Перемешиваем файлы и разделяем на обучающую и валидационную выборки
    np.random.seed(42)
    np.random.shuffle(chunk_files)
    
    # Используем 20% для валидации
    val_size = int(0.2 * len(chunk_files))
    val_files = chunk_files[:val_size]
    train_files = chunk_files[val_size:]
    
    print(f"Разделение: {len(train_files)} файлов для обучения, {len(val_files)} для валидации")
    
    # Функция для объединения данных из нескольких файлов
    def combine_chunks(file_list):
        all_sequences = []
        all_targets = []
        all_metadata = []
        
        for filename in tqdm(file_list, desc="Объединение чанков"):
            try:
                data = np.load(os.path.join(sequences_dir, filename), allow_pickle=True)
                
                sequences = data['sequences']
                targets = data['targets']
                metadata = data['metadata']
                
                all_sequences.append(sequences)
                all_targets.append(targets)
                all_metadata.extend(metadata)
            except Exception as e:
                print(f"Ошибка при загрузке файла {filename}: {e}")
                continue
        
        # Объединяем все в единые массивы
        combined_sequences = np.vstack(all_sequences) if all_sequences else np.array([])
        combined_targets = np.concatenate(all_targets) if all_targets else np.array([])
        
        return combined_sequences, combined_targets, all_metadata
    
    # Создаем обучающий датасет
    print("Создание обучающего датасета...")
    train_sequences, train_targets, train_metadata = combine_chunks(train_files)
    
    # Сохраняем обучающий датасет
    print(f"Сохранение обучающего датасета ({len(train_sequences)} последовательностей)...")
    np.savez_compressed(
        train_dataset_file,
        sequences=train_sequences,
        targets=train_targets,
        metadata=train_metadata
    )
    
    # Создаем валидационный датасет
    print("Создание валидационного датасета...")
    val_sequences, val_targets, val_metadata = combine_chunks(val_files)
    
    # Сохраняем валидационный датасет
    print(f"Сохранение валидационного датасета ({len(val_sequences)} последовательностей)...")
    np.savez_compressed(
        val_dataset_file,
        sequences=val_sequences,
        targets=val_targets,
        metadata=val_metadata
    )
    
    print(f"Датасеты успешно созданы и сохранены:")
    print(f"  - Обучающий: {train_dataset_file} ({len(train_sequences)} последовательностей)")
    print(f"  - Валидационный: {val_dataset_file} ({len(val_sequences)} последовательностей)")
    
    return train_dataset_file, val_dataset_file

class SequenceDataGenerator:
    """Генератор данных для работы с объединенным датасетом последовательностей"""
    
    def __init__(self, dataset_file, batch_size=32, num_classes=44, shuffle=True):
        """
        Инициализирует генератор данных
        
        Parameters:
        -----------
        dataset_file : str
            Путь к файлу с объединенным датасетом (.npz)
        batch_size : int
            Размер батча
        num_classes : int
            Количество классов для one-hot кодирования
        shuffle : bool
            Перемешивать ли данные перед выдачей батчей
        """
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        
        # Загружаем датасет
        print(f"Загрузка датасета из {dataset_file}...")
        dataset = np.load(dataset_file, allow_pickle=True)
        
        self.sequences = dataset['sequences']
        self.targets = dataset['targets']
        self.metadata = dataset['metadata'] if 'metadata' in dataset else None
        
        self.num_samples = len(self.sequences)
        print(f"Загружено {self.num_samples} последовательностей")
        
        # Индексы для перемешивания
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Количество батчей
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
    def __len__(self):
        """Возвращает количество батчей в генераторе"""
        return self.num_batches
    
    def on_epoch_end(self):
        """Вызывается в конце каждой эпохи"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        """Делает класс итерируемым"""
        self.current_batch = 0
        return self
    
    def __next__(self):
        """Возвращает следующий батч"""
        if self.current_batch >= self.num_batches:
            self.on_epoch_end()
            raise StopIteration
        
        # Индексы для текущего батча
        batch_indices = self.indices[self.current_batch * self.batch_size:
                                     (self.current_batch + 1) * self.batch_size]
        
        # Получаем данные по индексам
        batch_sequences = self.sequences[batch_indices]
        batch_targets = self.targets[batch_indices]
        
        # Преобразуем в one-hot
        batch_sequences_one_hot = np.array([
            [to_one_hot(patch, self.num_classes) for patch in sequence]
            for sequence in batch_sequences
        ])
        
        batch_targets_one_hot = np.zeros((len(batch_targets), self.num_classes))
        for i, target in enumerate(batch_targets):
            batch_targets_one_hot[i, target] = 1
        
        # Создаем метки для изменений
        change_labels = []
        for i, sequence in enumerate(batch_sequences):
            target_class = batch_targets[i]
            last_frame = sequence[-1]
            center_y, center_x = PATCH_SIZE // 2, PATCH_SIZE // 2
            last_class = last_frame[center_y, center_x]
            
            changed = 1.0 if last_class != target_class else 0.0
            change_labels.append(changed)
        
        change_labels = np.array(change_labels).reshape(-1, 1)
        
        self.current_batch += 1
        return batch_sequences_one_hot, {
            'change_probability': change_labels,
            'class_probabilities': batch_targets_one_hot
        }
    
    def generate(self):
        """Генератор для бесконечной выдачи батчей (для model.fit)"""
        while True:
            # Перезапускаем итератор, когда он исчерпан
            try:
                for batch_sequences, batch_targets in self:
                    yield batch_sequences, batch_targets
            except StopIteration:
                self.on_epoch_end()
                self.current_batch = 0


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


def plot_training_history(history):
    """Визуализирует историю обучения модели"""
    # Создаем директорию для графиков, если её нет
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Строим график потерь
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Строим график точности для основной задачи (классификация)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['class_probabilities_accuracy'], label='Train Accuracy')
    if 'val_class_probabilities_accuracy' in history.history:
        plt.plot(history.history['val_class_probabilities_accuracy'], label='Validation Accuracy')
    plt.title('Class Prediction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=300)
    
    # Строим график точности для предсказания изменений
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['change_probability_accuracy'], label='Train Accuracy')
    if 'val_change_probability_accuracy' in history.history:
        plt.plot(history.history['val_change_probability_accuracy'], label='Validation Accuracy')
    plt.title('Change Prediction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(plots_dir, 'change_prediction_history.png'), dpi=300)
    
    plt.close('all')


def phase3_train_model():
    """Фаза 3: Обучение модели"""
    print("Фаза 3: Обучение модели...")
    
    model_path = os.path.join(MODELS_DIR, 'infrastructure_model.h5')
    if os.path.exists(model_path):
        print(f"Модель уже обучена и сохранена в {model_path}. Пропускаем фазу 3.")
        # Если нужно переобучить, раскомментируйте следующую строку
        # os.remove(model_path)
    
    # Пути к датасетам
    train_dataset_file = os.path.join(OUTPUT_DIR, "train_dataset.npz")
    val_dataset_file = os.path.join(OUTPUT_DIR, "val_dataset.npz")
    
    if not os.path.exists(train_dataset_file):
        raise FileNotFoundError(f"Не найден файл обучающего датасета: {train_dataset_file}")
    
    if not os.path.exists(val_dataset_file):
        print(f"Предупреждение: файл валидационного датасета не найден: {val_dataset_file}")
        val_dataset_file = None
    
    # Создаем генераторы данных
    train_gen = SequenceDataGenerator(train_dataset_file, batch_size=BATCH_SIZE)
    val_gen = SequenceDataGenerator(val_dataset_file, batch_size=BATCH_SIZE) if val_dataset_file else None
    
    # Создаем модель
    input_shape = (PATCH_SIZE, PATCH_SIZE, NUM_CLASSES)
    model = create_integrated_model(input_shape, SEQUENCE_LENGTH, NUM_CLASSES)
    
    # Выводим информацию о модели
    model.summary()
    
    # Настраиваем обратные вызовы
    callbacks = [
        EarlyStopping(monitor='val_loss' if val_gen else 'loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True),
        # Сохранение модели после каждой эпохи
        ModelCheckpoint(
            os.path.join(MODELS_DIR, 'infrastructure_model_epoch_{epoch:02d}.h5'), 
            save_freq='epoch'
        ),
        # Очистка памяти каждые 50 батчей
        tf.keras.callbacks.LambdaCallback(
            on_batch_end=lambda batch, logs: gc.collect() if batch % 50 == 0 else None
        )
    ]
    
    # Определяем количество шагов для каждой эпохи
    steps_per_epoch = min(2000, len(train_gen))
    validation_steps = min(500, len(val_gen)) if val_gen else None
    
    print(f"Шагов на эпоху: {steps_per_epoch}, валидационных шагов: {validation_steps}")
    
    # Обучаем модель
    history = model.fit(
        train_gen.generate(),
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen.generate() if val_gen else None,
        validation_steps=validation_steps,
        epochs=30,  # Можно настроить
        callbacks=callbacks,
        verbose=1
    )
    
    # Сохраняем историю обучения
    history_file = os.path.join(OUTPUT_DIR, 'training_history.npy')
    np.save(history_file, history.history)
    
    # Визуализируем результаты обучения
    plot_training_history(history)
    
    print(f"Модель обучена и сохранена в {model_path}")
    return model, history

def main(phase=0, start_phase=1, end_phase=6, use_separated_model=True):
    if phase == 0:
        phase0_imeges_to_maps()
    elif phase == 1:
        phase1_preprocess_images()
    elif phase == 2:
        phase2_build_training_dataset()
    elif phase == 3:
        phase3_train_model()

    else:
        print(f"Неизвестная фаза: {phase}")

    print("\nВыполнение завершено!")


if __name__ == '__main__':
    freeze_support()
    main(3, use_separated_model=True)

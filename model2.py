import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gc
import math
import time
from multiprocessing import freeze_support

import numpy as np
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
from concurrent.futures import ProcessPoolExecutor

PATCH_SIZE = 25
SEQUENCE_LENGTH = 5
BATCH_SIZE = 32
MAX_WORKERS = 10
STRIDE = 15
SAMPLE_INTERVAL = 5

BASE_DIR = 'C:\HSE\Okit\pythonProject2'
PATCHES_DIR = 'C:\HSE\Okit\pythonProject2\patches2'
OUTPUT_DIR = 'C:\HSE\Okit\pythonProject2\output3\\v2'
MODELS_DIR = 'C:\HSE\Okit\pythonProject2\models2'
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
        print(f"Обнаружено изображение с {image.shape[2]} каналами, использую только RGB")
        image = image[:, :, :3]

    height, width = image.shape[:2]
    class_map = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            pixel_rgb = tuple(image[i, j])
            closest = closest_color(pixel_rgb, color_to_class)
            class_map[i, j] = color_to_class[closest]

    return class_map


# Глобальная функция для обработки пакета координат
def process_chunk(args):
    """Обрабатывает и сразу сохраняет фрагменты"""
    chunk_coords, img, height, width, patch_size, year_dir, year = args

    saved_positions = []

    for y_start, x_start in chunk_coords:
        if y_start + patch_size > height or x_start + patch_size > width:
            continue

        # Извлекаем фрагмент
        patch = img[y_start:y_start + patch_size, x_start:x_start + patch_size]

        # Преобразуем фрагмент в карту классов (векторизовано)
        class_map = vectorized_convert_to_class_map(patch)

        # Сразу сохраняем каждый патч (без промежуточного NPZ)
        patch_filename = f"{year}_{y_start}_{x_start}.npy"
        patch_path = os.path.join(year_dir, patch_filename)
        np.save(patch_path, class_map)

        saved_positions.append((y_start, x_start))

    return saved_positions


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

        # Проверяем и обрезаем, если изображение имеет альфа-канал
        if img.shape[2] > 3:
            print(f"Обнаружено изображение с {img.shape[2]} каналами, использую только RGB")
            img = img[:, :, :3]

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

        # Ограничиваем количество процессов
        max_workers = min(40, mp.cpu_count())
        chunk_size = len(sampled_coords) // max_workers

        # Разделяем координаты на примерно равные части
        coords_chunks = [sampled_coords[i:i + chunk_size] for i in range(0, len(sampled_coords), chunk_size)]

        # Подготовка аргументов для обработки чанков
        chunk_args = [(chunk, img, height, width, patch_size, year_dir, year) for chunk in coords_chunks]

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


def phase1_preprocess_images():
    """Фаза 1: Предварительная обработка изображений с проверкой уже обработанных"""
    print("Фаза 1: Предварительная обработка изображений...")

    all_image_paths = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith('.png')]
    all_image_paths.sort(key=lambda x: extract_year(os.path.basename(x)))

    print(f"Найдено {len(all_image_paths)} изображений карт")

    # Проверяем, какие годы уже обработаны
    processed_years = []
    if os.path.exists(os.path.join(OUTPUT_DIR, "years_processed.npy")):
        processed_years = np.load(os.path.join(OUTPUT_DIR, "years_processed.npy"), allow_pickle=True).tolist()
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
    """Фаза 2: Создание обучающих данных с анализом изменений классов"""
    print("Фаза 2: Создание обучающих данных и анализ изменений...")

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

    # Счетчики для статистики
    total_sequences = 0
    changed_sequences = 0
    class_changes = {}  # Словарь для отслеживания изменений между классами
    class_frequencies = {}  # Частота встречаемости каждого класса

    # Создаем директорию для анализа
    analysis_dir = os.path.join(OUTPUT_DIR, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    print("Анализ изменений классов в последовательностях...")

    sorted_years = sorted(years_processed)

    # Строим обучающие данные и собираем статистику
    for i in range(len(sorted_years) - SEQUENCE_LENGTH):
        year_sequence = sorted_years[i:i + SEQUENCE_LENGTH + 1]
        seq_idx = i

        print(f"Обработка последовательности лет {year_sequence}")

        # Позиции из первого года в последовательности
        positions = all_positions[year_sequence[0]]

        seq_dir = os.path.join(train_data_dir, f"seq_{seq_idx}")
        os.makedirs(seq_dir, exist_ok=True)

        # Счетчики для текущей последовательности лет
        sequence_total = 0
        sequence_changed = 0

        for pos_idx, pos in enumerate(tqdm(positions, desc=f"Обработка патчей для {year_sequence}")):
            y_start, x_start = pos

            # Проверяем наличие всех файлов в последовательности
            sequence_valid = True
            for year in year_sequence:
                patch_path = os.path.join(PATCHES_DIR, str(year), f"{year}_{y_start}_{x_start}.npy")
                if not os.path.exists(patch_path):
                    sequence_valid = False
                    break

            if not sequence_valid:
                continue

            # Последовательность валидна, увеличиваем счетчик
            total_sequences += 1
            sequence_total += 1

            # Загружаем начальный и конечный патч
            first_year = year_sequence[0]
            last_year = year_sequence[-1]

            first_patch_path = os.path.join(PATCHES_DIR, str(first_year), f"{first_year}_{y_start}_{x_start}.npy")
            last_patch_path = os.path.join(PATCHES_DIR, str(last_year), f"{last_year}_{y_start}_{x_start}.npy")

            first_patch = np.load(first_patch_path)
            last_patch = np.load(last_patch_path)

            # Определяем центральный пиксель
            center_y, center_x = PATCH_SIZE // 2, PATCH_SIZE // 2
            first_class = first_patch[center_y, center_x]
            last_class = last_patch[center_y, center_x]

            # Обновляем частоту встречаемости классов
            class_frequencies[first_class] = class_frequencies.get(first_class, 0) + 1
            class_frequencies[last_class] = class_frequencies.get(last_class, 0) + 1

            # Проверяем, изменился ли класс
            if first_class != last_class:
                changed_sequences += 1
                sequence_changed += 1

                # Отслеживаем изменения между конкретными классами
                class_pair = (first_class, last_class)
                class_changes[class_pair] = class_changes.get(class_pair, 0) + 1

            # Загружаем всю последовательность патчей и сохраняем
            sequence = []
            for year in year_sequence[:-1]:
                patch_path = os.path.join(PATCHES_DIR, str(year), f"{year}_{y_start}_{x_start}.npy")
                patch = np.load(patch_path)
                sequence.append(patch)

            target_year = year_sequence[-1]
            target_path = os.path.join(PATCHES_DIR, str(target_year), f"{target_year}_{y_start}_{x_start}.npy")
            target_patch = np.load(target_path)
            target_class = target_patch[center_y, center_x]

            seq_filename = f"seq_{pos_idx}.npy"
            target_filename = f"target_{pos_idx}.npy"
            change_filename = f"change_{pos_idx}.npy"  # Добавляем информацию об изменении

            np.save(os.path.join(seq_dir, seq_filename), np.array(sequence))
            np.save(os.path.join(seq_dir, target_filename), target_class)
            # Сохраняем информацию, изменился ли класс (1 - да, 0 - нет)
            np.save(os.path.join(seq_dir, change_filename), 1 if first_class != last_class else 0)

            # Очищаем память каждые 1000 патчей
            if pos_idx % 1000 == 0:
                gc.collect()

        # Выводим статистику для текущей последовательности лет
        if sequence_total > 0:
            change_percent = (sequence_changed / sequence_total) * 100
            print(
                f"Последовательность лет {year_sequence}: найдено {sequence_changed} изменений из {sequence_total} патчей ({change_percent:.2f}%)")

    # Выводим общую статистику
    if total_sequences > 0:
        change_percent = (changed_sequences / total_sequences) * 100
        print(f"\nОбщая статистика:")
        print(f"Всего последовательностей: {total_sequences}")
        print(f"Последовательностей с изменениями: {changed_sequences} ({change_percent:.2f}%)")

        # Сохраняем статистику в файл
        stats_file = os.path.join(analysis_dir, "change_statistics.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Общая статистика:\n")
            f.write(f"Всего последовательностей: {total_sequences}\n")
            f.write(f"Последовательностей с изменениями: {changed_sequences} ({change_percent:.2f}%)\n\n")

            f.write(f"Частота встречаемости классов:\n")
            for cls, count in sorted(class_frequencies.items(), key=lambda x: x[1], reverse=True):
                f.write(f"Класс {cls}: {count} ({count / sum(class_frequencies.values()) * 100:.2f}%)\n")

            f.write(f"\nНаиболее частые изменения классов:\n")
            for (cls1, cls2), count in sorted(class_changes.items(), key=lambda x: x[1], reverse=True)[:20]:
                f.write(
                    f"Класс {cls1} -> Класс {cls2}: {count} ({count / changed_sequences * 100:.2f}% от всех изменений)\n")

        print(f"Статистика сохранена в {stats_file}")

        # Создаем визуализации
        try:
            # Круговая диаграмма частоты классов
            plt.figure(figsize=(12, 8))
            classes = list(class_frequencies.keys())
            values = list(class_frequencies.values())

            # Отображаем только топ-10 классов для лучшей читаемости
            if len(classes) > 10:
                top_indices = np.argsort(values)[-10:]
                other_sum = sum(values) - sum(np.array(values)[top_indices])
                classes = [classes[i] for i in top_indices] + ['Другие']
                values = [values[i] for i in top_indices] + [other_sum]

            plt.pie(values, labels=classes, autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')
            plt.title('Распределение классов')
            plt.savefig(os.path.join(analysis_dir, "class_distribution.png"))
            plt.close()

            # Гистограмма изменений классов
            plt.figure(figsize=(15, 10))
            top_changes = sorted(class_changes.items(), key=lambda x: x[1], reverse=True)[:15]
            labels = [f"{c1}->{c2}" for (c1, c2), _ in top_changes]
            counts = [count for _, count in top_changes]

            plt.bar(labels, counts)
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Изменение класса')
            plt.ylabel('Количество')
            plt.title('Топ-15 наиболее частых изменений классов')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, "class_changes.png"))
            plt.close()

            print(f"Графики сохранены в директории {analysis_dir}")
        except Exception as e:
            print(f"Ошибка при создании визуализаций: {e}")

    print("Создание обучающих данных завершено")
    return train_data_dir


def main(phase=0, start_phase=1, end_phase=6, use_separated_model=True):
    if phase > 0:
        if phase == 1:
            phase1_preprocess_images()
        elif phase == 2:
            phase2_build_training_data()

        else:
            print(f"Неизвестная фаза: {phase}")
    else:
        # Выполняем все фазы последовательно
        for phase in range(start_phase, end_phase + 1):
            print(f"\n{'=' * 20} Выполнение фазы {phase} {'=' * 20}\n")

            if phase == 1:
                phase1_preprocess_images()
            elif phase == 2:
                phase2_build_training_data()

            gc.collect()

    print("\nВыполнение завершено!")


if __name__ == '__main__':
    freeze_support()
    main(1, use_separated_model=True)  # Использовать раздельное обучение

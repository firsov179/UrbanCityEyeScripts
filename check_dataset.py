import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
import argparse

# Константы
OUTPUT_DIR = 'C:\HSE\Okit\diplom3\output'
NUM_CLASSES = 44

# Названия классов для более информативных графиков
class_names = {
    0: "empty",
    1: "train_station",
    2: "station",
    3: "train_yard",
    4: "transportation",
    5: "light_rail",
    6: "monorail",
    7: "narrow_gauge",
    8: "railway_platform",
    9: "railway_rail",
    10: "railway_station",
    11: "subway",
    12: "tram",
    13: "bridleway",
    14: "cycleway",
    15: "footway",
    16: "living_street",
    17: "motorway",
    18: "motorway_link",
    19: "highway_no",
    20: "path",
    21: "pedestrian",
    22: "pedestrian_gravel",
    23: "primary",
    24: "primary_link",
    25: "residential",
    26: "secondary",
    27: "secondary_link",
    28: "service",
    29: "steps",
    30: "tertiary",
    31: "tertiary_link",
    32: "track",
    33: "trunk",
    34: "trunk_link",
    35: "unclassified",
    36: "canal",
    37: "dam",
    38: "ditch",
    39: "dock",
    40: "river",
    41: "riverbank",
    42: "stream",
    43: "parking"
}

def load_dataset(dataset_file):
    """Загружает датасет и возвращает данные"""
    print(f"Загрузка датасета из {dataset_file}...")
    try:
        data = np.load(dataset_file, allow_pickle=True)
        sequences = data['sequences']
        targets = data['targets']
        metadata = data['metadata'] if 'metadata' in data else None
        
        print(f"Загружено {len(sequences)} последовательностей")
        return sequences, targets, metadata
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        return None, None, None

def analyze_dataset(dataset_file, output_dir=None):
    """Анализирует датасет и визуализирует результаты"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(dataset_file), "analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем данные
    sequences, targets, metadata = load_dataset(dataset_file)
    if sequences is None:
        return
    
    # 1. Распределение целевых классов
    analyze_target_distribution(targets, output_dir)
    
    # 2. Анализ изменений целевого пикселя
    analyze_pixel_changes(sequences, targets, output_dir)
    
    # 3. Анализ последовательностей переходов
    analyze_transition_sequences(sequences, targets, output_dir)
    
    # 4. Геопространственный анализ (если есть координаты в метаданных)
    if metadata is not None and len(metadata) > 0:
        if 'x' in metadata[0] and 'y' in metadata[0]:
            analyze_spatial_distribution(metadata, targets, output_dir)
    
    # 5. Анализ временных рядов (если есть годы в метаданных)
    #if metadata is not None and len(metadata) > 0:
        #if 'years' in metadata[0]:
            #analyze_temporal_patterns(metadata, targets, output_dir)
    
    print(f"Анализ завершен. Результаты сохранены в {output_dir}")

def analyze_target_distribution(targets, output_dir):
    """Анализирует распределение целевых классов"""
    print("Анализ распределения целевых классов...")
    
    # Подсчитываем количество каждого класса
    class_counts = Counter(targets)
    
    # Сортируем классы по частоте
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_classes)
    
    # Создаем метки с названиями классов
    class_labels = [f"{class_names.get(c, f'Class {c}')} ({c})" for c in classes]
    
    # Построение графика распределения классов
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), class_labels, rotation=90)
    plt.title('Распределение целевых классов')
    plt.ylabel('Количество образцов')
    plt.tight_layout()
    
    # Добавляем значения над столбцами
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom', rotation=0)
    
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=300)
    plt.close()
    
    # Построение круговой диаграммы для топ-10 классов
    plt.figure(figsize=(12, 10))
    top_n = 10
    if len(classes) > top_n:
        # Берем топ-N классов и объединяем остальные в "Другие"
        top_classes = classes[:top_n]
        top_counts = counts[:top_n]
        other_count = sum(counts[top_n:])
        
        pie_classes = list(top_classes) + ["Другие"]
        pie_counts = list(top_counts) + [other_count]
        pie_labels = [f"{class_names.get(c, f'Class {c}')} ({c}): {count}" for c, count in zip(top_classes, top_counts)]
        pie_labels.append(f"Другие: {other_count}")
    else:
        pie_classes = classes
        pie_counts = counts
        pie_labels = [f"{class_names.get(c, f'Class {c}')} ({c}): {count}" for c, count in zip(classes, counts)]
    
    plt.pie(pie_counts, labels=pie_labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title(f'Топ-{top_n} наиболее частых целевых классов')
    plt.savefig(os.path.join(output_dir, 'target_distribution_pie.png'), dpi=300)
    plt.close()
    
    # Сохраняем статистику в текстовый файл
    with open(os.path.join(output_dir, 'class_distribution.txt'), 'w') as f:
        f.write("Распределение целевых классов:\n")
        f.write("=" * 50 + "\n")
        f.write("Класс | Название | Количество | Процент\n")
        f.write("-" * 50 + "\n")
        
        total = sum(counts)
        for c, count in sorted_classes:
            percentage = count / total * 100
            f.write(f"{c} | {class_names.get(c, 'Unknown')} | {count} | {percentage:.2f}%\n")
        
        f.write("=" * 50 + "\n")
        f.write(f"Всего: {total}\n")

def analyze_pixel_changes(sequences, targets, output_dir):
    """Анализирует изменения целевого пикселя от последнего кадра к целевому значению"""
    print("Анализ изменений пикселей...")
    
    changes = []
    last_classes = []
    for i, seq in enumerate(tqdm(sequences, desc="Анализ изменений")):
        # Получаем класс центрального пикселя на последнем кадре
        center_y, center_x = seq.shape[1] // 2, seq.shape[2] // 2
        last_class = seq[0, center_y, center_x]
        target_class = targets[i]
        
        last_classes.append(last_class)
        changes.append(last_class != target_class)
    
    # Процент изменений
    change_percentage = np.mean(changes) * 100
    
    # Построение гистограммы изменений
    plt.figure(figsize=(10, 6))
    plt.bar(['Без изменений', 'С изменениями'], 
            [100 - change_percentage, change_percentage],
            color=['green', 'red'])
    plt.title('Процент изменений целевого пикселя')
    plt.ylabel('Процент образцов (%)')
    
    # Добавляем значения над столбцами
    plt.text(0, 100 - change_percentage + 1, f'{100 - change_percentage:.2f}%', 
             ha='center', va='bottom')
    plt.text(1, change_percentage + 1, f'{change_percentage:.2f}%', 
             ha='center', va='bottom')
    
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, 'pixel_changes.png'), dpi=300)
    plt.close()
    
    # Анализ матрицы переходов между классами
    transition_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    
    for last_class, target_class in zip(last_classes, targets):
        transition_matrix[last_class, target_class] += 1
    
    # Нормализация по строкам для получения вероятностей переходов
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    # Избегаем деления на ноль
    row_sums[row_sums == 0] = 1
    transition_probs = transition_matrix / row_sums
    
    # Находим наиболее частые переходы
    total_transitions = np.sum(transition_matrix)
    transitions_list = []
    
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if transition_matrix[i, j] > 0:
                from_class = class_names.get(i, f"Class {i}")
                to_class = class_names.get(j, f"Class {j}")
                count = transition_matrix[i, j]
                percentage = (count / total_transitions) * 100
                probability = transition_probs[i, j] * 100  # вероятность перехода из класса i в j
                
                transitions_list.append({
                    'from_class_id': i,
                    'to_class_id': j,
                    'from_class': from_class,
                    'to_class': to_class,
                    'count': count,
                    'percentage': percentage,
                    'probability': probability
                })
    
    # Сортируем по количеству
    transitions_list.sort(key=lambda x: x['count'], reverse=True)
    
    # Выводим топ-20 переходов
    with open(os.path.join(output_dir, 'class_transitions.txt'), 'w') as f:
        f.write("Наиболее частые переходы между классами:\n")
        f.write("=" * 100 + "\n")
        f.write("Ранг | Из класса | В класс | Количество | % от всех | Вероятность перехода\n")
        f.write("-" * 100 + "\n")
        
        for i, transition in enumerate(transitions_list[:50]):  # Топ-50
            f.write(f"{i+1} | {transition['from_class']} ({transition['from_class_id']}) | "
                   f"{transition['to_class']} ({transition['to_class_id']}) | "
                   f"{transition['count']} | {transition['percentage']:.2f}% | "
                   f"{transition['probability']:.2f}%\n")
    
    # Визуализация матрицы переходов (тепловая карта)
    plt.figure(figsize=(14, 12))
    mask = transition_matrix == 0  # Маска для скрытия нулевых переходов
    
    # Используем логарифмический масштаб для лучшей визуализации
    log_matrix = np.log1p(transition_matrix)  # log(1+x) чтобы избежать log(0)
    
    sns.heatmap(log_matrix, mask=mask, cmap='viridis', 
                xticklabels=range(NUM_CLASSES), 
                yticklabels=range(NUM_CLASSES))
    plt.title('Матрица переходов между классами (логарифмический масштаб)')
    plt.xlabel('Целевой класс')
    plt.ylabel('Исходный класс')
    plt.savefig(os.path.join(output_dir, 'transition_matrix.png'), dpi=300)
    plt.close()
    
    # Визуализация топ переходов
    top_n = 20
    top_transitions = transitions_list[:top_n]
    
    plt.figure(figsize=(15, 10))
    
    labels = [f"{t['from_class_id']}->{t['to_class_id']}" for t in top_transitions]
    values = [t['count'] for t in top_transitions]
    
    bars = plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=90)
    plt.title(f'Топ-{top_n} наиболее частых переходов между классами')
    plt.ylabel('Количество переходов')
    plt.tight_layout()
    
    # Добавляем значения над столбцами
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom', rotation=0)
    
    plt.savefig(os.path.join(output_dir, 'top_transitions.png'), dpi=300)
    plt.close()

def analyze_transition_sequences(sequences, targets, output_dir):
    """Анализирует последовательности переходов (паттерны изменений во времени)"""
    print("Анализ последовательностей переходов...")
    
    # Создаем массив последовательностей центральных пикселей
    center_sequences = []
    for seq in tqdm(sequences, desc="Извлечение центральных пикселей"):
        center_y, center_x = seq.shape[1] // 2, seq.shape[2] // 2
        center_pixels = [frame[center_y, center_x] for frame in seq]
        center_sequences.append(center_pixels)
    
    # Преобразуем в массив numpy
    center_sequences = np.array(center_sequences)
    
    # Анализ стабильности последовательностей
    # Последовательность стабильна, если все центральные пиксели имеют один и тот же класс
    stability = np.all(center_sequences == center_sequences[:, 0:1], axis=1)
    stability_percentage = np.mean(stability) * 100
    
    # Визуализация стабильности
    plt.figure(figsize=(10, 6))
    plt.bar(['Нестабильные', 'Стабильные'], 
            [100 - stability_percentage, stability_percentage],
            color=['orange', 'blue'])
    plt.title('Стабильность последовательностей центральных пикселей')
    plt.ylabel('Процент образцов (%)')
    
    # Добавляем значения над столбцами
    plt.text(0, 100 - stability_percentage + 1, f'{100 - stability_percentage:.2f}%', 
             ha='center', va='bottom')
    plt.text(1, stability_percentage + 1, f'{stability_percentage:.2f}%', 
             ha='center', va='bottom')
    
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, 'sequence_stability.png'), dpi=300)
    plt.close()
    
    # Анализ тенденций изменений
    # Для каждой последовательности определим тренд изменений
    sequence_length = center_sequences.shape[1]
    trend_data = []
    
    for i, seq in enumerate(center_sequences):
        changes = [seq[j] != seq[j+1] for j in range(sequence_length-1)]
        change_count = sum(changes)
        
        initial_class = seq[0]
        final_class = seq[-1]
        target_class = targets[i]
        
        # Общая характеристика изменений в последовательности
        if change_count == 0:
            trend = "Стабильная"
        elif change_count == 1:
            trend = "Одно изменение"
        elif change_count == 2:
            trend = "Два изменения"
        else:
            trend = "Множественные изменения"
        
        # Направление изменения
        if final_class == initial_class:
            direction = "Без изменений"
        else:
            direction = f"{initial_class} -> {final_class}"
        
        # Предсказуемость
        predictable = (final_class == target_class)
        
        trend_data.append({
            'initial_class': initial_class,
            'final_class': final_class,
            'target_class': target_class,
            'change_count': change_count,
            'trend': trend,
            'direction': direction,
            'predictable': predictable
        })
    
    # Преобразуем в DataFrame для анализа
    trend_df = pd.DataFrame(trend_data)
    
    # Визуализация распределения трендов
    trend_counts = trend_df['trend'].value_counts()
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(trend_counts.index, trend_counts.values)
    plt.title('Распределение типов изменений в последовательностях')
    plt.ylabel('Количество последовательностей')
    plt.xticks(rotation=45)
    
    # Добавляем значения над столбцами
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sequence_trends.png'), dpi=300)
    plt.close()
    
    # Предсказуемость изменений
    predictability = trend_df['predictable'].mean() * 100
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Непредсказуемые', 'Предсказуемые'], 
            [100 - predictability, predictability],
            color=['red', 'green'])
    plt.title('Предсказуемость последовательностей (финальный класс == целевой класс)')
    plt.ylabel('Процент образцов (%)')
    
    # Добавляем значения над столбцами
    plt.text(0, 100 - predictability + 1, f'{100 - predictability:.2f}%', 
             ha='center', va='bottom')
    plt.text(1, predictability + 1, f'{predictability:.2f}%', 
             ha='center', va='bottom')
    
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, 'sequence_predictability.png'), dpi=300)
    plt.close()
    
    # Сохраняем статистику в текстовый файл
    with open(os.path.join(output_dir, 'sequence_analysis.txt'), 'w') as f:
        f.write("Анализ последовательностей изменений:\n")
        f.write("=" * 60 + "\n")
        f.write(f"Общее количество последовательностей: {len(center_sequences)}\n")
        f.write(f"Стабильные последовательности: {stability.sum()} ({stability_percentage:.2f}%)\n")
        f.write(f"Нестабильные последовательности: {len(stability) - stability.sum()} ({100-stability_percentage:.2f}%)\n")
        f.write("\n")
        
        f.write("Распределение типов изменений:\n")
        for trend, count in trend_counts.items():
            percentage = (count / len(trend_df)) * 100
            f.write(f"- {trend}: {count} ({percentage:.2f}%)\n")
        
        f.write("\n")
        f.write(f"Предсказуемость (финальный класс совпадает с целевым): {predictability:.2f}%\n")
        
        # Анализ для непредсказуемых последовательностей
        unpredictable = trend_df[~trend_df['predictable']]
        unpredictable_count = len(unpredictable)
        if unpredictable_count > 0:
            f.write("\n")
            f.write(f"Анализ непредсказуемых последовательностей ({unpredictable_count}):\n")
            
            # Распределение по типам изменений
            unpred_trends = unpredictable['trend'].value_counts()
            for trend, count in unpred_trends.items():
                percentage = (count / unpredictable_count) * 100
                f.write(f"- {trend}: {count} ({percentage:.2f}%)\n")
            
            # Топ переходов для непредсказуемых
            unpred_transitions = unpredictable.groupby(['final_class', 'target_class']).size()
            unpred_transitions = unpred_transitions.reset_index().rename(columns={0: 'count'})
            unpred_transitions = unpred_transitions.sort_values('count', ascending=False)
            
            f.write("\n")
            f.write("Топ-10 непредсказуемых переходов (финальный -> целевой):\n")
            for i, row in unpred_transitions.head(10).iterrows():
                final_class = row['final_class']
                target_class = row['target_class']
                count = row['count']
                percentage = (count / unpredictable_count) * 100
                
                f.write(f"- {class_names.get(final_class, f'Class {final_class}')} ({final_class}) -> "
                        f"{class_names.get(target_class, f'Class {target_class}')} ({target_class}): "
                        f"{count} ({percentage:.2f}%)\n")

def analyze_spatial_distribution(metadata, targets, output_dir):
    """Анализирует пространственное распределение классов"""
    print("Анализ пространственного распределения...")
    
    # Извлекаем координаты
    coordinates = []
    for meta in metadata:
        if 'x' in meta and 'y' in meta:
            coordinates.append((meta['x'], meta['y']))
    
    if not coordinates:
        print("Координаты не найдены в метаданных. Пропускаем пространственный анализ.")
        return
    
    coordinates = np.array(coordinates)
    
    # Строим диаграмму рассеяния с цветовой кодировкой по классам
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=targets, cmap='tab20', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Класс')
    plt.title('Пространственное распределение классов')
    plt.xlabel('Координата X')
    plt.ylabel('Координата Y')
    plt.savefig(os.path.join(output_dir, 'spatial_distribution.png'), dpi=300)
    plt.close()
    
    # Анализ пространственных кластеров для топ-5 классов
    class_counts = Counter(targets)
    top_classes = [c for c, _ in class_counts.most_common(5)]
    
    plt.figure(figsize=(15, 12))
    
    for i, class_id in enumerate(top_classes):
        class_coords = coordinates[targets == class_id]
        
        plt.subplot(2, 3, i+1)
        plt.scatter(class_coords[:, 0], class_coords[:, 1], s=10, alpha=0.7)
        plt.title(f'Класс {class_id}: {class_names.get(class_id, "Неизвестный")}')
        plt.xlabel('X')
        plt.ylabel('Y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_clusters.png'), dpi=300)
    plt.close()

def analyze_temporal_patterns(metadata, targets, output_dir):
    """Анализирует временные паттерны изменений"""
    print("Анализ временных паттернов...")
    
    # Извлекаем годы
    year_sequences = []
    for meta in metadata:
        if 'years' in meta:
            year_sequences.append(meta['years'])
    
    if not year_sequences:
        print("Информация о годах не найдена в метаданных. Пропускаем временной анализ.")
        return
    
    # Анализ охваченных периодов
    all_years = []
    for years in year_sequences:
        all_years.extend(years)
    
    unique_years = sorted(set(all_years))
    
    # Распределение по годам
    year_counts = Counter(all_years)
    sorted_years = sorted(year_counts.items())
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar([str(y) for y, _ in sorted_years], [c for _, c in sorted_years])
    plt.title('Распределение образцов по годам')
    plt.xlabel('Год')
    plt.ylabel('Количество образцов')
    plt.xticks(rotation=90)
    
    # Добавляем значения над столбцами
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'year_distribution.png'), dpi=300)
    plt.close()
    
    # Анализ типичных временных интервалов в последовательностях
    intervals = []
    for years in year_sequences:
        for i in range(len(years)-1):
            interval = years[i+1] - years[i]
            intervals.append(interval)
    
    interval_counts = Counter(intervals)
    sorted_intervals = sorted(interval_counts.items())
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar([str(i) for i, _ in sorted_intervals], [c for _, c in sorted_intervals])
    plt.title('Распределение временных интервалов между последовательными годами')
    plt.xlabel('Интервал (лет)')
    plt.ylabel('Количество')
    
    # Добавляем значения над столбцами
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'interval_distribution.png'), dpi=300)
    plt.close()
    
    # Анализ изменений по временным периодам
    if len(year_sequences[0]) == len(year_sequences[1]):  # Проверяем, что все последовательности одинаковой длины
        sequence_length = len(year_sequences[0])
        
        # Для каждого шага в последовательности анализируем распределение классов
        center_sequences = []
        for i, seq in enumerate(tqdm(sequences, desc="Извлечение центральных пикселей")):
            center_y, center_x = seq.shape[1] // 2, seq.shape[2] // 2
            center_pixels = [frame[center_y, center_x] for frame in seq]
            center_sequences.append(center_pixels)
        
        center_sequences = np.array(center_sequences)
        
        # Для каждого временного шага анализируем распределение классов
        plt.figure(figsize=(15, 10))
        
        for step in range(sequence_length):
            step_classes = center_sequences[:, step]
            class_counts = Counter(step_classes)
            
            # Берем только 5 самых частых классов для ясности
            top_classes = class_counts.most_common(5)
            
            plt.subplot(2, 3, step+1)
            
            labels = [f"{class_names.get(c, f'Class {c}')} ({c})" for c, _ in top_classes]
            values = [count for _, count in top_classes]
            
            plt.bar(labels, values)
            plt.title(f'Топ-5 классов на шаге {step+1}')
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'class_distribution_over_time.png'), dpi=300)
        plt.close()
        
        # Анализ эволюции классов во времени
        class_evolution = defaultdict(list)
        
        for step in range(sequence_length):
            step_classes = center_sequences[:, step]
            class_counts = Counter(step_classes)
            total = sum(class_counts.values())
            
            for class_id in range(NUM_CLASSES):
                percentage = (class_counts.get(class_id, 0) / total) * 100
                class_evolution[class_id].append(percentage)
        
        # Выбираем топ-10 классов по среднему проценту для отображения
        avg_percentages = {class_id: np.mean(percentages) 
                          for class_id, percentages in class_evolution.items()}
        
        top_classes = sorted(avg_percentages.items(), key=lambda x: x[1], reverse=True)[:10]
        
        plt.figure(figsize=(12, 8))
        
        for class_id, _ in top_classes:
            percentages = class_evolution[class_id]
            plt.plot(range(1, sequence_length+1), percentages, 
                     marker='o', label=f"{class_names.get(class_id, f'Class {class_id}')} ({class_id})")
        
        plt.title('Эволюция топ-10 классов во времени')
        plt.xlabel('Шаг последовательности')
        plt.ylabel('Процент образцов (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'class_evolution.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Анализ датасета последовательностей для обучения модели')
    parser.add_argument('--dataset', type=str, default=os.path.join(OUTPUT_DIR, 'train_dataset.npz'),
                        help='Путь к файлу датасета (.npz)')
    parser.add_argument('--output', type=str, default=None,
                        help='Директория для сохранения результатов анализа')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Ошибка: файл датасета {args.dataset} не найден.")
        return
    
    analyze_dataset(args.dataset, args.output)

if __name__ == '__main__':
    main()

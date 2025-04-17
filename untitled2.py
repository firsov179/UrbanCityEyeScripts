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

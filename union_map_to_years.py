import xml.etree.ElementTree as ET
from datetime import datetime


# Функция для фильтрации объектов по заданному году
def filter_by_year(osm_data, year):
    root = ET.fromstring(osm_data)
    filtered_elements = ET.Element("osm")

    for element in root:
        start_date = element.find("tag[@k='start_date']")
        end_date = element.find("tag[@k='end_date']")

        # Функция для извлечения года из даты
        def get_year(date_str):
            sep = date_str.find('-')
            if sep != -1 and sep != 0:
                return int(date_str[:sep])
            try:
                return int(date_str)
            except:
                return -1

        # Извлекаем год из start_date и end_date
        start_year = get_year(start_date.attrib['v']) if start_date is not None else None
        start_year = start_year if start_year is not None else -1

        if  start_year > year:
            continue

        end_year = get_year(end_date.attrib['v']) if end_date is not None else None
        end_year = end_year if end_year is not None and end_year != -1 else 2026

        if end_year < year:
            continue

        filtered_elements.append(element)

    return ET.tostring(filtered_elements)


# Основной код
if __name__ == "__main__":
    # Загружаем данные из файла
    with open("merged_map_london.osm", "rb") as f:
        osm_data = f.read()

    # Создаем карты для каждого 5-го года с 1561 по 2021
    for year in range(1500, 2025, 5):
        filtered_osm = filter_by_year(osm_data, year)

        # Сохраняем результат в файл
        with open(f"maps_by_years/map_{year}.osm", "wb") as f:
            f.write(filtered_osm)

        print(f"Карта для {year} года успешно сохранена в map_{year}.osm")

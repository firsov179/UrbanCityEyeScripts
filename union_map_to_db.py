import xml.etree.ElementTree as ET
import psycopg2
import time
import os
from collections import defaultdict

# Параметры подключения к PostgreSQL
db_params = {
    'host': 'localhost',
    'user': 'urbanfuture',
    'password': 'FireAndBear',
    'database': 'city_simulation_db'
}

# Определение объектов для каждого режима моделирования
TRANSPORT_OBJECTS = {
    'highway', 'railway', 'waterway', 'aeroway', 'ferry', 'public_transport',
    'building:transportation', 'building:train_station', 'amenity:ferry_terminal',
    'amenity:bus_station', 'amenity:taxi'
}

HOUSING_OBJECTS = {
    'building', 'landuse', 'amenity', 'leisure', 'shop', 'tourism',
    'historic', 'man_made', 'office', 'residential', 'commercial'
}

# Категории объектов для каждого режима
TRANSPORT_CATEGORIES = [
    'highway:', 'railway:', 'waterway:', 'building:transportation', 'building:train_station',
    'building:train_station; railway:', 'amenity:parking'
]

HOUSING_CATEGORIES = [
    'building:', 'landuse:', 'amenity:', 'leisure:',
    'shop:', 'tourism:', 'historic:'
]

# Явно определенные категории для транспорта
EXPLICIT_TRANSPORT = {
    'amenity:bus_station', 'amenity:ferry_terminal', 'amenity:taxi', 'amenity:charging_station',
    'amenity:bicycle_parking', 'amenity:bicycle_rental', 'amenity:car_sharing',
    'amenity:motorcycle_parking', 'building:hangar', 'building:garage', 'building:garages',
    'building:parking', 'railway:station', 'railway:halt', 'railway:tram_stop', 'railway:subway_entrance'
}

# Явно определенные категории для жилья
EXPLICIT_HOUSING = {
    'building:residential', 'building:apartments', 'building:house', 'building:dormitory',
    'building:hotel', 'building:office', 'building:commercial', 'building:retail',
    'amenity:school', 'amenity:hospital', 'amenity:university', 'amenity:college',
    'amenity:library', 'amenity:theatre', 'amenity:cinema', 'amenity:restaurant', 'amenity:cafe',
    'amenity:pub', 'amenity:bar', 'amenity:marketplace', 'amenity:place_of_worship',
    'leisure:park', 'leisure:garden', 'leisure:playground'
}

# Добавление городов с информацией на английском и русском
cities_info = [
    {
        'name': 'London',
        'name_ru': 'Лондон',
        'country': 'United Kingdom',
        'country_ru': 'Великобритания',
        'foundation': 'Traditionally 43 AD (Roman Foundation)',
        'foundation_ru': 'Традиционно 43 год н.э. (Римское основание)',
        'description': 'London is the capital and largest city of England and the United Kingdom. Known for the arts, commerce, education, and finance.',
        'description_ru': 'Лондон — столица и крупнейший город Англии и Великобритании. Известен искусством, коммерцией, образованием и финансами.',
        'start_date': 1500,
        'end_date': 2050,
        'osm_file': 'merged_map_london.osm'
    },
    {
        'name': 'Paris',
        'name_ru': 'Париж',
        'country': 'France',
        'country_ru': 'Франция',
        'foundation': 'Approximately 3rd century BC by the Parisii, a Celtic people',
        'foundation_ru': 'Около III века до н.э. племенем паризиев, кельтским народом',
        'description': 'Paris is the capital and most populous city of France. Known for its art, fashion, gastronomy, and culture.',
        'description_ru': 'Париж — столица и самый густонаселённый город Франции. Известен искусством, модой, гастрономией и культурой.',
        'start_date': 1500,
        'end_date': 2050,
        'osm_file': None # 'merged_map_paris.osm'
    },
    {
        'name': 'Moscow',
        'name_ru': 'Москва',
        'country': 'Russia',
        'country_ru': 'Россия',
        'foundation': 'Approximately 1147 yesr by Yuri Dolgorukiy',
        'foundation_ru': 'Около 1147 года н.э. удельным князем Юрием Долгоруким',
        'description': 'Moscow is the capital and largest city of Russia. Known for its architecture, historic buildings like the Kremlin and Saint Basil’s Cathedral.',
        'description_ru': 'Москва — столица и крупнейший город России. Известна архитектурой, историческими зданиями, такими как Кремль и Собор Василия Блаженного.',
        'start_date': 1500,
        'end_date': 2050,
        'osm_file': None  # Нет файла для Москвы
    },
    {
        'name': 'Rome',
        'name_ru': 'Рим',
        'country': 'Italy',
        'country_ru': 'Италия',
        'foundation': 'Traditionally 753 BC',
        'foundation_ru': 'Традиционно 753 год до н.э.',
        'description': 'Rome is the capital city of Italy, known as the "Eternal City". Famous for its ancient ruins such as the Forum and the Colosseum.',
        'description_ru': 'Рим — столица Италии, известен как "Вечный город". Знаменит древними руинами, такими как Форум и Колизей.',
        'start_date': 1500,
        'end_date': 2050,
        'osm_file': None  # Нет файла для Рима
    },
    {
        'name': 'Saint Petersburg',
        'name_ru': 'Санкт-Петербург',
        'country': 'Russia',
        'country_ru': 'Россия',
        'foundation': 'May 27, 1703',
        'foundation_ru': '27 мая 1703 года',
        'description': 'Saint Petersburg is a major city in Russia, founded by Tsar Peter the Great. Known for its imperial history and architecture.',
        'description_ru': 'Санкт-Петербург — крупный город в России, основанный Петром Великим. Известен имперской историей и архитектурой.',
        'start_date': 1705,
        'end_date': 2050,
        'osm_file': None  # Нет файла для Санкт-Петербурга
    },
    {
        'name': 'Seoul',
        'name_ru': 'Сеул',
        'country': 'South Korea',
        'country_ru': 'Южная Корея',
        'foundation': 'Founded in 18 BC as Wiryeseong',
        'foundation_ru': 'Основан в 18 году до н.э. как Виресон',
        'description': 'Seoul is the capital and largest metropolis of South Korea. It features a combination of modern skyscrapers, high-tech infrastructure, temples, and palaces.',
        'description_ru': 'Сеул — столица и крупнейший город Южной Кореи. Сочетает в себе современные небоскребы, высокотехнологичную инфраструктуру, храмы и дворцы.',
        'start_date': 1500,
        'end_date': 2050,
        'osm_file': None #'merged_map_seoul.osm'
    }
]


def get_year(date_str):
    """Извлекает год из строки с датой"""
    if not date_str:
        return None

    sep = date_str.find('-')
    if sep != -1 and sep != 0:
        try:
            return int(date_str[:sep])
        except ValueError:
            pass

    try:
        return int(date_str)
    except (ValueError, TypeError):
        return None


def determine_mode(role):
    """Определяет, к какому режиму моделирования относится объект"""
    if not role:
        return None

    # Проверяем явно определенные категории
    if role in EXPLICIT_TRANSPORT:
        return 'transport'

    if role in EXPLICIT_HOUSING:
        return 'housing'

    # Проверяем по префиксам для транспорта
    for category in TRANSPORT_CATEGORIES:
        if role.startswith(category):
            return 'transport'

    # Проверяем по префиксам для жилья
    for category in HOUSING_CATEGORIES:
        if role.startswith(category):
            return 'housing'

    # Проверяем по тегам, разделенным через ";"
    if ";" in role:
        parts = role.split(";")
        for part in parts:
            part = part.strip()
            # Для транспорта
            for category in TRANSPORT_CATEGORIES:
                if part.startswith(category):
                    return 'transport'
            # Для жилья
            for category in HOUSING_CATEGORIES:
                if part.startswith(category):
                    return 'housing'

    # По умолчанию относим к жилью
    return 'housing'


def round_year_to_nearest_five(year):
    """Округляет год до ближайшего кратного 5"""
    return ((year + 2) // 5) * 5


def setup_initial_data(conn):
    """Создает начальные данные: города, режимы и симуляции по годам"""
    cursor = conn.cursor()
    city_ids = {}

    # Проверяем, нет ли уже городов в базе данных
    cursor.execute("SELECT id, name FROM City;")
    existing_cities = {row[1]: row[0] for row in cursor.fetchall()}

    for city_info in cities_info:
        city_name = city_info['name']
        if city_name in existing_cities:
            city_ids[city_name] = existing_cities[city_name]
            print(f"City '{city_name}' already exists with ID: {city_ids[city_name]}")
        else:
            cursor.execute("""
                INSERT INTO City (name, name_ru, country, country_ru, foundation, foundation_ru, description, description_ru) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (city_info['name'], city_info['name_ru'], city_info['country'], city_info['country_ru'],
                  city_info['foundation'], city_info['foundation_ru'], city_info['description'],
                  city_info['description_ru']))

            city_id = cursor.fetchone()[0]
            city_ids[city_name] = city_id
            print(f"Created city '{city_name}' with ID: {city_id}")

    # Проверяем, нет ли уже режимов в базе данных
    cursor.execute("SELECT id, name FROM Mode;")
    existing_modes = {row[1]: row[0] for row in cursor.fetchall()}

    # Добавление режимов моделирования
    if 'Transport Infrastructure Modeling' in existing_modes:
        transport_mode_id = existing_modes['Transport Infrastructure Modeling']
        print(f"Mode 'Transport Infrastructure Modeling' already exists with ID: {transport_mode_id}")
    else:
        cursor.execute("INSERT INTO Mode (name) VALUES ('Transport Infrastructure Modeling') RETURNING id;")
        transport_mode_id = cursor.fetchone()[0]
        print(f"Created mode 'Transport Infrastructure Modeling' with ID: {transport_mode_id}")

    if 'Housing Development Modeling' in existing_modes:
        housing_mode_id = existing_modes['Housing Development Modeling']
        print(f"Mode 'Housing Development Modeling' already exists with ID: {housing_mode_id}")
    else:
        cursor.execute("INSERT INTO Mode (name) VALUES ('Housing Development Modeling') RETURNING id;")
        housing_mode_id = cursor.fetchone()[0]
        print(f"Created mode 'Housing Development Modeling' with ID: {housing_mode_id}")

    # Создание словаря для хранения ID симуляций по городам
    simulation_ids = {}
    for city_info in cities_info:
        city_name = city_info['name']
        simulation_ids[city_name] = {'transport': {}, 'housing': {}}

    # Проверяем, какие симуляции уже существуют
    cursor.execute("""
        SELECT s.id, c.name, m.name, s.year 
        FROM Simulation s 
        JOIN City c ON s.city_id = c.id 
        JOIN Mode m ON s.mode_id = m.id;
    """)
    existing_simulations = cursor.fetchall()

    for sim_id, city_name, mode_name, year in existing_simulations:
        if city_name in simulation_ids:
            if mode_name == 'Transport Infrastructure Modeling':
                simulation_ids[city_name]['transport'][year] = sim_id
            elif mode_name == 'Housing Development Modeling':
                simulation_ids[city_name]['housing'][year] = sim_id

    # Создаем недостающие симуляции для каждого города
    for city_info in cities_info:
        city_name = city_info['name']
        city_id = city_ids[city_name]
        
        # Округляем начальную и конечную даты до кратных 5
        start_date = round_year_to_nearest_five(city_info['start_date'])
        end_date = round_year_to_nearest_five(city_info['end_date'])

        for year in range(start_date, end_date + 1, 5):  # С шагом 5 лет
            # Проверяем, существует ли симуляция для транспортной инфраструктуры
            if year not in simulation_ids[city_name]['transport']:
                cursor.execute(
                    "INSERT INTO Simulation (city_id, mode_id, year) VALUES (%s, %s, %s) RETURNING id;",
                    (city_id, transport_mode_id, year)
                )
                transport_sim_id = cursor.fetchone()[0]
                simulation_ids[city_name]['transport'][year] = transport_sim_id

            # Проверяем, существует ли симуляция для городской застройки
            if year not in simulation_ids[city_name]['housing']:
                cursor.execute(
                    "INSERT INTO Simulation (city_id, mode_id, year) VALUES (%s, %s, %s) RETURNING id;",
                    (city_id, housing_mode_id, year)
                )
                housing_sim_id = cursor.fetchone()[0]
                simulation_ids[city_name]['housing'][year] = housing_sim_id

        print(
            f"Created simulations for {city_name} for years from {start_date} to {end_date} (every 5 years) in both modes")

    conn.commit()
    return city_ids, {'transport': transport_mode_id, 'housing': housing_mode_id}, simulation_ids


def parse_osm_and_fill_database(osm_file_path, city_name, city_id, simulation_ids, conn):
    """
    Парсит OSM файл и заполняет базу данных геообъектами для указанного города

    Args:
        osm_file_path: Путь к OSM файлу
        city_name: Название города
        city_id: ID города в базе данных
        simulation_ids: Словарь с ID симуляций
        conn: Соединение с базой данных
    """
    print(f"Starting OSM file processing for {city_name}: {osm_file_path}")

    # Загружаем данные из файла
    try:
        tree = ET.parse(osm_file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error reading OSM file: {e}")
        return

    # Подготовка словаря для хранения узлов (nodes)
    nodes_dict = {}

    # Первый проход - извлечение всех узлов
    print("First pass - extracting all nodes...")
    node_count = 0
    for element in root:
        if element.tag == 'node':
            node_id = element.get('id')
            lat = float(element.get('lat'))
            lon = float(element.get('lon'))
            nodes_dict[node_id] = (lon, lat)
            node_count += 1
            if node_count % 10000 == 0:
                print(f"Processed {node_count} nodes...")

    print(f"Extracted {len(nodes_dict)} nodes from OSM file")

    # Проверяем, нет ли уже объектов для этого города
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM GeoObjectSimulation gos
        JOIN Simulation s ON gos.simulation_id = s.id
        WHERE s.city_id = %s
    """, (city_id,))
    existing_objects_count = cursor.fetchone()[0]

    if existing_objects_count > 0:
        print(f"City {city_name} already has {existing_objects_count} geo objects. Skipping import.")
        return

    # Второй проход - обработка и вставка только линий и полигонов
    print("Second pass - processing ways and relations only...")

    processed_count = 0
    skipped_count = 0
    transport_count = 0
    housing_count = 0

    for element in root:
        # Обрабатываем только линии (way) и отношения (relation), пропускаем узлы (node)
        if element.tag not in ('way', 'relation'):
            continue

        # Начинаем новую транзакцию для каждого элемента
        try:
            # Извлекаем информацию об объекте
            element_id = element.get('id')
            element_type = element.tag

            # Проверяем наличие тегов
            start_date = None
            end_date = None
            name = None
            role = None

            # Собираем все теги в строку для description
            all_tags_str = []

            for tag in element.findall('tag'):
                k = tag.get('k')
                v = tag.get('v')

                if k == 'start_date':
                    start_date = v
                elif k == 'end_date':
                    end_date = v
                elif k == 'name':
                    name = v
                elif k in ('highway', 'railway', 'amenity', 'building', 'landuse', 'natural', 'waterway'):
                    role = f"{k}:{v}" if role is None else f"{role}; {k}:{v}"
                else:
                    # Добавляем каждый тег в общую строку
                    all_tags_str.append(f'{k}: {v}')

            # Формируем description из всех тегов
            description = ",".join(all_tags_str) if all_tags_str else None

            # Если нет тегов с ролью, определяем роль по типу элемента
            if not role:
                role = element_type

            # Формируем имя объекта
            if not name:
                name = f"{element_type}_{element_id}"

            start_year = get_year(start_date) or 1400  # По умолчанию с начала периода
            start_year = round_year_to_nearest_five(start_year)
            
            end_year = get_year(end_date) or 2100  # По умолчанию до конца периода
            end_year = round_year_to_nearest_five(end_year)
            
            start_year = max(1500, start_year)
            end_year = min(2050, end_year)

            # Определяем, к какому режиму моделирования относится объект
            mode_type = determine_mode(role)

            # Если не удалось определить режим, пропускаем объект
            if not mode_type:
                skipped_count += 1
                continue

            # Создаем геометрию на основе типа элемента
            geom_query = None

            if element_type == 'way':
                # Для путей собираем все точки
                node_refs = []
                for nd in element.findall('nd'):
                    ref = nd.get('ref')
                    if ref in nodes_dict:
                        node_refs.append(nodes_dict[ref])

                if len(node_refs) > 1:
                    # Проверяем, является ли путь замкнутым (полигоном)
                    is_polygon = (len(node_refs) > 3 and node_refs[0] == node_refs[-1])

                    points_text = ", ".join([f"ST_MakePoint({lon}, {lat})" for lon, lat in node_refs])

                    if is_polygon:
                        geom_query = f"ST_SetSRID(ST_MakePolygon(ST_MakeLine(ARRAY[{points_text}])), 4326)"
                    else:
                        geom_query = f"ST_SetSRID(ST_MakeLine(ARRAY[{points_text}]), 4326)"
                else:
                    # Недостаточно точек для линии или полигона
                    skipped_count += 1
                    continue

            elif element_type == 'relation':
                # Для отношений используем упрощенный подход - пропускаем их
                skipped_count += 1
                continue

            if not geom_query:
                skipped_count += 1
                continue

            # Начинаем новую транзакцию для каждого объекта
            with conn.cursor() as object_cursor:
                # Добавляем объект в таблицу GeoObject
                object_cursor.execute(
                    f"INSERT INTO GeoObject (name, role, description, location) VALUES (%s, %s, %s, {geom_query}) RETURNING id;",
                    (name, role, description)
                )
                geo_object_id = object_cursor.fetchone()[0]

                for year in range(start_year, end_year + 1, 5):
                    if year in simulation_ids[city_name][mode_type]:
                        object_cursor.execute(
                            "INSERT INTO GeoObjectSimulation (simulation_id, geo_object_id) VALUES (%s, %s);",
                            (simulation_ids[city_name][mode_type][year], geo_object_id)
                        )

            conn.commit()
            processed_count += 1

            # Считаем объекты по режимам
            if mode_type == 'transport':
                transport_count += 1
            elif mode_type == 'housing':
                housing_count += 1

            # Выводим статус
            if processed_count % 100 == 0:
                print(
                    f"Processed {processed_count} objects (Transport: {transport_count}, Housing: {housing_count}), skipped {skipped_count}...")

        except Exception as e:
            conn.rollback()  # Откатываем неудачную транзакцию
            print(f"Error processing {element_type}_{element_id}: {e}")
            skipped_count += 1

    print(
        f"OSM file processing completed for {city_name}. Added: {processed_count} (Transport: {transport_count}, Housing: {housing_count}), Skipped: {skipped_count}")


def main():
    start_time = time.time()

    try:
        # Подключение к базе данных
        conn = psycopg2.connect(**db_params)

        # Создание начальных данных
        city_ids, mode_ids, simulation_ids = setup_initial_data(conn)

        # Обработка OSM файлов для каждого города
        for city_info in cities_info:
            city_name = city_info['name']
            osm_file = city_info.get('osm_file')

            if osm_file and os.path.exists(osm_file):
                parse_osm_and_fill_database(
                    osm_file,
                    city_name,
                    city_ids[city_name],
                    simulation_ids,
                    conn
                )
            elif osm_file:
                print(f"OSM file for {city_name} not found: {osm_file}")
            else:
                print(f"No OSM file specified for {city_name}")

        elapsed_time = time.time() - start_time
        print(f"Database population completed in {elapsed_time:.2f} seconds!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    main()

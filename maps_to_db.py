import xml.etree.ElementTree as ET
import psycopg2
from psycopg2.extras import execute_values
import os
from datetime import datetime

# Параметры подключения к PostgreSQL
db_params = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'postgres',
    'database': 'city_simulation_db'
}

OSM_FILE = "merged_map.osm"


def get_year(date_str):
    """Извлечение года из строки даты"""
    if not date_str:
        return None

    sep = date_str.find('-')
    if sep != -1 and sep != 0:
        try:
            return int(date_str[:sep])
        except ValueError:
            return None
    try:
        return int(date_str)
    except ValueError:
        return None


def load_osm_data():
    """Загрузка данных из OSM файла"""
    print(f"Загрузка данных из {OSM_FILE}...")
    tree = ET.parse(OSM_FILE)
    root = tree.getroot()
    return root


def extract_node_coordinates(osm_root):
    """Извлечение координат узлов для построения линий и полигонов"""
    nodes = {}
    for node in osm_root.findall('node'):
        node_id = node.attrib.get('id')
        lat = float(node.attrib.get('lat', 0))
        lon = float(node.attrib.get('lon', 0))
        nodes[node_id] = (lon, lat)
    return nodes


def extract_geo_objects(osm_root):
    """Извлечение геообъектов из OSM файла"""
    geo_objects = []

    # Сначала извлекаем координаты всех узлов
    nodes = extract_node_coordinates(osm_root)

    # Обрабатываем только way и relation (линии и полигоны)
    for element in osm_root:
        if element.tag not in ['way', 'relation']:
            continue

        obj = {
            'osm_id': element.attrib.get('id'),
            'osm_type': element.tag,
            'name': None,
            'role': None,
            'description': None,
            'start_year': None,
            'end_year': None,
            'geometry': None,
            'tags': {}
        }

        # Обработка тегов
        for tag in element.findall('tag'):
            key = tag.attrib.get('k')
            value = tag.attrib.get('v')

            obj['tags'][key] = value

            if key == 'name':
                obj['name'] = value
            elif key == 'start_date':
                obj['start_year'] = get_year(value)
            elif key == 'end_date':
                obj['end_year'] = get_year(value)
            elif key == 'role':
                obj['role'] = value
            elif key == 'description':
                obj['description'] = value

        # Если имя не задано, используем тип + id
        if not obj['name']:
            obj['name'] = f"{obj['osm_type']}_{obj['osm_id']}"

        # Если роль не задана, используем тег amenity или building
        if not obj['role']:
            obj['role'] = obj['tags'].get('amenity') or obj['tags'].get('building') or 'unknown'

        # Если описания нет, создаем из тегов
        if not obj['description']:
            obj['description'] = "; ".join([f"{k}={v}" for k, v in obj['tags'].items()
                                            if k not in ['name', 'start_date', 'end_date']])

        # Обработка геометрии для линий (way)
        if element.tag == 'way':
            # Получаем все узлы пути
            node_refs = [nd.attrib.get('ref') for nd in element.findall('nd')]

            # Собираем координаты узлов
            coords = []
            for ref in node_refs:
                if ref in nodes:
                    coords.append(nodes[ref])

            if len(coords) >= 2:
                # Проверяем, является ли путь замкнутым (полигоном)
                is_polygon = node_refs[0] == node_refs[-1] and len(coords) >= 4

                if is_polygon:
                    # Создаем полигон
                    coords_str = ', '.join([f"{lon} {lat}" for lon, lat in coords])
                    obj['geometry'] = f"POLYGON(({coords_str}))"
                else:
                    # Создаем линию
                    coords_str = ', '.join([f"{lon} {lat}" for lon, lat in coords])
                    obj['geometry'] = f"LINESTRING({coords_str})"

                geo_objects.append(obj)

        # Для relation требуется более сложная логика, которую можно добавить при необходимости
        # В этом примере мы пропустим relation для простоты

    return geo_objects


def add_constraints():
    """Добавление ограничений в таблицы"""
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    try:
        # Проверяем существование ограничений перед их добавлением

        # Для таблицы City
        cursor.execute("""
        SELECT constraint_name
        FROM information_schema.table_constraints
        WHERE table_name = 'city' AND constraint_name = 'unique_city_name';
        """)
        if not cursor.fetchone():
            cursor.execute("""
            ALTER TABLE City ADD CONSTRAINT unique_city_name UNIQUE (name);
            """)
            print("Добавлено ограничение unique_city_name")

        # Для таблицы Mode
        cursor.execute("""
        SELECT constraint_name
        FROM information_schema.table_constraints
        WHERE table_name = 'mode' AND constraint_name = 'unique_mode_name';
        """)
        if not cursor.fetchone():
            cursor.execute("""
            ALTER TABLE Mode ADD CONSTRAINT unique_mode_name UNIQUE (name);
            """)
            print("Добавлено ограничение unique_mode_name")

        # Для таблицы Simulation
        cursor.execute("""
        SELECT constraint_name
        FROM information_schema.table_constraints
        WHERE table_name = 'simulation' AND constraint_name = 'unique_simulation_city_mode_year';
        """)
        if not cursor.fetchone():
            cursor.execute("""
            ALTER TABLE Simulation ADD CONSTRAINT unique_simulation_city_mode_year UNIQUE (city_id, mode_id, year);
            """)
            print("Добавлено ограничение unique_simulation_city_mode_year")

        # Для таблицы GeoObject
        cursor.execute("""
        SELECT constraint_name
        FROM information_schema.table_constraints
        WHERE table_name = 'geoobject' AND constraint_name = 'unique_geo_object_name_role';
        """)
        if not cursor.fetchone():
            cursor.execute("""
            ALTER TABLE GeoObject ADD CONSTRAINT unique_geo_object_name_role UNIQUE (name, role);
            """)
            print("Добавлено ограничение unique_geo_object_name_role")

        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при добавлении ограничений: {e}")
    finally:
        cursor.close()
        conn.close()


def insert_city_and_mode():
    """Вставка данных города и режима моделирования"""
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    try:
        # Вставка города
        cursor.execute("""
        INSERT INTO City (name) 
        VALUES ('London') 
        ON CONFLICT (name) DO NOTHING 
        RETURNING id;
        """)
        city_result = cursor.fetchone()

        if city_result:
            city_id = city_result[0]
        else:
            cursor.execute("SELECT id FROM City WHERE name = 'London';")
            city_id = cursor.fetchone()[0]

        # Вставка режима моделирования
        cursor.execute("""
        INSERT INTO Mode (name) 
        VALUES ('Historical Development') 
        ON CONFLICT (name) DO NOTHING 
        RETURNING id;
        """)
        mode_result = cursor.fetchone()

        if mode_result:
            mode_id = mode_result[0]
        else:
            cursor.execute("SELECT id FROM Mode WHERE name = 'Historical Development';")
            mode_id = cursor.fetchone()[0]

        conn.commit()
        return city_id, mode_id

    except Exception as e:
        conn.rollback()
        print(f"Ошибка при вставке города и режима: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def create_simulations(city_id, mode_id):
    """Создание симуляций для годов с 1500 по 2020 с шагом 5 лет"""
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    simulation_ids = {}
    years = range(1500, 2025, 5)

    try:
        for year in years:
            cursor.execute("""
            INSERT INTO Simulation (city_id, mode_id, year)
            VALUES (%s, %s, %s)
            ON CONFLICT (city_id, mode_id, year) DO NOTHING
            RETURNING id;
            """, (city_id, mode_id, year))

            result = cursor.fetchone()
            if result:
                simulation_ids[year] = result[0]
            else:
                cursor.execute("""
                SELECT id FROM Simulation 
                WHERE city_id = %s AND mode_id = %s AND year = %s
                """, (city_id, mode_id, year))
                result = cursor.fetchone()
                if result:
                    simulation_ids[year] = result[0]

        conn.commit()
        return simulation_ids

    except Exception as e:
        conn.rollback()
        print(f"Ошибка при создании симуляций: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def insert_geo_objects(geo_objects):
    """Вставка геообъектов в таблицу GeoObject"""
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    geo_object_ids = {}
    inserted_count = 0

    try:
        for obj in geo_objects:
            # Пропускаем объекты без геометрии
            if not obj['geometry']:
                continue

            # Задаем значения по умолчанию
            name = obj['name'] or 'Unnamed Object'
            role = obj['role'] or 'Unknown'
            description = obj['description'] or ''

            try:
                cursor.execute("""
                INSERT INTO GeoObject (name, role, description, location)
                VALUES (%s, %s, %s, ST_GeomFromText(%s, 4326))
                ON CONFLICT (name, role) DO UPDATE 
                SET description = EXCLUDED.description
                RETURNING id;
                """, (name, role, description, obj['geometry']))

                db_id = cursor.fetchone()[0]
                inserted_count += 1

                # Сохраняем информацию о годах существования объекта
                start_year = obj['start_year'] if obj['start_year'] is not None else 1500
                end_year = obj['end_year'] if obj['end_year'] is not None else 2025

                geo_object_ids[db_id] = {
                    'start_year': start_year,
                    'end_year': end_year
                }
            except Exception as e:
                print(f"Ошибка при вставке геообъекта {name}: {e}")
                print(f"Геометрия: {obj['geometry']}")
                continue

        conn.commit()
        print(f"Успешно вставлено {inserted_count} геообъектов")
        return geo_object_ids

    except Exception as e:
        conn.rollback()
        print(f"Ошибка при вставке геообъектов: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def link_objects_to_simulations(geo_object_ids, simulation_ids):
    """Связывание геообъектов с симуляциями на основе годов существования"""
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    try:
        # Массив для хранения данных для массовой вставки
        link_data = []

        for geo_object_id, years_info in geo_object_ids.items():
            start_year = years_info['start_year']
            end_year = years_info['end_year']

            # Связываем объект с симуляциями за годы его существования
            for year, simulation_id in simulation_ids.items():
                if start_year <= year <= end_year:
                    link_data.append((simulation_id, geo_object_id))

        # Разбиваем на партии по 1000 элементов
        for i in range(0, len(link_data), 1000):
            batch = link_data[i:i + 1000]
            # Массовая вставка данных
            execute_values(cursor, """
            INSERT INTO GeoObjectSimulation (simulation_id, geo_object_id)
            VALUES %s
            ON CONFLICT DO NOTHING
            """, batch)

        conn.commit()
        print(f"Связано {len(link_data)} объектов с симуляциями")

    except Exception as e:
        conn.rollback()
        print(f"Ошибка при связывании объектов с симуляциями: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def main():
    # Добавляем ограничения в таблицы
    add_constraints()

    # Загружаем OSM данные
    osm_root = load_osm_data()

    # Извлекаем геообъекты с информацией о годах существования
    geo_objects = extract_geo_objects(osm_root)
    print(f"Извлечено {len(geo_objects)} линий и полигонов из OSM файла")

    # Вставляем город и режим моделирования
    city_id, mode_id = insert_city_and_mode()

def main():
    """Основная функция"""
    try:
        # Добавляем ограничения в таблицы
        add_constraints()

        # Загружаем OSM данные
        osm_root = load_osm_data()

        # Извлекаем геообъекты с информацией о годах существования
        geo_objects = extract_geo_objects(osm_root)
        print(f"Извлечено {len(geo_objects)} объектов из OSM файла")

        # Вставляем город и режим моделирования
        city_id, mode_id = insert_city_and_mode()
        print(f"Город Лондон (ID: {city_id}) и режим моделирования (ID: {mode_id}) созданы")

        # Создаем симуляции для каждого года
        simulation_ids = create_simulations(city_id, mode_id)
        print(f"Создано {len(simulation_ids)} симуляций для разных лет")

        # Вставляем геообъекты
        geo_object_ids = insert_geo_objects(geo_objects)
        print(f"Вставлено {len(geo_object_ids)} геообъектов в базу данных")

        # Связываем геообъекты с симуляциями
        link_objects_to_simulations(geo_object_ids, simulation_ids)

        print("Загрузка данных успешно завершена")

    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")


if __name__ == "__main__":
    main()


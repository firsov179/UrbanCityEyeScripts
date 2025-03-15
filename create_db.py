import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Параметры подключения к PostgreSQL
db_params = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'postgres',
    'database': 'postgres'
}

# Имя новой базы данных
new_db_name = 'city_simulation_db'


def create_database():
    """Создание новой базы данных с поддержкой PostGIS"""
    conn = psycopg2.connect(**db_params)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Проверка существования базы данных
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{new_db_name}'")
    exists = cursor.fetchone()

    if not exists:
        print(f"Создание базы данных {new_db_name}...")
        cursor.execute(f"CREATE DATABASE {new_db_name}")
        print(f"База данных {new_db_name} успешно создана")
    else:
        print(f"База данных {new_db_name} уже существует")

    cursor.close()
    conn.close()


def setup_database_schema():
    """Настройка схемы базы данных с таблицами"""
    # Обновление параметров для подключения к новой базе данных
    db_params['database'] = new_db_name

    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Включение расширения PostGIS
    cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    print("Расширение PostGIS включено")

    # Создание таблиц

    # 1. City (Город)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS City (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        name_ru VARCHAR(255) NOT NULL,
        country VARCHAR(255) NOT NULL,
        country_ru VARCHAR(255) NOT NULL,
        foundation TEXT,
        foundation_ru TEXT,
        description TEXT,
        description_ru TEXT
    );
    """)

    # 2. Mode (Режим моделирования)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Mode (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL
    );
    """)

    # 3. Simulation (Моделирование)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Simulation (
        id SERIAL PRIMARY KEY,
        city_id INTEGER REFERENCES City(id),
        mode_id INTEGER REFERENCES Mode(id),
        year INTEGER NOT NULL
    );
    """)

    # 4. GeoObject (Объект инфраструктуры)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS GeoObject (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        role VARCHAR(255),
        role_ru VARCHAR(255),
        description TEXT,
        description_ru TEXT,
        location GEOMETRY NOT NULL
    );
    """)

    # 5. GeoObjectSimulation (Таблица для оптимизации запросов)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS GeoObjectSimulation (
        id SERIAL PRIMARY KEY,
        simulation_id INTEGER REFERENCES Simulation(id),
        geo_object_id INTEGER REFERENCES GeoObject(id)
    );
    """)

    # 6. HistoricalPeriod (Историческая эпоха)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS HistoricalPeriod (
        id SERIAL PRIMARY KEY
    );
    """)

    # Создание индексов для оптимизации запросов
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_geoobject_location ON GeoObject USING GIST(location);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_simulation_city ON Simulation(city_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_simulation_mode ON Simulation(mode_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_geo_object_simulation_sim ON GeoObjectSimulation(simulation_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_geo_object_simulation_obj ON GeoObjectSimulation(geo_object_id);")

    conn.commit()
    print("Все таблицы успешно созданы")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    try:
        create_database()
        setup_database_schema()
        print("База данных успешно настроена")
    except Exception as e:
        print(f"Ошибка при настройке базы данных: {e}")

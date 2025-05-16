import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Параметры подключения к PostgreSQL
db_params = {
    'host': 'localhost',
    'user': 'urbanfuture',
    'password': 'FireAndBear',
    'database': 'city_simulation_db'
}


def clean_database():
    """Очищает все данные из таблиц базы данных, сохраняя структуру"""
    try:
        # Подключение к базе данных
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        print("Starting database cleanup...")

        # Получаем список всех таблиц в базе данных
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE';
        """)

        tables = cursor.fetchall()

        # Временно отключаем ограничения внешних ключей для облегчения очистки
        cursor.execute("SET CONSTRAINTS ALL DEFERRED;")

        # Очищаем каждую таблицу
        for table in tables:
            table_name = table[0]
            try:
                print(f"Cleaning table: {table_name}")
                cursor.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
            except Exception as e:
                print(f"Error clearing table {table_name}: {e}")

        # Сбрасываем все последовательности (auto-increment counters)
        cursor.execute("""
            SELECT sequence_name 
            FROM information_schema.sequences 
            WHERE sequence_schema = 'public';
        """)

        sequences = cursor.fetchall()

        for sequence in sequences:
            sequence_name = sequence[0]
            try:
                print(f"Resetting sequence: {sequence_name}")
                cursor.execute(f"ALTER SEQUENCE {sequence_name} RESTART WITH 1;")
            except Exception as e:
                print(f"Error resetting sequence {sequence_name}: {e}")

        print("Database cleanup completed successfully!")

    except Exception as e:
        print(f"Error during database cleanup: {e}")
    finally:
        if 'conn' in locals():
            cursor.close()
            conn.close()


if __name__ == "__main__":
    clean_database()

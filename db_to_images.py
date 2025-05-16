import geopandas as gpd
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
from shapely import wkt
import json
import os

# Создаем директорию для сохранения изображений, если она не существует
os.makedirs('maps_by_years_image', exist_ok=True)

# Настройки подключения к базе данных
db_connection_string = "postgresql://postgres:postgres@localhost:5432/city_simulation_db"
engine = create_engine(db_connection_string)

# Определение цветов для различных типов объектов
highway_values = [
    'bridleway', 'footway', 'motorway', 'pedestrian', 'primary',
    'residential', 'secondary', 'service', 'stairs', 'steps',
    'tertiary', 'track', 'trunk', 'trunk_link', 'unclassified'
]

railway_values = [
    'rail', 'station', 'subway', 'tram'
]

waterway_values = [
    'river', 'stream', 'canal'
]

colors = {
    'bridleway': (0, 255, 0),  # green
    'footway': (169, 169, 169),  # darkgrey
    'motorway': (0, 0, 255),  # red
    'pedestrian': (255, 165, 0),  # orange
    'primary': (128, 0, 128),  # purple
    'residential': (255, 105, 180),  # hot pink
    'secondary': (255, 140, 0),  # dark orange
    'service': (0, 255, 255),  # cyan
    'stairs': (165, 42, 42),  # brown
    'steps': (84, 58, 232),  # blue purple
    'tertiary': (144, 238, 144),  # lightgreen
    'track': (173, 216, 230),  # lightblue
    'trunk': (139, 0, 0),  # darkred
    'trunk_link': (255, 69, 0),  # orange red
    'unclassified': (255, 0, 0),  # blue
    'rail': (0, 0, 0),  # black
    'station': (0, 0, 139),  # darkblue
    'subway': (0, 100, 0),  # darkgreen
    'tram': (238, 130, 238),  # violet
    'river': (0, 191, 255),  # deep sky blue
    'stream': (135, 206, 250),  # light sky blue
    'canal': (70, 130, 180)  # steel blue
}


def rgb_to_normalized(rgb):
    return tuple(c / 255 for c in rgb)


normalized_colors = {key: rgb_to_normalized(value) for key, value in colors.items()}

# Получение списка годов из базы данных
years_query = """
SELECT DISTINCT year 
FROM simulation 
ORDER BY year
"""

years_df = pd.read_sql(years_query, engine)
years = [year for year in range(1500, 2024, 5)]

def get_color_from_role(role):
    if pd.isna(role) or role == '':
        return None

    for color_key in normalized_colors.keys():
        if color_key in role:
            return normalized_colors[color_key]
    return None


# Для каждого года создаем карту
for year in years:
    print(f"Processing year: {year}")

    # SQL запрос для получения геоданных за определенный год
    query = """
    SELECT 
        g.id, g.name, g.role, g.description, ST_AsText(g.location) as location, g.role_ru, g.description_ru
    FROM 
        geoobject g
    JOIN 
        geoobjectsimulation gs ON g.id = gs.geo_object_id
    JOIN 
        simulation s ON gs.simulation_id = s.id
    WHERE 
        s.year = %s AND s.mode_id = 1
    """

    # Получаем данные из базы
    data_df = pd.read_sql(query, con=engine, params=(year,))

    print(len(data_df))
    # Преобразуем WKT в геометрии для GeoDataFrame
    geometries = []
    highway_values_list = []
    railway_values_list = []
    waterway_values_list = []

    for _, row in data_df.iterrows():
        role = row['role'] if row['role'] else ''
        location_text = row['location'] if row['location'] else ''

        try:
            if any(location_text.upper().startswith(wkt_type) for wkt_type in
                   ['LINESTRING', 'POLYGON', 'MULTIPOLYGON', 'POINT', 'MULTIPOINT']):
                geom = wkt.loads(location_text)
            else:
                try:
                    geom_json = json.loads(location_text)
                    from shapely.geometry import shape

                    geom = shape(geom_json)
                except:
                    geom = None

            geometries.append(geom)
        except Exception as e:
            print(f"Error parsing geometry for id {row['id']}: {e}")
            print(f"Location string: {location_text[:100]}...")  # Выводим часть строки для диагностики
            geometries.append(None)

    # Создаем GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'id': data_df['id'],
        'name': data_df['name'],
        'role': data_df['role'],
        'geometry': geometries
    }, geometry='geometry')

    # Удаляем строки с пустыми геометриями
    gdf = gdf.dropna(subset=['geometry'])

    # Определяем цвета на основе ролей
    gdf['color'] = gdf['role'].apply(get_color_from_role)

    # Фильтруем только те строки, которым мы смогли назначить цвет
    filtered_gdf = gdf[gdf['color'].notna()].copy()

    # Если нет данных после фильтрации, переходим к следующему году
    if filtered_gdf.empty:
        print(f"No data for year {year} after filtering")
        continue

    # Визуализация
    fig, ax = plt.subplots(figsize=(25, 10))
    filtered_gdf.plot(ax=ax, color=filtered_gdf['color'], linewidth=0.5)

    # Ограничиваем область отображения
    ax.set_xlim(-0.23, 0)  # Longitude
    ax.set_ylim(51.46, 51.6)  # Latitude

    # Убираем заголовок и метки осей
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Сохранение графика в файл
    plt.savefig(f'maps_by_years_image_db/{year}.png', bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(fig)  # Закрываем фигуру, чтобы освободить память
    print(f'maps_by_years_image_db/{year}.png')

print("Processing complete!")

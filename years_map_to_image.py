import geopandas as gpd
import matplotlib.pyplot as plt


highway_values = [
    'bridleway', 'footway', 'motorway', 'pedestrian', 'primary',
    'residential', 'secondary', 'service', 'stairs', 'steps',
    'tertiary', 'track', 'trunk', 'trunk_link', 'unclassified'
]

railway_values = [
    'rail', 'station', 'subway', 'tram'
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
    'tram': (238, 130, 238)  # violet
}

for year in range(1500, 2025, 5):
    print(year)
    geojson_file = f'maps_by_years_geojson/GeoJSON/map_{year} — lines.GeoJSON'  # Укажите путь к вашему файлу
    gdf = gpd.read_file(geojson_file)


    # Фильтруем данные
    filtered_gdf = gdf[gdf['highway'].isin(highway_values) | gdf['railway'].isin(railway_values)].copy()


    def rgb_to_normalized(rgb):
        return tuple(c / 255 for c in rgb)

    normalized_colors = {key: rgb_to_normalized(value) for key, value in colors.items()}

    # Создаем новый столбец для цвета
    filtered_gdf.loc[:, 'color'] = filtered_gdf.apply(
        lambda row: normalized_colors.get(row['highway'], normalized_colors.get(row['railway'], None)), axis=1
    )

    # Визуализация
    fig, ax = plt.subplots(figsize=(100, 100))
    filtered_gdf.plot(ax=ax, color=filtered_gdf['color'], linewidth=2)


    # Ограничиваем область отображения
    ax.set_xlim(-0.25, 0.1)  # Longitude
    ax.set_ylim(51.44, 51.57)  # Latitude

    # Убираем заголовок и метки осей
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Сохранение графика в файл перед показом (можно пропустить, если не нужно)
    plt.savefig(f'maps_by_years_image/{year}.png', bbox_inches='tight', pad_inches=0)  # Сохраните график в файл

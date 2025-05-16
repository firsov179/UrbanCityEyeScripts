from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsVectorFileWriter,
    QgsWkbTypes
)
import os

# Укажите путь к вашему OSM файлу
osm_file_path = 'C:\HSE\Okit\pythonProject1\maps_by_years\map_1500.osm'

# Загрузите OSM файл как векторный слой
osm_layer = QgsVectorLayer(osm_file_path, 'OSM Layer', 'memory')

if not osm_layer.isValid():
    print("Слой не загружен! Проверьте путь к файлу.")
else:
    # Фильтруем только линии
    line_features = [feature for feature in osm_layer.getFeatures() if feature.geometry().wkbType() == QgsWkbTypes.LineGeometry]

    print(f"Найдено линий: {len(line_features)}")  # Выводим количество найденных линий

    if len(line_features) == 0:
        print("Нет линий для сохранения.")
    else:
        # Создаем новый векторный слой для линий
        line_layer = QgsVectorLayer('LineString?crs=EPSG:4326', 'Lines', 'memory')
        line_provider = line_layer.dataProvider()

        # Добавляем атрибуты из исходного слоя
        #line_provider.addAttributes(osm_layer.fields())
        #line_layer.updateFields()

        # Добавляем линии в новый слой
        if line_provider.addFeatures(line_features):
            print("Линии успешно добавлены в новый слой.")
        else:
            print("Ошибка при добавлении линий в новый слой.")

        print(len(line_features))
        print(len(line_provider))
        # Укажите путь для сохранения GeoJSON файла
        output_geojson_path = 'C:\HSE\Okit\pythonProject1\map_1500.geojson'

        # Сохраняем слой в формате GeoJSON
        error = QgsVectorFileWriter.writeAsVectorFormat(
            line_layer,
            output_geojson_path,
            'utf-8',
            line_layer.crs(),
            'GeoJSON'
        )

        if error[0] == QgsVectorFileWriter.NoError:
            print(f"Линии успешно сохранены в {output_geojson_path}")
        else:
            print(f"Ошибка при сохранении файла: {error[1]}")

import cv2
import numpy as np
import json

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

# Загрузка изображения
image_path = 'C:\HSE\Okit\pythonProject1\maps_by_years_image\\2015.png'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
print(image.dtype)

# Проверьте, успешно ли загружено изображение
if image is None:
    raise ValueError(f"Не удалось загрузить изображение по пути: {image_path}")

if image.dtype != np.uint8:
    image = image.astype(np.uint8)

# Список для хранения GeoJSON объектов
features = []

def get_hsv_range(color):
    bgr_color = [color[2], color[1], color[0]]

    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

    lower_bound = np.array([hsv_color[0] - 10, 0, 0])
    upper_bound = np.array([hsv_color[0] + 10, 255, 255])

    return lower_bound, upper_bound


# Размеры изображения
height, width = image.shape[:2]

# Границы координат
lon_min, lon_max = -0.25, 0.1  # Longitude
lat_min, lat_max = 51.44, 51.57  # Latitude

# Функция для преобразования пикселей в координаты
def pixel_to_geo(x, y):
    lon = lon_min + (lon_max - lon_min) * (x / width)
    lat = lat_max - (lat_max - lat_min) * (y / height)
    return lon, lat

kernel = np.ones((5, 5), np.uint8)


for road_type, color in colors.items():
    lower_bound, upper_bound = get_hsv_range(color)
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB), lower_bound, upper_bound)

    # Дилатация маски
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 5, True)
        coords = []
        for point in approx[:, 0]:
            x, y = point
            lon, lat = pixel_to_geo(x, y)
            coords.append([lon, lat])

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "highway": road_type
            }
        }
        features.append(feature)
# Создание GeoJSON структуры
geojson = {
    "type": "FeatureCollection",
    "features": features
}

# Сохранение в файл
with open('output.geojson', 'w') as f:
    json.dump(geojson, f)

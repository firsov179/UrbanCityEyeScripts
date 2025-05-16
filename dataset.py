import requests
import xml.etree.ElementTree as ET


# Функция для получения данных из OSM API
def fetch_osm_data(bbox):
    url = f"https://www.openhistoricalmap.org/api/0.6/map?bbox={bbox}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Error fetching data for bbox {bbox}: {response.status_code}")
        return None


# Функция для разбивки bbox на 9 частей
def split_bbox(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_step = (max_lon - min_lon) / 3
    lat_step = (max_lat - min_lat) / 3

    # Создаем 9 меньших bbox
    bboxes = []
    for i in range(3):
        for j in range(3):
            new_bbox = (
                min_lon + i * lon_step,
                min_lat + j * lat_step,
                min_lon + (i + 1) * lon_step,
                min_lat + (j + 1) * lat_step
            )
            bboxes.append(new_bbox)

    return bboxes


# Функция для объединения OSM данных в один XML файл
def merge_osm_data(osm_data_list):
    merged_root = ET.Element("osm")

    for osm_data in osm_data_list:
        if osm_data is not None:
            root = ET.fromstring(osm_data)
            for element in root:
                merged_root.append(element)

    return ET.tostring(merged_root)


# Основной код
if __name__ == "__main__":
    # Исходный bbox

    bbox = (-0.22, 51.35, 0.1, 51.65) # london
    #bbox = ( 126.8433, 37.4580, 127.2004, 37.6523) # seoul
    #bbox = (2.1746, 48.7671, 2.49463, 48.98934)
    # Разбиваем bbox на 9 частей
    smaller_bboxes = split_bbox(bbox)

    osm_data_list = []

    # Получаем данные для каждой части
    for bbox in smaller_bboxes:
        print("*")
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        osm_data = fetch_osm_data(bbox_str)
        if osm_data:
            osm_data_list.append(osm_data)

    # Объединяем все данные в один OSM файл
    merged_osm = merge_osm_data(osm_data_list)

    # Сохраняем результат в файл
    with open("merged_map_london.osm", "wb") as f:
        f.write(merged_osm)

    print("Данные успешно объединены и сохранены в merged_map.osm")

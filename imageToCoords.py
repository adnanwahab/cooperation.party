from main import imageToCoords
import glob
import json
import re

def get_room_id(url):
    match = re.search(r'rooms/(\d+)', url)
    if match:
        return match.group(1)
    else:
        return None

for file_name in glob('data/airbnb/apt'):
    urls = json.load(open(file_name, 'w'))
    location = file_name.replace('.json', '')
    for apt_url in urls:
        gm_list = json.load(open(f'data/airbnb/gm/{get_room_id(apt_url)}'))
        imageToCoords(gm_list, location, apt_url)
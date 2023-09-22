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

for file_name in glob.glob('data/airbnb/apt/*'):
    print(file_name)
    try: urls = json.load(open(file_name, 'w'))
    except Exception as err: 
        print('oh noe', err, file_name) 
        continue
    location = file_name.replace('.json', '')
    for apt_url in urls:
        print('apt_url')
        gm_list = json.load(open(f'data/airbnb/gm/{get_room_id(apt_url)}'))

  
    imageToCoords(gm_list, location, apt_url)
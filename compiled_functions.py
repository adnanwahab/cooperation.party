
import random
from collections import defaultdict
import concurrent.futures
import h3
import re
import random
import json
import glob
from collections import defaultdict
from shapely.geometry import shape, Point
import random 
import requests
import easyocr
from fastapi import Request, FastAPI
import random
import json 
import subprocess
import json
import os
import youtube_dl
import openai
import re
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from collections import defaultdict
from pydantic import BaseModel
import json
import os.path
# nltk.download('punkt')
# nltk.download('universal_tagset')
# nltk.download('words')
from time import time
import os
import openai
env_var = open('.env').read().split('=')[1]

document_query_cache = json.load(open('data/document_query_cache.json'))

def geo_locate_without_ocr():
    r = requests.get('https://www.airbnb.com/rooms/38858788?adults=1&children=0&enable_m3_private_room=true&infants=0&pets=0&check_in=2024-01-19&check_out=2024-01-24&source_impression_id=p3_1696444189_RgRy6Rskg62fmlPK&previous_page_section_name=1000&federated_search_id=af3a171c-1563-4003-b5da-eaeee13030c0')
    idx = r.text.find('Getting around')
    _ = r.text[idx:idx+800]
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": "you will be provided with html strings - > approximate an address and then return an approximate geo lat long coordinates"
        },
        {
        "role": "user",
        "content": _
        }
    ],
        temperature=0,
        max_tokens=3157,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response

#decorate - cache -> fn + parameters -> stringify+hash the paramers = fn+hash(paramers) = key
#and make it save to filesystem if a parameter is added 
def unstructured_geoSpatial_house_template_query(_):
    if _ in document_query_cache: return document_query_cache[_]
    openai.api_key = env_var
    openai.organization = "org-Yz814AiRJVl9JGvXXiL9ZXPl"
    #find 10 houses and each house is close to the residents favorite preferences (two people like library, two people like atm,  two people like vending_machine,  all of them like bench and they all dislike parking_space but half like bank and the other half prefer clinic and some prefer place_of_worship while others prefer research_institute and some like disco and the others prefer country)
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": "You will be provided with unstructured data, and your task is to parse it into a 2d matrix of p-values from 0 to 1 for each person for a GIS suitability analysis.\n\n{person_1:  {\n        \"yoga\": 0.7,\n        \"kick_boxing\": 0.5,\n        \"rock_climbing\": 0.6,\n        \"wind_surfing\": 1.0,\n        \"bars\": 0.0,\n        \"libraries\": 0.5,\n        \"bookstores\": 0.5,\n        \"appreciation_rate\": 0.7,\n        \"rental_preference\": 0.3,\n        \"disco\": 0.5,\n        \"country\": 0.5\n      }, \nperson_2:  {\n        \"yoga\": 0.7,\n        \"kick_boxing\": 0.5,\n        \"rock_climbing\": 0.6,\n        \"wind_surfing\": 1.0,\n        \"bars\": 0.0,\n        \"libraries\": 0.5,\n        \"bookstores\": 0.5,\n        \"appreciation_rate\": 0.7,\n        \"rental_preference\": 0.3,\n        \"disco\": 0.5,\n        \"country\": 0.5\n      }\n}\nbut repeat for 10 people.  only return json in the format"
        },
        {
        "role": "user",
        "content": _
        }
    ],
        temperature=0,
        max_tokens=3157,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    result = response['choices'][0]['message']['content']
    result = json.loads(result)
    document_query_cache[_] = result
    json.dump(document_query_cache, open('data/document_query_cache.json', 'w+'))
    return result

def get_json_if_possible(apt):
    args = [
        "node",
        "rpc/airbnb_get_img_url.js",
        f'{location}'
    ]
    completed_process = subprocess.run(args)
    imageToCoords(url_list, location, get_room_id(url))

    if os.path.exists(f'data/airbnb/geocoordinates/{get_room_id(apt)}.json'):
        data = json.load(open(f'data/airbnb/geocoordinates/{get_room_id(apt)}.json'))
        if (len(data) > 0): 
            data = data[0]
            data = data.split(':')
            data[0] = float(data[0])
            data[1] = float(data[1])
            return data
        else: return [0,0]
    else:
        return [0, 0]

def find_best_house(apt, documentContext, i):
    city_name = 'Tokyo--Japan'
    def rankApt(personCoefficentPreferences, apt):
        diff = 0
        for key in personCoefficentPreferences:
            if key not in apt: continue
            diff += abs(apt[key] - personCoefficentPreferences[key])
        return diff 
    print('documentContext', documentContext)
    
    personCoefficentPreferences = documentContext['sliders']
    apt_list = json.load(open(f'data/airbnb/apt/{city_name}.json'))[:50]
    return print('not used atm')
    #geocoordinates = [get_json_if_possible(apt, location) for apt in apt_list]
    keys = personCoefficentPreferences.keys()
    apts  = []
    for idx, _ in enumerate(geocoordinates): 
        apt = {
            'url': apt_list[idx],
            'loc': geocoordinates[idx]
        } 
        for key in keys:
            coords = _
            apt[key] = random.random()
        apts.append(apt)

    totals = defaultdict(int)
    for apt in apts: 
        for key in keys: 
            print(apt[key])
            totals[key] += apt[key]

    for apt in apts: 
        for key in keys: 
            if totals[key] == 0: totals[key] += .01
            apt[key] = apt[key] / totals[key]
    return sorted(apts, key=lambda apt: rankApt(personCoefficentPreferences, apt))[0]

def ocrImage(fp):
    reader = easyocr.Reader(['en'])
    if 'ted_search_id' in fp: 
        print('wtf', fp)
        return [139, 35]
    if fp =='ted_search_id=eb732468-761a-45fe-95ee-ce0cd255b52': return print('wtf')
    print(fp)
    extract_info = reader.readtext(fp)
    print(extract_info)
    sorted(extract_info, key=lambda _: _[1])
    if (not extract_info): return False
    return extract_info[0][1]   

def geoCode(address, city):
    accessToken = "pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg"  # Replace with your actual access token
    geocodeUrl = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}%2C%20{city}.json?access_token={accessToken}"
    response = requests.get(geocodeUrl)
    data = response.json()
    if 'features' in data and len(data['features']) > 0:
        location = data['features'][0]['geometry']['coordinates']
        return location

isochroneLibraryCache = {}

def isochroneLibrary(longitude, latitude, documentContext):
    if latitude in isochroneLibraryCache:  return isochroneLibraryCache[latitude]
    latitude = float(latitude)
    longitude = float(longitude) 
    contours_minutes = 15
    contours_minutes = 30
    assert(latitude < 90 and latitude > -90)
    isochrone_url = f'https://api.mapbox.com/isochrone/v1/mapbox/walking/{longitude}%2C{latitude}?contours_minutes={contours_minutes}&polygons=true&denoise=0&generalize=0&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg'
    geojson_data = requests.get(isochrone_url).json()

    coffee_shops = fetch_coffee_shops(longitude, latitude, documentContext['sliders'].keys())
    data = []
    for shop in coffee_shops: 
        if 'lat' not in shop or 'lon' not in shop: 
            continue
        point_to_check = Point(shop['lon'], shop['lat'])
        for feature in geojson_data['features']:
            polygon = shape(feature['geometry'])
            if polygon.contains(point_to_check):
                data.append(shop)
    if len(data) > 0:
        isochroneLibraryCache[latitude] = [data, geojson_data, latitude, longitude] 
        return [data, geojson_data, latitude, longitude]
    else : return False

def imageToCoords(url_list, location='_', apt_url='_'):
    fp = f'data/airbnb/geocoordinates/{apt_url}.json'
    if os.path.exists(fp): return json.load(open(fp, 'r'))
    if len(url_list) < 1: return [0, 0]
    #print(url_list)
    cache = set()
    print(fp)
    for _ in url_list[:18]:
        response = requests.get(_)
        if response.status_code == 200:
            with open('tmp/'+_[-50:-1], 'wb') as f:
                f.write(response.content)
        ocr = ocrImage(_[-50:-1])
        if not ocr: continue
        coords = geoCode(ocr, location)
        if not coords: continue
        cache.add(str(coords[0]) + ':' + str(coords[1]))
        if coords: break
    json.dump(list(cache), open(fp, 'w'))
    return list(cache)

def get_room_id(url):
    match = re.search(r'rooms/(\d+)', url)
    if match:
        return match.group(1)
    else:
        return None

def map_of_all_airbnbs(_,__, i):
    cities = [json.load(open(_)) for _ in  glob.glob('data/airbnb/apt/*')]
    geoCode = [json.load(open(f'data/airbnb/geocoordinates/{get_room_id(listing)}.json')) 
               for city in cities 
               for listing in city 
               if os.path.exists(f'data/airbnb/geocoordinates/{get_room_id(listing)}.json')
               ]
    return {'data': geoCode, 'component': '<map>', 'geoCoordCache': geoCoordCache }

def filter_by_poi(_, documentContext, sentence):
    poi = sentence.strip().split(' ')[2]
    if 'sliders' not in documentContext: documentContext['sliders'] = {}
    if poi not in documentContext['sliders']: documentContext['sliders'][poi] = .5
    if (_ == 'hello-world'): return {'component': '<slider>', 'data': _, 'label': poi}
    if (type(_) is not list): _ = _['data']
    return {'component': '<slider>', 'data': _, 'label': poi}

def groupBySimilarity(sentences, documents, i):
    from sentence_transformers import SentenceTransformer,util
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentences = [item for sublist in sentences for item in sublist]
    encodings = model.encode(sentences, convert_to_tensor=True, device='cpu')
    clusters = util.community_detection(encodings, min_community_size=1, threshold=0.55)
    sentenceClusters = []
    for s in clusters:
        sentenceCluster = []
        for id, sentenceId in enumerate(s):
            sentenceCluster.append(sentences[sentenceId])
        sentenceClusters.append(sentenceCluster)
    
    return sentenceClusters

def continentRadio(_, __, i):
    from ipynb.fs.defs.geospatial import getCityList
    return {
        'key':'continent',
        'data': ['Europe', 'North America', 'Asia', 'South America', 'Africa', 'Australia and Oceania', 'Others/Islands'], 
        'component': '<Radio>'
        }

def cityRadio(_, __, i):
    from ipynb.fs.defs.geospatial import getCityList
    return {'key':'city','data': getCityList(), 'component': '<Radio>'}

def getAirbnbs(_, componentData, i):
    from ipynb.fs.defs.geospatial import getAllApt_
    if 'city' not in componentData: return 'hello-world'
    location = componentData['city']
    location = location.replace(', ', '--')
    if location == '': return 'hello-world'
    fp = f'data/airbnb/apt/{location}.json'
    fp_gm = lambda apt: f'/data/airbnb/gm/{get_room_id(apt)}.json'
    # if os.path.exists(fp):
    #     apts = json.load(open(fp, 'r'))
    #     gm = [os.path.exists(fp_gm(apt)) for apt in apts]
    #     gm = [_ for _ in gm if _ == True]
    #     print(len(apts) == len(gm), 'all apt found')
    #     if len(apts) == len(gm) and len(gm) != 0: return [apt['link'] for apt in apts]
    args = [
        "node",
        "rpc/getAptInCity.js",
        location
    ]
    #completed_process = subprocess.run(args)
    args = [
        "node",
        "rpc/airbnb_get_img_url.js",
        f'data/airbnb/apt/{location}.json'
    ]
    #completed_process = subprocess.run(args)
    #print(location)
    apts = json.load(open(fp, 'r'))

    return [apt for apt in apts] #return component:list -> keep consistent no implicit schemas

def filter_by_distance_to_shopping_store(airbnbs, documentContext, i):
    if (type(airbnbs) is dict): airbnbs = airbnbs['data']
    #print('airbnbs!', airbnbs)
    #return ['asdf', 'hello']
    #print(airbnbs, airbnbs)
    #return ['asdf', 'hello']
    #SSreturn airbnbs[:10]
    #subprocess.run(['node', 'airbnb_get_img_url.js', 'jaipur--india_apt.json'])
    if airbnbs =='hello-world': return 'hello world'
    #writes to listing_url.json
    #for each listing_url -> get img url
    #for each img url -> OCR
    #for each OCR -> geocode
    #for each geocode -> get nearby shopping stores
    #sort list of appartments by distance to shopping store
    #make better
    #imageToCoords() #apt_url -> coordinate
    #getPlacesOfInterest() #coordiante -> get distance to shopping store
    #print ('airbnbs', airbnbs)
    #for each apt
    #return airbnbs[:10]
    #document -> compile to fn -> each one 
    #please one night of peace and quiet it and i promise you'll see code you couldn't imagine. no matter how many decades you've written code. i promise.
    cache = {}
    def doesExist(url):
        if url not in cache: 
            cache[url]  = True
            return True
        return False

    def gm_get(fp):
        if os.path.exists(fp): return json.load(open(fp))
        else: return []

    airbnbs = [apt for apt in airbnbs if doesExist(apt)]
    gm_urls = [gm_get(fp) for fp in [f"data/airbnb/gm/{get_room_id(apt_url)}.json" for apt_url in airbnbs]]
    print('gm_urls', gm_urls)
    geoCoordinates = [imageToCoords(_, documentContext['city'], get_room_id(airbnbs[idx]) ) for idx, _ in enumerate(gm_urls[:6])]

    geoCoordinates = [coord[0].split(':') for coord in geoCoordinates if len(coord) > 1]
    _ = [isochroneLibrary(pt[0], pt[1], documentContext) for idx, pt in enumerate(geoCoordinates)]

    return [_ for _ in _ if _ != False]
# def createDocumentContext():
#     liveUpdateWhenWrittenTo = {} #client reads from val, push update to client w/ SSE
#     _ = {}
#     savedGet = _.__getitem__
#     savedWrite = _.__setitem__
#     def registerWatch(key):
#         liveUpdateWhenWrittenTo[key] = registerWatch.__closure__
#         return savedGet(key)
#     def registerWatch(key, value):
#         liveUpdateWhenWrittenTo[key]
#         return savedWrite(key, value)
#     _.__getitem__ = registerWatch
#     _.__setitem__ = rerunGetters
#     return _

def getYoutube(url, i):
    youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s'}).download(['https://www.youtube.com/watch?v=a02S4yHHKEw&ab_channel=AllThatOfficial'])
    audio_file= open("audio.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    open('transcript.txt', 'w').write(transcript)
    return '<audio src="audio.mp3">'

def poll(_, second, i):
    return 'lots of cool polling data'
    return open('./poll.json', 'r').read()

hasRendered = False
hasRenderedContent = False
def arxiv (_, sentence, i):
    global hasRendered, hasRenderedContent  # Declare as global to modify
    import pdfplumber
    import glob
    fileList = glob.glob('./*.pdf')[:3]
    content = []
    if hasRendered: return hasRenderedContent
    for f in fileList:
        with pdfplumber.open(f) as pdf:
            # Loop through each page
            for i, page in enumerate(pdf.pages):
            # Extract text from the page
                text = page.extract_text()
                content.append(text)
            #print(f'Content from page {i + 1}:\n{text}')
    hasRendered = True
    hasRenderedContent = content
    return content

def trees_histogram(_, sentence, i):
    from ipynb.fs.defs.geospatial import trees_histogram
    return trees_histogram()

def twitch_comments(_, sentence, i):
    return json.load(open('./data/twitch.json', 'r'))

def getTopics(sentences, sentence, i):
    counts = defaultdict(int)
    for sentence in sentences:
        for word in sentence.split(' '):
            counts[word] += 1
    topics = []
    for k in counts:
        if counts[k] > 2:
            topics.append(k)
    return topics

def trees_map(_, sentence, i):
    from ipynb.fs.defs.geospatial import trees_map
    return trees_map()[:100000]

def pokemon(_, __, i):
    from ipynb.fs.defs.Pokemon_Dota_Arxiv import generate_team
    return generate_team()

def satellite_housing(_, sentence):
    requests.get('https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v11/static/-122.4241,x.78,14.25,0,60/600x600?access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg')
    return 'for each satellite images in area find anything that matches criteria'

geoCoordCache = {}
def fetch_coffee_shops(longitude, latitude, amenities = []):
    if round(longitude, 1) in geoCoordCache: 
        #print('WE GOT THE CACHE', len(geoCoordCache[round(longitude, 1)]))
        return geoCoordCache[round(longitude, 1)]
    # if (os.path.exists(f'data/airbnb/poi/{longitude}_{latitude}_places.json')):
    # return json.load(open(f'data/airbnb/poi/{longitude}_{latitude}_places.json', 'r'))
    places = []
    for i in amenities:
        query = f"""
        [out:json][timeout:25];
        (
            node["amenity"="{i}"]({latitude - 0.01},{longitude - 0.01},{latitude + 0.01},{longitude + 0.01});
        );
        out body;
        """ 
        overpass_url = "https://overpass-api.de/api/interpreter"
        response = requests.get(overpass_url, params={'data': query})
        #print(response.status_code, longitude, latitude, amenities)
        if response.status_code == 200:
            data = response.json()
            coffee_shops = data['elements']
            places += coffee_shops
    if len(places) > 0:
        geoCoordCache[round(longitude, 1)] = places
    #json.dump(places, open(f'data/airbnb/poi/{listing}_places.json', 'w'))
    return places

def storeAggregation(h3_cells, columns):
    _ = {}
    for col in columns: _[col] = {}
    for cell in h3_cells:
        for col in columns: _[col][cell] = h3_cells[cell][col]
    for col in columns:
        json.dump(_[col], open(f'data/airbnb/h3_poi/{col}.json', 'w+'))
    

def retrieveAggregation(columns):
    _ = {}
    for col in columns:
        if not os.path.exists(f'data/airbnb/h3_poi/{col}.json'): continue
        cell_poi_count = json.load(open(f'data/airbnb/h3_poi/{col}.json'))
        for cell in cell_poi_count:
            if cell not in _:
                _[cell] = {}
            _[cell][col] = cell_poi_count[cell]
    return _

def fetch_coffee_shops(longitude, latitude, amenities=''):
    places = []
    query = f"""
    [out:json][timeout:25];
    (
        node["amenity"="{amenities}"]({latitude - 0.01},{longitude - 0.01},{latitude + 0.01},{longitude + 0.01});
    );
    out body;
    """ 
    overpass_url = "https://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={'data': query})

    if response.status_code == 200:
        data = response.json()
 
        coffee_shops = data['elements']
        places += coffee_shops
    if len(places) > 0:
        geoCoordCache[round(longitude, 2)] = places
    return places
def get_room_id(url):
    match = re.search(r'rooms/(\d+)', url)
    if match:
        return match.group(1)
    else:
        return None

cities = {
    "New-York-City--USA": [
        40.7128,
        -74.006
    ],
    "San-Francisco--USA": [
        37.7749,
        -122.4194
    ],
    "Vancouver--Canada": [
        49.2827,
        -123.1207
    ],
    "New-Orleans--USA": [
        29.9511,
        -90.0715
    ],
    "Los-Angeles--USA": [
        34.0522,
        -118.2437
    ],
    "Chicago--USA": [
        41.8781,
        -87.6298
    ],
    "Toronto--Canada": [
        43.6532,
        -79.3832
    ],
    "Mexico-City--Mexico": [
        19.4326,
        -99.1332
    ],
    "Montreal--Canada": [
        45.5017,
        -73.5673
    ],
    "Boston--USA": [
        42.3601,
        -71.0589
    ],
    "Miami--USA": [
        25.7617,
        -80.1918
    ],
    "Austin--USA": [
        30.2672,
        -97.7431
    ],
    "Quebec-City--Canada": [
        46.8139,
        -71.2082
    ],
    "Seattle--USA": [
        47.6062,
        -122.3321
    ],
    "Nashville--USA": [
        36.1627,
        -86.7816
    ],
    "Tokyo--Japan": [
        35.6895,
        139.6917
    ],
    "Kyoto--Japan": [
        35.0116,
        135.7681
    ],
    "Bangkok--Thailand": [
        13.7563,
        100.5018
    ],
    "Hong-Kong--China": [
        22.3193,
        114.1694
    ],
    "Singapore": [
        1.3521,
        103.8198
    ],
    "Seoul--South-Korea": [
        37.5665,
        126.978
    ],
    "Beijing--China": [
        39.9042,
        116.4074
    ],
    "Dubai--UAE": [
        25.276987,
        55.296249
    ],
    "Taipei--Taiwan": [
        25.033,
        121.5654
    ],
    "Istanbul--Turkey": [
        41.0082,
        28.9784
    ],
    "Hanoi--Vietnam": [
        21.0285,
        105.8544
    ],
    "Mumbai--India": [
        19.076,
        72.8777
    ],
    "Kuala-Lumpur--Malaysia": [
        3.139,
        101.6869
    ],
    "Jaipur--India": [
        26.9124,
        75.7873
    ],
    "Rio-de-Janeiro--Brazil": [
        -22.9068,
        -43.1729
    ],
    "Buenos-Aires--Argentina": [
        -34.6037,
        -58.3816
    ],
    "Cartagena--Colombia": [
        10.391,
        -75.4794
    ],
    "Lima--Peru": [
        -12.0464,
        -77.0428
    ],
    "Santiago--Chile": [
        -33.4489,
        -70.6693
    ],
    "Cusco--Peru": [
        -13.5319,
        -71.9675
    ],
    "Medellín--Colombia": [
        6.2476,
        -75.5709
    ],
    "Quito--Ecuador": [
        -0.1807,
        -78.4678
    ],
    "Montevideo--Uruguay": [
        -34.9011,
        -56.1911
    ],
    "Bogota--Colombia": [
        4.71,
        -74.0721
    ],
    "Cape-Town--South-Africa": [
        -33.9249,
        18.4241
    ],
    "Marrakech--Morocco": [
        31.6295,
        -7.9811
    ],
    "Cairo--Egypt": [
        30.8025,
        31.2357
    ],
    "Dakar--Senegal": [
        14.6928,
        -17.4467
    ],
    "Zanzibar-City--Tanzania": [
        -6.1659,
        39.2026
    ],
    "Accra--Ghana": [
        5.6037,
        -0.1869
    ],
    "Addis-Ababa--Ethiopia": [
        9.03,
        38.74
    ],
    "Victoria-Falls--Zimbabwe/Zambia": [
        -17.9243,
        25.8572
    ],
    "Nairobi--Kenya": [
        -1.286389,
        36.817223
    ],
    "Tunis--Tunisia": [
        36.8065,
        10.1815
    ],
    "Sydney--Australia": [
        -33.8688,
        151.2093
    ],
    "Melbourne--Australia": [
        -37.8136,
        144.9631
    ],
    "Auckland--New-Zealand": [
        -36.8485,
        174.7633
    ],
    "Wellington--New-Zealand": [
        -41.2865,
        174.7762
    ],
    "Brisbane--Australia": [
        -27.4698,
        153.0251
    ],
    "Honolulu--Hawaii--USA": [
        21.3069,
        -157.8583
    ],
    "Bali--Indonesia": [
        -8.3405,
        115.092
    ],
    "Santorini--Greece": [
        36.3932,
        25.4615
    ],
    "Maldives-(Male)": [
        4.1755,
        73.5093
    ],
    "Phuket--Thailand": [
        7.8804,
        98.3923
    ],
    "Ibiza--Spain": [
        38.9067,
        1.4206
    ],
    "Seychelles-(Victoria)": [
        -4.6191,
        55.4513
    ],
    "Havana--Cuba": [
        23.1136,
        -82.3666
    ],
    "Punta-Cana--Dominican-Republic": [
        18.582,
        -68.4055
    ],
    "Dubrovnik--Croatia": [
        42.6507,
        18.0944
    ],
    "Ljubljana--Slovenia": [
        46.0569,
        14.5058
    ],
    "Tallinn--Estonia": [
        59.437,
        24.7536
    ],
    "Riga--Latvia": [
        56.9496,
        24.1052
    ],
    "Sarajevo--Bosnia-and-Herzegovina": [
        43.8563,
        18.4131
    ],
    "Vilnius--Lithuania": [
        54.6872,
        25.2797
    ],
    "Tbilisi--Georgia": [
        41.7151,
        44.8271
    ],
    "Yerevan--Armenia": [
        40.1792,
        44.4991
    ],
    "Baku--Azerbaijan": [
        40.4093,
        49.8671
    ],
    "Belgrade--Serbia": [
        44.7866,
        20.4489
    ],
    "Skopje--North-Macedonia": [
        41.9973,
        21.428
    ],
    "Banff--Canada": [
        51.1784,
        -115.5708
    ],
    "Queenstown--New-Zealand": [
        -45.0312,
        168.6626
    ],
    "Reykjavik-(as-a-gateway-to-Icelandic-nature)": [
        64.1466,
        -21.9426
    ],
    "Ushuaia--Argentina-(Gateway-to-Antarctica)": [
        -54.8019,
        -68.303
    ],
    "Kathmandu--Nepal-(Gateway-to-the-Himalayas)": [
        27.7172,
        85.324
    ]
}

#map apturl to city
#map city to lat long
#add a noise value 
# when you have internet access -> 
# polyfill the precise location using screenshot -> 
# cache the OCR so you have a hashmap of strings to geo coordinates
# dont just use the longest string, try all of them
# make that all a background process 
def get_lat_long(url, location): 
    apt = get_room_id(url)
    apt = get_room_id(url)
    _ = cities[location.replace('.json', '')].copy()

    _[0] += random.random() * .1
    _[1] += random.random() * .1
    #print('_lat_long', _)
    return _
    if (not os.path.exists(f'data/airbnb/geocoordinates/{apt}.json')):
        args = [
            "node",
            "rpc/airbnb_get_img_url.js",
            f'{location}'
        ]
        #completed_process = subprocess.run(args)
        #return [35, 139]
        if os.path.exists(f'data/airbnb/geocoordinates/{apt}.json'):
            url_list = json.load(open(f'data/airbnb/gm/{apt}.json'))
        else: url_list = []
        return imageToCoords(url_list, location, get_room_id(url))
        
    data = json.load(open(f'data/airbnb/geocoordinates/{apt}.json'))
    if len(data) == 0: data = [0,0]
    else: 
        data = data[0]
        data = data.split(':')
    data = [float(data[1]), float(data[0])]
    return data

def _housing(url, h3_cells,idx , loc):
    shit = loc
    lat = shit[0]
    lng = shit[1]
    h3_cell_id = h3.geo_to_h3(lat, lng, resolution=7)
    _coefficents = h3_cells[h3_cell_id]
    ret = {
        'url': url,
        'location': shit,
        'h3_cell': h3_cell_id,  
        'coefficents': _coefficents
    }
    return ret 

def key_function(apt, user_preferences, idx):
    dist = 0
    for idx, key in enumerate(apt['coefficents']):
        dist += apt['coefficents'][key] - user_preferences[idx]
    return dist
    
def make_fetch_shops_for_cell(poi_names, h3_cells):
    def fetch_shops_for_cell(hex_id):
        results = h3_cells[hex_id]
        ll = h3.h3_to_geo(hex_id)
     
        for key in poi_names:
            if key not in results or results[key] == 0:
                val = len(fetch_coffee_shops(ll[1], ll[0], key))
                if val == 0: val = .0000000000000001
                results[key] = val
        return (hex_id, results)
    return fetch_shops_for_cell

def aggregate_poi_in_h3_cell(h3_cells, fn):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for hex_id, results in executor.map(fn, h3_cells.keys()):
            #print(hex_id)
            pass
    return h3_cells
poi_names = [s.strip() for s in json.load(open('./poi_names.json'))]
#print(poi_names)
import random
preferences = {}

coefficents = preferences
people_names = 'fred bob sally panda velma alref wilbur steven dan michael'.split(' ')
people_preferences = {}

for person in people_names: people_preferences[person] = [random.random() for _ in range(10)]
people_preferences = {
 'fred': [1, 0, 0, 0, 0, 0, 0, 0, 0, .5],
 'bob': [0, 1, 0, 0, 0, 0, 0, 0, .5, 0],
 'sally': [0, 1, 0, 0, 0, 0, .5, 0, 0, 0],
 'panda': [0, 0, 1, 0, 0, .5, 0, 0, 0, 0],
 'velma': [0, 0, 0, 1, .5, 0, 0, 0, 0, 0],
 'alref': [0, 0, 0, 0, .5, 1, 0, 0, 0, 0],
 'wilbur': [0, 0, .5, 0, 1, 0, 0, 0, 0, 0],
 'steven': [0, .5, 0, 0, 0, 1, 0, 0, 0, 0],
 'dan': [0, 0, .5, 0, 0, 0, 0, 0, 1, 0],
 'michael': [.5, 0, 0, 0, 0, 0, 0, 0, 1, 0]
 }
def max_index(lst): return lst.index(max(lst))
def second_largest_index(lst):
    index_of_max = lst.index(max(lst))
    lst_without_max = lst[:index_of_max] + lst[index_of_max+1:]
    second_largest = max(lst_without_max)
    return lst.index(second_largest)

import copy
def getIsoChrone(point):
    # #return {'features': [{'properties': {'fill-opacity': 0.33, 'fillColor': '#4286f4', 'opacity': 0.33, 'fill': '#4286f4', 'fillOpacity': 0.33, 'color': '#4286f4', 'contour': 60, 'metric': 'time'}, 'geometry': {'coordinates': [[[139.378, 36.036366], [139.374782, 36.036], [139.376918, 36.034918], [139.38053, 36.02653], [139.381046, 36.016], [139.38441, 36.01441], [139.385107, 36.012], [139.384368, 35.992], [139.379885, 35.988], [139.379111, 35.982889], [139.360847, 35.972], [139.358915, 35.967085], [139.359169, 35.961168], [139.362, 35.956539], [139.375125, 35.952], [139.369994, 35.940006], [139.371094, 35.928], [139.368362, 35.92], [139.373146, 35.916], [139.375134, 35.909134], [139.382, 35.906843], [139.387277, 35.9], [139.384523, 35.897477], [139.382917, 35.892], [139.38381, 35.88219], [139.377272, 35.880729], [139.372972, 35.876], [139.372998, 35.872], [139.376827, 35.868], [139.377976, 35.859976], [139.379255, 35.876], [139.382, 35.87845], [139.387016, 35.878984], [139.388554, 35.885446], [139.392128, 35.888], [139.388211, 35.892], [139.392262, 35.896], [139.394, 35.905446], [139.400008, 35.912], [139.402, 35.912882], [139.406, 35.910136], [139.414044, 35.916044], [139.413521, 35.896479], [139.403675, 35.888], [139.403766, 35.88], [139.406436, 35.872], [139.40552, 35.86848], [139.402652, 35.867348], [139.401035, 35.864], [139.401944, 35.855944], [139.41, 35.857227], [139.414, 35.855322], [139.434, 35.854501], [139.438, 35.857142], [139.443043, 35.853043], [139.446, 35.852551], [139.450343, 35.856343], [139.458, 35.855388], [139.465868, 35.852], [139.458, 35.844829], [139.450576, 35.847424], [139.448392, 35.845608], [139.450852, 35.84], [139.450158, 35.832], [139.449628, 35.828], [139.446152, 35.823848], [139.445603, 35.816397], [139.450972, 35.812], [139.447916, 35.806084], [139.443337, 35.804], [139.441669, 35.796], [139.450843, 35.788], [139.452311, 35.784], [139.457753, 35.783753], [139.461403, 35.779403], [139.463719, 35.776], [139.461385, 35.772], [139.458, 35.77003], [139.45, 35.769223], [139.442, 35.76474], [139.43572, 35.76], [139.43449, 35.75551], [139.432807, 35.756], [139.43276, 35.76], [139.43, 35.761912], [139.424459, 35.761541], [139.422, 35.759439], [139.414, 35.761965], [139.395332, 35.756], [139.394, 35.751664], [139.39223, 35.75423], [139.388899, 35.754899], [139.378, 35.762957], [139.37, 35.763063], [139.366905, 35.764905], [139.365177, 35.772], [139.367125, 35.776], [139.363659, 35.78], [139.369244, 35.784], [139.371269, 35.788], [139.370306, 35.792], [139.366, 35.797182], [139.356907, 35.797093], [139.35355, 35.79555], [139.344181, 35.8], [139.345168, 35.804832], [139.358, 35.805524], [139.366, 35.814233], [139.37, 35.81065], [139.378, 35.810587], [139.379352, 35.812], [139.374665, 35.836], [139.370752, 35.836752], [139.37005, 35.84405], [139.366, 35.844456], [139.366047, 35.843953], [139.369857, 35.843857], [139.369739, 35.835739], [139.372792, 35.834792], [139.373184, 35.832], [139.372294, 35.824], [139.367497, 35.818503], [139.360973, 35.817027], [139.354, 35.809859], [139.342, 35.810301], [139.338, 35.807244], [139.330478, 35.807522], [139.33, 35.803651], [139.326, 35.807104], [139.318, 35.80919], [139.306, 35.806006], [139.301766, 35.796], [139.295545, 35.794455], [139.294972, 35.784], [139.290249, 35.779751], [139.282, 35.777439], [139.278, 35.780042], [139.273188, 35.78], [139.27302, 35.776], [139.283363, 35.768], [139.282223, 35.759777], [139.280475, 35.761525], [139.281752, 35.763752], [139.278, 35.762922], [139.274, 35.758934], [139.268157, 35.757843], [139.262, 35.750848], [139.258, 35.755743], [139.253162, 35.755162], [139.25, 35.760947], [139.242, 35.755677], [139.237436, 35.756], [139.237156, 35.752], [139.234, 35.750438], [139.227552, 35.753552], [139.226, 35.758019], [139.22043, 35.75757], [139.217686, 35.756], [139.223324, 35.753324], [139.22391, 35.74609], [139.218, 35.742851], [139.215615, 35.744], [139.215919, 35.745919], [139.207835, 35.744], [139.2116, 35.732], [139.21, 35.730822], [139.206553, 35.732553], [139.197816, 35.732], [139.20368, 35.72168], [139.214, 35.722257], [139.218609, 35.716609], [139.218465, 35.711535], [139.215275, 35.710725], [139.210908, 35.704], [139.218393, 35.700393], [139.219347, 35.697347], [139.222, 35.696663], [139.228979, 35.697021], [139.23, 35.700363], [139.230665, 35.696664], [139.234208, 35.696208], [139.234386, 35.692], [139.231002, 35.690998], [139.23, 35.687663], [139.229249, 35.691249], [139.221678, 35.691678], [139.220897, 35.694897], [139.209431, 35.695431], [139.206, 35.698535], [139.20431, 35.69769], [139.202764, 35.692], [139.207218, 35.685218], [139.222366, 35.684], [139.218822, 35.683178], [139.218287, 35.679713], [139.215024, 35.678976], [139.214, 35.675383], [139.209446, 35.675446], [139.206, 35.678989], [139.193368, 35.676632], [139.193357, 35.675357], [139.202814, 35.672], [139.192467, 35.669533], [139.190129, 35.664], [139.194, 35.662539], [139.198, 35.666392], [139.206, 35.664849], [139.208136, 35.658137], [139.21, 35.657205], [139.215457, 35.658543], [139.218807, 35.664807], [139.222, 35.660914], [139.230753, 35.660753], [139.234, 35.656181], [139.241711, 35.660289], [139.246, 35.66032], [139.243782, 35.656], [139.246086, 35.652086], [139.246123, 35.643877], [139.244743, 35.646743], [139.238772, 35.647228], [139.23412, 35.64388], [139.214403, 35.643596], [139.214072, 35.639928], [139.21, 35.639787], [139.202, 35.646795], [139.200124, 35.645876], [139.198, 35.639259], [139.191073, 35.638927], [139.18375, 35.636], [139.186, 35.634028], [139.198265, 35.632265], [139.191412, 35.630588], [139.190178, 35.623822], [139.185816, 35.623816], [139.18528, 35.62728], [139.169716, 35.627716], [139.166, 35.633613], [139.159241, 35.630759], [139.158, 35.623523], [139.156859, 35.626859], [139.153453, 35.628547], [139.161466, 35.636], [139.154, 35.638609], [139.150717, 35.635283], [139.145424, 35.635424], [139.142, 35.64771], [139.134, 35.641368], [139.126992, 35.648992], [139.122, 35.651279], [139.119731, 35.646269], [139.114943, 35.643056], [139.11, 35.642845], [139.10413, 35.648], [139.102, 35.653488], [139.096451, 35.653549], [139.091959, 35.648], [139.096324, 35.64], [139.094, 35.638099], [139.088235, 35.636], [139.074, 35.634712], [139.057418, 35.635417], [139.054, 35.637411], [139.04291, 35.63509], [139.038, 35.630689], [139.029284, 35.628716], [139.028217, 35.624], [139.031671, 35.62], [139.026, 35.615081], [139.014, 35.613289], [139.01, 35.614814], [139.006, 35.612742], [139.003332, 35.613332], [139.002528, 35.616528], [138.997616, 35.615616], [139.00075, 35.61475], [139.002, 35.611249], [139.018, 35.611015], [139.026, 35.608919], [139.030933, 35.611067], [139.034, 35.6146], [139.038, 35.612808], [139.046, 35.617404], [139.05, 35.616523], [139.059325, 35.618674], [139.061547, 35.62], [139.053768, 35.628232], [139.086289, 35.628], [139.088446, 35.62], [139.082, 35.614267], [139.078, 35.616595], [139.073506, 35.616], [139.076995, 35.610995], [139.086, 35.60968], [139.094, 35.613586], [139.106, 35.61272], [139.11, 35.609005], [139.114, 35.608729], [139.114, 35.606162], [139.109645, 35.603645], [139.114, 35.601758], [139.11939, 35.596], [139.112766, 35.592], [139.114, 35.588075], [139.128097, 35.592], [139.124874, 35.6], [139.127294, 35.601294], [139.129715, 35.600285], [139.13, 35.602561], [139.132126, 35.598126], [139.138, 35.59559], [139.141456, 35.600544], [139.146, 35.600614], [139.146595, 35.595405], [139.143097, 35.592], [139.143088, 35.586912], [139.138354, 35.584], [139.142, 35.581052], [139.146607, 35.580607], [139.15, 35.572945], [139.152217, 35.577783], [139.156063, 35.58], [139.156262, 35.584], [139.153197, 35.587197], [139.153522, 35.592478], [139.162, 35.594079], [139.166, 35.589596], [139.174, 35.588992], [139.178, 35.597111], [139.182, 35.596863], [139.186, 35.593113], [139.194, 35.593285], [139.197628, 35.587628], [139.201034, 35.588966], [139.206484, 35.588484], [139.214, 35.58189], [139.219495, 35.581495], [139.218964, 35.571036], [139.206212, 35.564], [139.216148, 35.562148], [139.222403, 35.563597], [139.226, 35.573016], [139.228004, 35.568], [139.234566, 35.564], [139.23, 35.558208], [139.220276, 35.556], [139.218, 35.550695], [139.209657, 35.556343], [139.209712, 35.551712], [139.21239, 35.55039], [139.213039, 35.548], [139.213277, 35.539277], [139.218, 35.53745], [139.230087, 35.543913], [139.229057, 35.552943], [139.239699, 35.554301], [139.242, 35.560549], [139.242884, 35.556], [139.246555, 35.552555], [139.246471, 35.547529], [139.242, 35.547188], [139.234, 35.538537], [139.228537, 35.537463], [139.23, 35.531181], [139.232732, 35.533268], [139.239667, 35.534333], [139.245745, 35.540255], [139.25, 35.540233], [139.25004, 35.53604], [139.254, 35.534845], [139.256787, 35.530787], [139.274, 35.52487], [139.278, 35.526693], [139.282, 35.525274], [139.286, 35.528474], [139.294, 35.526395], [139.298, 35.528886], [139.302, 35.524811], [139.31, 35.526967], [139.322, 35.524183], [139.324609, 35.525391], [139.324002, 35.532], [139.325727, 35.532272], [139.326168, 35.528168], [139.331142, 35.525142], [139.342957, 35.524], [139.338524, 35.516], [139.343879, 35.509879], [139.343213, 35.504], [139.349253, 35.5], [139.355994, 35.492], [139.36093, 35.476], [139.364429, 35.472], [139.36583, 35.46], [139.37, 35.460711], [139.372897, 35.438897], [139.37594, 35.43394], [139.375821, 35.430179], [139.374, 35.427348], [139.361733, 35.427733], [139.358, 35.430591], [139.351014, 35.426986], [139.334, 35.425454], [139.328225, 35.417775], [139.321612, 35.416388], [139.321011, 35.412989], [139.306469, 35.412469], [139.306008, 35.416008], [139.30413, 35.416], [139.305968, 35.415968], [139.30573, 35.40827], [139.299302, 35.408], [139.306064, 35.407936], [139.306457, 35.411543], [139.322, 35.409741], [139.32366, 35.41434], [139.33076, 35.41524], [139.338, 35.422178], [139.347072, 35.42], [139.346746, 35.415254], [139.343661, 35.410339], [139.338, 35.40885], [139.336667, 35.405333], [139.33138, 35.404], [139.333363, 35.403363], [139.334, 35.395126], [139.335015, 35.402985], [139.35, 35.402984], [139.352444, 35.405556], [139.355815, 35.405815], [139.35964, 35.404], [139.362, 35.397991], [139.365088, 35.404], [139.37, 35.405545], [139.371612, 35.4], [139.376967, 35.396], [139.37752, 35.37648], [139.373843, 35.376157], [139.374, 35.370158], [139.374576, 35.375424], [139.378469, 35.376], [139.381203, 35.388], [139.37956, 35.4], [139.375693, 35.404], [139.376837, 35.421163], [139.395594, 35.422407], [139.401664, 35.428336], [139.405265, 35.428735], [139.406, 35.430196], [139.410347, 35.428], [139.418, 35.419359], [139.422, 35.421745], [139.434, 35.421239], [139.438, 35.423372], [139.44153, 35.42847], [139.442973, 35.436], [139.447411, 35.44], [139.442553, 35.448553], [139.433912, 35.455912], [139.443867, 35.46], [139.442917, 35.464], [139.450679, 35.464679], [139.458, 35.456775], [139.462, 35.455695], [139.466, 35.450495], [139.479565, 35.441565], [139.482, 35.44089], [139.494, 35.44359], [139.502, 35.442948], [139.504084, 35.44], [139.500663, 35.436], [139.501596, 35.432], [139.49822, 35.428], [139.497991, 35.424], [139.4949, 35.42], [139.49, 35.419357], [139.485295, 35.416], [139.492171, 35.408], [139.490259, 35.400259], [139.494, 35.398451], [139.496892, 35.398892], [139.498, 35.396656], [139.492259, 35.392], [139.496884, 35.386884], [139.500471, 35.370471], [139.506, 35.367713], [139.514, 35.369121], [139.522, 35.364866], [139.527532, 35.366468], [139.531127, 35.370873], [139.542, 35.37216], [139.55, 35.378148], [139.555771, 35.372], [139.557782, 35.363782], [139.566, 35.363716], [139.57, 35.357037], [139.578, 35.354684], [139.582153, 35.355847], [139.581912, 35.360088], [139.591362, 35.36], [139.594521, 35.356], [139.592044, 35.353956], [139.59097, 35.34703], [139.588158, 35.345842], [139.586, 35.341031], [139.581643, 35.34], [139.583544, 35.336], [139.58843, 35.332], [139.588299, 35.328], [139.591032, 35.325032], [139.603316, 35.317316], [139.604906, 35.308], [139.612063, 35.302063], [139.616672, 35.292], [139.621518, 35.291518], [139.622, 35.284288], [139.622285, 35.291715], [139.619184, 35.293184], [139.618867, 35.3], [139.615088, 35.301088], [139.614, 35.306471], [139.607195, 35.312], [139.608481, 35.32], [139.605373, 35.324627], [139.608615, 35.328], [139.602, 35.334215], [139.598824, 35.331176], [139.597309, 35.332], [139.599193, 35.34], [139.597344, 35.348], [139.606641, 35.352641], [139.619622, 35.349622], [139.622, 35.345415], [139.626, 35.345804], [139.632328, 35.349672], [139.634571, 35.35543], [139.641096, 35.355096], [139.642, 35.351714], [139.643124, 35.354876], [139.647043, 35.356], [139.651157, 35.36], [139.646, 35.366491], [139.642, 35.380687], [139.630099, 35.38], [139.640554, 35.378554], [139.640575, 35.373425], [139.638, 35.368211], [139.63, 35.368026], [139.626, 35.362566], [139.622, 35.363358], [139.615602, 35.36], [139.601424, 35.359424], [139.602, 35.361516], [139.606, 35.360941], [139.61, 35.362443], [139.614, 35.360866], [139.617624, 35.368376], [139.626, 35.373316], [139.629928, 35.38], [139.621121, 35.384], [139.620839, 35.388], [139.618785, 35.388785], [139.619735, 35.396], [139.62441, 35.4], [139.624629, 35.405371], [139.638, 35.41349], [139.65, 35.413056], [139.658, 35.405695], [139.661802, 35.408198], [139.674, 35.409794], [139.67622, 35.408], [139.672906, 35.402906], [139.678, 35.402973], [139.682, 35.400913], [139.688426, 35.401574], [139.689508, 35.404492], [139.692432, 35.405568], [139.692432, 35.410432], [139.685577, 35.411576], [139.684927, 35.414927], [139.68147, 35.416], [139.679532, 35.42], [139.688389, 35.425611], [139.691336, 35.433336], [139.690197, 35.436197], [139.679907, 35.437907], [139.676938, 35.444], [139.674, 35.445437], [139.673537, 35.448463], [139.677214, 35.448786], [139.678384, 35.452384], [139.686, 35.452446], [139.687129, 35.449129], [139.69, 35.448565], [139.700852, 35.449148], [139.701606, 35.452394], [139.704736, 35.453264], [139.705368, 35.456], [139.704736, 35.458736], [139.697529, 35.463529], [139.697803, 35.468197], [139.706, 35.468283], [139.70933, 35.47267], [139.716042, 35.473958], [139.718, 35.476358], [139.724845, 35.477155], [139.726, 35.480174], [139.734, 35.480246], [139.742, 35.474147], [139.744566, 35.48], [139.741337, 35.484], [139.75341, 35.48459], [139.754, 35.488248], [139.769501, 35.488499], [139.769844, 35.492156], [139.777505, 35.492495], [139.777833, 35.496167], [139.78148, 35.49652], [139.78148, 35.50348], [139.777867, 35.504133], [139.793469, 35.504531], [139.793735, 35.508], [139.797415, 35.508585], [139.798, 35.512217], [139.798445, 35.508445], [139.80215, 35.50815], [139.802458, 35.504458], [139.810156, 35.504156], [139.810471, 35.500471], [139.81416, 35.50016], [139.814488, 35.496488], [139.822166, 35.496166], [139.822504, 35.492503], [139.82617, 35.49217], [139.82652, 35.48852], [139.834265, 35.488265], [139.834537, 35.484537], [139.838183, 35.484183], [139.838559, 35.480559], [139.846191, 35.480191], [139.846579, 35.476579], [139.850197, 35.476197], [139.850601, 35.472601], [139.858308, 35.472308], [139.858625, 35.468624], [139.862213, 35.468213], [139.862654, 35.464654], [139.870336, 35.464336], [139.870682, 35.460682], [139.874233, 35.460233], [139.874717, 35.456717], [139.882372, 35.456372], [139.882759, 35.452759], [139.88626, 35.45226], [139.8868, 35.448799], [139.890273, 35.448273], [139.890845, 35.444845], [139.898292, 35.444292], [139.898895, 35.440895], [139.902308, 35.440308], [139.902952, 35.436952], [139.907159, 35.436], [139.896631, 35.425369], [139.896718, 35.418717], [139.906, 35.415954], [139.914162, 35.416162], [139.918, 35.410471], [139.925776, 35.416], [139.924149, 35.42], [139.930253, 35.420253], [139.931073, 35.417073], [139.934297, 35.416], [139.935002, 35.413002], [139.940528, 35.408], [139.942, 35.399631], [139.95, 35.404062], [139.962, 35.401414], [139.968894, 35.386894], [139.976674, 35.382674], [139.978, 35.376157], [139.980779, 35.38], [139.971941, 35.392], [139.971712, 35.408], [139.961555, 35.416], [139.963388, 35.421388], [139.962, 35.425964], [139.954842, 35.424842], [139.95, 35.428836], [139.945788, 35.428], [139.945218, 35.436], [139.948856, 35.44], [139.947788, 35.441787], [139.938, 35.446216], [139.934, 35.443077], [139.91, 35.443519], [139.909077, 35.447077], [139.9057, 35.4477], [139.905114, 35.451114], [139.897722, 35.451722], [139.897186, 35.455186], [139.893734, 35.455733], [139.893215, 35.459215], [139.885628, 35.459628], [139.885272, 35.463272], [139.881762, 35.463762], [139.8813, 35.4673], [139.873664, 35.467664], [139.873337, 35.471337], [139.869782, 35.471782], [139.869359, 35.475359], [139.861692, 35.475692], [139.861391, 35.479391], [139.8578, 35.4798], [139.857406, 35.483406], [139.853806, 35.483806], [139.853424, 35.487424], [139.846, 35.487719], [139.845451, 35.49145], [139.837735, 35.491735], [139.837477, 35.495477], [139.833827, 35.495827], [139.833489, 35.499488], [139.829832, 35.499832], [139.829499, 35.503499], [139.821838, 35.503838], [139.821522, 35.507522], [139.817842, 35.507842], [139.817531, 35.511531], [139.809848, 35.511848], [139.809551, 35.515551], [139.805852, 35.515852], [139.805557, 35.519557], [139.797864, 35.519864], [139.798, 35.528211], [139.800013, 35.522013], [139.802, 35.521056], [139.811764, 35.522236], [139.815307, 35.536], [139.804701, 35.550701], [139.801642, 35.551642], [139.801086, 35.559086], [139.797729, 35.559729], [139.797118, 35.563118], [139.793456, 35.564], [139.792147, 35.570147], [139.787661, 35.572], [139.789653, 35.572347], [139.789826, 35.584], [139.793643, 35.584357], [139.794, 35.588175], [139.794385, 35.584385], [139.802, 35.584203], [139.802457, 35.580456], [139.809544, 35.580456], [139.81, 35.584195], [139.81414, 35.58414], [139.814488, 35.580488], [139.829178, 35.580822], [139.829733, 35.584267], [139.833142, 35.584858], [139.833529, 35.599529], [139.83, 35.599765], [139.82956, 35.60356], [139.825809, 35.604], [139.829601, 35.604399], [139.829864, 35.608136], [139.833579, 35.608421], [139.833857, 35.612143], [139.837565, 35.612435], [139.8378, 35.624], [139.841618, 35.624382], [139.841877, 35.632123], [139.8456, 35.6324], [139.846, 35.644148], [139.846361, 35.640361], [139.862, 35.640155], [139.862338, 35.636338], [139.870113, 35.636113], [139.87044, 35.62044], [139.89, 35.620221], [139.89048, 35.616479], [139.894159, 35.616159], [139.894496, 35.612496], [139.901504, 35.612496], [139.901843, 35.616157], [139.905537, 35.616463], [139.906, 35.624278], [139.906619, 35.620619], [139.913381, 35.620619], [139.913815, 35.624185], [139.921467, 35.624533], [139.921817, 35.628183], [139.925426, 35.628574], [139.926, 35.632301], [139.937323, 35.632677], [139.937759, 35.636241], [139.941111, 35.636889], [139.941111, 35.643111], [139.937767, 35.643767], [139.937378, 35.647378], [139.933801, 35.647801], [139.933419, 35.651419], [139.926, 35.651726], [139.92552, 35.65552], [139.921772, 35.656], [139.925501, 35.656499], [139.926, 35.660286], [139.933394, 35.660606], [139.934, 35.664291], [139.949339, 35.664661], [139.949339, 35.671339], [139.933835, 35.671835], [139.941509, 35.672491], [139.942, 35.680212], [139.946171, 35.680171], [139.946637, 35.672637], [139.950212, 35.672211], [139.950662, 35.668662], [139.954, 35.668331], [139.957338, 35.668662], [139.958, 35.672329], [139.958709, 35.668709], [139.981104, 35.668896], [139.98231, 35.67631], [139.983446, 35.661446], [139.987084, 35.66], [139.984078, 35.657922], [139.984078, 35.654078], [140.006, 35.6526], [140.008801, 35.653199], [140.010415, 35.656415], [140.011695, 35.653695], [140.018479, 35.652479], [140.023756, 35.645756], [140.034286, 35.640286], [140.040493, 35.630493], [140.046, 35.627365], [140.053155, 35.628845], [140.058, 35.633572], [140.063372, 35.634628], [140.065967, 35.640033], [140.070082, 35.636082], [140.079363, 35.634637], [140.081971, 35.636029], [140.08152, 35.64], [140.084348, 35.641652], [140.085341, 35.644659], [140.091652, 35.646348], [140.100368, 35.657632], [140.103517, 35.658483], [140.104515, 35.661485], [140.111429, 35.662571], [140.114, 35.665736], [140.118, 35.664804], [140.122, 35.66786], [140.13, 35.669487], [140.13261, 35.673389], [140.138539, 35.675461], [140.139282, 35.678718], [140.150177, 35.68], [140.150056, 35.683944], [140.152076, 35.684], [140.149975, 35.684], [140.14967, 35.68033], [140.138, 35.68074], [140.134, 35.677404], [140.12831, 35.67769], [140.126, 35.675605], [140.122, 35.67651], [140.118, 35.675057], [140.115169, 35.670831], [140.11, 35.670011], [140.107587, 35.666413], [140.098, 35.665532], [140.094, 35.661999], [140.09, 35.666404], [140.07796, 35.667959], [140.082961, 35.676], [140.074, 35.684666], [140.066, 35.683171], [140.059048, 35.678952], [140.058477, 35.675523], [140.05449, 35.676], [140.056058, 35.68], [140.053014, 35.683014], [140.053624, 35.688], [140.044927, 35.692], [140.040899, 35.710899], [140.038, 35.712282], [140.033448, 35.711448], [140.029619, 35.715619], [140.02147, 35.71547], [140.014, 35.72346], [140.002274, 35.728], [140.001139, 35.731139], [139.997361, 35.731361], [139.996985, 35.734985], [139.99367, 35.736], [139.99, 35.741722], [139.978, 35.743641], [139.974637, 35.748], [139.974061, 35.756], [139.970048, 35.760048], [139.966, 35.760984], [139.962, 35.759634], [139.957637, 35.767637], [139.952977, 35.769023], [139.950141, 35.768141], [139.948898, 35.776], [139.943581, 35.777581], [139.934077, 35.788077], [139.925101, 35.791101], [139.917567, 35.799567], [139.917499, 35.803499], [139.915521, 35.805521], [139.906, 35.798922], [139.902182, 35.800182], [139.89592, 35.808], [139.892334, 35.82], [139.888769, 35.822768], [139.887976, 35.828], [139.889246, 35.840754], [139.89486, 35.844], [139.893005, 35.852995], [139.903078, 35.854922], [139.91, 35.859951], [139.914885, 35.859115], [139.92265, 35.884], [139.921065, 35.891065], [139.925276, 35.896724], [139.932075, 35.897925], [139.933614, 35.900386], [139.937128, 35.900872], [139.938, 35.904159], [139.942, 35.900311], [139.946506, 35.900506], [139.95, 35.897883], [139.955234, 35.898766], [139.955123, 35.904], [139.951476, 35.912], [139.95587, 35.91413], [139.960567, 35.921433], [139.964871, 35.924], [139.965176, 35.928824], [139.96867, 35.932], [139.96744, 35.93856], [139.972374, 35.94], [139.970869, 35.940869], [139.970702, 35.947298], [139.974219, 35.947781], [139.974275, 35.955725], [139.977402, 35.956], [139.973911, 35.956089], [139.973358, 35.948642], [139.969777, 35.948223], [139.96928, 35.944721], [139.964386, 35.941614], [139.959646, 35.926354], [139.95047, 35.91953], [139.942461, 35.919539], [139.934, 35.915558], [139.928181, 35.918181], [139.926, 35.922676], [139.922, 35.922836], [139.92057, 35.92], [139.926202, 35.904], [139.91954, 35.90246], [139.918353, 35.899647], [139.914, 35.898982], [139.904835, 35.910835], [139.890172, 35.915828], [139.878, 35.909246], [139.881531, 35.903531], [139.884384, 35.902384], [139.888944, 35.890944], [139.891612, 35.889612], [139.891873, 35.88], [139.87936, 35.87064], [139.878028, 35.863972], [139.874241, 35.863759], [139.874, 35.861873], [139.859688, 35.872], [139.857463, 35.88], [139.846, 35.884656], [139.842, 35.880875], [139.838, 35.884756], [139.83, 35.883132], [139.826, 35.885761], [139.821586, 35.88], [139.803492, 35.870508], [139.794, 35.871081], [139.774, 35.887308], [139.77, 35.882098], [139.76523, 35.88323], [139.768468, 35.892], [139.766, 35.902203], [139.758, 35.904771], [139.754188, 35.912188], [139.746, 35.917306], [139.742, 35.918955], [139.738368, 35.915632], [139.73637, 35.916], [139.716883, 35.934883], [139.702, 35.939804], [139.704941, 35.944], [139.69, 35.950818], [139.68431, 35.95831], [139.668358, 35.970358], [139.666709, 35.980709], [139.663025, 35.984], [139.66357, 35.99043], [139.666675, 35.992], [139.666424, 35.999576], [139.668426, 36], [139.666345, 36.000345], [139.665212, 35.996], [139.66123, 35.992], [139.660907, 35.98], [139.664391, 35.976], [139.664783, 35.966782], [139.675181, 35.954819], [139.674, 35.952556], [139.67, 35.953488], [139.664434, 35.952], [139.664508, 35.94], [139.667748, 35.937748], [139.669304, 35.931304], [139.681165, 35.924], [139.68583, 35.908], [139.684354, 35.904], [139.685368, 35.9], [139.67875, 35.896], [139.681301, 35.892], [139.6732, 35.88], [139.670496, 35.871504], [139.664926, 35.869074], [139.659088, 35.869088], [139.649631, 35.871631], [139.645476, 35.876], [139.647158, 35.884], [139.636136, 35.890136], [139.636344, 35.893656], [139.640518, 35.896], [139.634, 35.904275], [139.618, 35.908206], [139.602, 35.907361], [139.594, 35.912594], [139.5911, 35.9069], [139.593045, 35.892], [139.586431, 35.888], [139.588139, 35.884], [139.582573, 35.88], [139.582826, 35.863174], [139.576408, 35.866408], [139.576689, 35.872], [139.564459, 35.882458], [139.563572, 35.890428], [139.566786, 35.892], [139.562453, 35.892453], [139.558, 35.896946], [139.55, 35.889525], [139.546, 35.888588], [139.526, 35.897841], [139.516901, 35.898901], [139.518545, 35.92], [139.507003, 35.929003], [139.493485, 35.928515], [139.49, 35.927031], [139.486377, 35.928377], [139.485988, 35.931988], [139.477778, 35.931778], [139.470281, 35.940281], [139.466, 35.94012], [139.458121, 35.931879], [139.449246, 35.928754], [139.45, 35.923834], [139.446089, 35.923911], [139.442, 35.920614], [139.421735, 35.919735], [139.421242, 35.923242], [139.417933, 35.923933], [139.417717, 35.927717], [139.414, 35.930647], [139.413913, 35.936], [139.41704, 35.93696], [139.417803, 35.944197], [139.427601, 35.946399], [139.428981, 35.953019], [139.43291, 35.95709], [139.439782, 35.958218], [139.442997, 35.960997], [139.458, 35.95912], [139.458733, 35.964], [139.462139, 35.968], [139.458078, 35.972], [139.461091, 35.976909], [139.466, 35.976817], [139.47, 35.981188], [139.498, 35.984411], [139.462, 35.986515], [139.450765, 35.975235], [139.443884, 35.974116], [139.434706, 35.963294], [139.428187, 35.961813], [139.426853, 35.959147], [139.423907, 35.958093], [139.422591, 35.951409], [139.415737, 35.952], [139.41994, 35.968], [139.418, 35.970467], [139.406, 35.977519], [139.400921, 35.973079], [139.394, 35.971662], [139.385558, 35.979558], [139.38521, 35.984], [139.38838, 35.988], [139.38693, 36.016], [139.38288, 36.02], [139.38373, 36.02627], [139.386053, 36.028], [139.384114, 36.032], [139.378, 36.036366]]], 'type': 'Polygon'}, 'type': 'Feature'}, {'properties': {'fill-opacity': 0.33, 'fillColor': '#04e813', 'opacity': 0.33, 'fill': '#04e813', 'fillOpacity': 0.33, 'color': '#04e813', 'contour': 10, 'metric': 'time'}, 'geometry': {'coordinates': [[[139.654, 35.697779], [139.642, 35.695827], [139.634, 35.691449], [139.63, 35.693541], [139.626639, 35.691361], [139.616566, 35.689434], [139.615573, 35.684], [139.621888, 35.68], [139.618, 35.677466], [139.612578, 35.68], [139.61297, 35.672], [139.617616, 35.667616], [139.628592, 35.664], [139.63, 35.659065], [139.653891, 35.655891], [139.662, 35.652794], [139.667802, 35.662198], [139.689945, 35.68], [139.69183, 35.684], [139.687149, 35.685149], [139.682, 35.683031], [139.67, 35.683788], [139.670165, 35.688], [139.665818, 35.691818], [139.659542, 35.693542], [139.654, 35.697779]]], 'type': 'Polygon'}, 'type': 'Feature'}, {'properties': {'fill-opacity': 0.33, 'fillColor': '#6706ce', 'opacity': 0.33, 'fill': '#6706ce', 'fillOpacity': 0.33, 'color': '#6706ce', 'contour': 5, 'metric': 'time'}, 'geometry': {'coordinates': [[[139.654, 35.684549], [139.646, 35.685227], [139.634847, 35.68], [139.637826, 35.671826], [139.642, 35.668146], [139.654, 35.665888], [139.658, 35.667265], [139.662738, 35.672], [139.658, 35.68124], [139.654, 35.684549]]], 'type': 'Polygon'}, 'type': 'Feature'}], 'type': 'FeatureCollection'}
    lat = point[1]
    lng = point[0]
    # url = f"https://api.mapbox.com/isochrone/v1/mapbox/driving/{lng},{lat}"
    # url = f"https://api.mapbox.com/isochrone/v1/mapbox/driving/{139.65},{35}"
    # url = f"https://api.mapbox.com/isochrone/v1/mapbox/driving/139.65,34.99038"
    url = f"https://api.mapbox.com/isochrone/v1/mapbox/walking/139.75%2C35.676?contours_minutes=15%2C30%2C45%2C60&polygons=true&denoise=1&generalize=0&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg"
    params = {
        # 'contours_minutes': '5,10,15',
        # 'contours_colors': '6706ce,04e813,4286f4',
        # 'polygons': 'true',
        # 'access_token': 'pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}: {response.text}")
community_cache = {}
def attempt_at_building_communities(_, documentContext, sentence):
    if _ == False: _ = f'Tokyo--Japan.json'
    if type(_) == list: 
        # args = [
        #     "python",
        #     "run_all_fn.py",
        #     json.dumps(_)
        # ]
        # completed_process = subprocess.run(args)
        # print('run all fn has complete')
        return [attempt_at_building_communities(city, documentContext, sentence) for city in _]
    cache_key = hash(json.dumps(_))
    if cache_key in community_cache: return community_cache[cache_key]

    #all_houses = json.load(open(f'data/osm_houses/apt/{_}_houses.json'))
    all_houses = json.load(open('data/airbnb/apt/'+_))
    if len(all_houses) is 0: return []
    geo_coords = [get_lat_long(url, _) for url in all_houses]
    print(geo_coords, 'attempt_at_building_communities')
    people_housing_list = {}

    user_preferences = unstructured_geoSpatial_house_template_query(sentence)
    for idx, person in enumerate(user_preferences):
        name = people_names[idx]
        selected_poi_names = [k for k in user_preferences[person].keys() if k in poi_names]
        people_preferences[name] = [user_preferences[person][key] for key in user_preferences[person]
                                    if key in poi_names
                                    ]
    print('selected_poi_names', selected_poi_names)
    totals = defaultdict(int) 
    h3_cells = retrieveAggregation(selected_poi_names) #{}
    for location in geo_coords: 
        hex_id = h3.geo_to_h3(location[0], location[1], 7)
        if hex_id not in h3_cells: 
            h3_cells[hex_id] = {}
            for col in selected_poi_names: 
                if col not in h3_cells[hex_id]:
                    h3_cells[hex_id][col] = 0
    
    aggregate_poi_in_h3_cell(h3_cells, make_fetch_shops_for_cell(selected_poi_names, h3_cells))
    storeAggregation(h3_cells, selected_poi_names)

    print('coefficents', coefficents)
    for hex_id in h3_cells:
        for key in selected_poi_names:
            totals[key] = max(totals[key], h3_cells[hex_id][key])
            
    h3_cell_counts = copy.deepcopy(h3_cells)

    for hex_id in h3_cells:
        for key in coefficents:
            h3_cells[hex_id][key] = h3_cells[hex_id][key] / totals[key]
            
    _houses = [_housing(url, h3_cells,idx,geo_coords[idx]) for idx, url in enumerate(all_houses)]
    json.dump(_houses, open('_houses.json', 'w+'))
    json.dump(h3_cells, open('h3_cells.json', 'w+'))

    def distanceToTokyo(house):
        point = house['location']
        city_location = cities[_.replace('.json', '')]
        __ = point[1] - city_location[1]
        ____ = point[0] - city_location[0]
        
        dist = math.sqrt(__ * __ + ____ * ____)
        return dist

    people_housing_list = {}
    for idx, person in enumerate(people_names):
        people_housing_list[person] = sorted(_houses, key=distanceToTokyo)[:3000] #sorted by choose a centroid 
        people_housing_list[person] = sorted(people_housing_list[person], key=lambda apt: -key_function(apt, people_preferences[person], idx))
    def getCentroid(houses):
        lat_sum = 0
        lng_sum = 0
        for house in houses: 
            lat_lng = house['location']
            lat_sum += lat_lng[0]
            lng_sum += lat_lng[1]
        return house['location']
    iterations = 2
    indices = [i for i in range(len(people_housing_list))]
    print('people_housing_list', len(people_housing_list))
    candidate = [people_housing_list[person][int(random.random()*  len(people_housing_list))] for idx in indices]
    isochrone = getIsoChrone([1,2])
    top_10_candidates = []
    while iterations > 0:
        for idx, person in enumerate(people_housing_list):
            candidate = [people_housing_list[person][indices[idx]] for idx in indices]
            top_10_candidates.append(candidate)
            point = getCentroid(candidate)
            isochrone = getIsoChrone([1,2])
            feature = isochrone['features'][0]
            polygon = shape(feature['geometry'])
            def house_test(house):
                l = house['location']
                pt = Point(l[0], l[1])
                return polygon.contains(pt)
            within_commute_distance = len([True for house in candidate if house_test(house)]) == len(candidate)
            iterations -= 1
            if within_commute_distance: break
            else: 
                indices[idx] += 1

    reports = []
    for idx, person in enumerate(people_names):
        house = candidate[idx]
        houses_within_suggested_neighborhood = []
        for _house in _houses:
            if _house['h3_cell'] == house['h3_cell']:
                houses_within_suggested_neighborhood.append(_house)
        report = {
            'location': _,
            'name': person,
            'house_suggestion':house['url'] ,
            'house': house,
            'reasoning_explanation': get_reasoning_explanation(people_preferences[person], house, totals, h3_cell_counts, selected_poi_names),
            'houses_within_suggested_neighborhood':houses_within_suggested_neighborhood
        }
        reports.append(report)

    for report in reports: 
        distances = {}
        for other_person in reports: 
            key = other_person['name']
            coords_1 = other_person['house']['location']
            coords_2 = report['house']['location']
            distances[key] = str(round(h3.point_dist(coords_1, coords_2, unit='m') / 2200, 2)) + 'mi'
        report['commutes'] = distances
    result = {'reports': reports, 'isochrone': isochrone, '_houses' : sorted(_houses, key=distanceToTokyo)[:1000],
            'hexes': h3_cell_counts,
            'reasoning_adjustment': 'these conditionare slighly mutually exclusive. you selected price as most important -> heres to get a good deal in japan. if you lower the preference for crime, then youll get a cheaper place. if you lower the slider for commercial, youll get more hipster places and you may have better conversations with "your people".',
            'candidates': top_10_candidates,
            'centroid': point,
            'city': _.replace('.json', '')
            }
    community_cache[cache_key] = result
    return result


def get_reasoning_explanation(prefs, house, totals, h3_cell_counts, selected_poi_names  ):
    reasoning_explanation = []

    max_idx = max_index(prefs)
    counter = 0
    for idx, val in sorted(list(enumerate(prefs)), key=lambda _: -_[1]):
        hex_number = house['h3_cell']
        key = selected_poi_names[idx]
        num_studios = h3_cell_counts[hex_number][key]
        counter += 1
        reasoning_explanation += [f'{counter} {key}  = pref = {val} neighborhood =  {num_studios} / {totals[key]} \n']
    reasoning_explanation = '\n'.join(reasoning_explanation)
    return reasoning_explanation

import math
import asyncio

def getCityList():
    return {
  "Europe": [
    "Paris, France",
    "Rome, Italy",
    "Barcelona, Spain",
    "Amsterdam, Netherlands",
    "London, United Kingdom",
    "Prague, Czech Republic",
    "Vienna, Austria",
    "Budapest, Hungary",
    "Berlin, Germany",
    "Athens, Greece",
    "Venice, Italy",
    "Lisbon, Portugal",
    "Copenhagen, Denmark",
    "Stockholm, Sweden",
    "Edinburgh, Scotland",
    "Dublin, Ireland",
    "Reykjavik, Iceland",
    "Madrid, Spain",
    "Oslo, Norway",
    "Zurich, Switzerland"
  ],
  "North America": [
    "New York City, USA",
    "San Francisco, USA",
    "Vancouver, Canada",
    "New Orleans, USA",
    "Los Angeles, USA",
    "Chicago, USA",
    "Toronto, Canada",
    "Mexico City, Mexico",
    "Montreal, Canada",
    "Boston, USA",
    "Miami, USA",
    "Austin, USA",
    "Quebec City, Canada",
    "Seattle, USA",
    "Nashville, USA"
  ],
  "Asia": [
    "Tokyo, Japan",
    "Kyoto, Japan",
    "Bangkok, Thailand",
    "Hong Kong, China",
    "Singapore, Singapore",
    "Seoul, South Korea",
    "Beijing, China",
    "Dubai, UAE",
    "Taipei, Taiwan",
    "Istanbul, Turkey",
    "Hanoi, Vietnam",
    "Jerusalem, Israel",
    "Mumbai, India",
    "Kuala Lumpur, Malaysia",
    "Jaipur, India"
  ],
  "South America": [
    "Rio de Janeiro, Brazil",
    "Buenos Aires, Argentina",
    "Cartagena, Colombia",
    "Lima, Peru",
    "Santiago, Chile",
    "Cusco, Peru",
    "Medellín, Colombia",
    "Quito, Ecuador",
    "Montevideo, Uruguay",
    "Bogota, Colombia"
  ],
  "Africa": [
    "Cape Town, South Africa",
    "Marrakech, Morocco",
    "Cairo, Egypt",
    "Dakar, Senegal",
    "Zanzibar City, Tanzania",
    "Accra, Ghana",
    "Addis Ababa, Ethiopia",
    "Victoria Falls, Zimbabwe/Zambia",
    "Nairobi, Kenya",
    "Tunis, Tunisia"
  ],
  "Australia and Oceania": [
    "Sydney, Australia",
    "Melbourne, Australia",
    "Auckland, New Zealand",
    "Wellington, New Zealand",
    "Brisbane, Australia"
  ],
  "Others/Islands": [
    "Honolulu, Hawaii, USA",
    "Bali, Indonesia",
    "Santorini, Greece",
    "Maldives (Male)",
    "Phuket, Thailand",
    "Ibiza, Spain",
    "Seychelles (Victoria)",
    "Havana, Cuba",
    "Punta Cana, Dominican Republic",
    "Dubrovnik, Croatia"
  ],
  "Lesser-known Gems": [
    "Ljubljana, Slovenia",
    "Tallinn, Estonia",
    "Riga, Latvia",
    "Sarajevo, Bosnia and Herzegovina",
    "Vilnius, Lithuania",
    "Tbilisi, Georgia",
    "Yerevan, Armenia",
    "Baku, Azerbaijan",
    "Belgrade, Serbia",
    "Skopje, North Macedonia"
  ],
  "For Nature Lovers": [
    "Banff, Canada",
    "Queenstown, New Zealand",
    "Reykjavik (as a gateway to Icelandic nature)",
    "Ushuaia, Argentina (Gateway to Antarctica)",
    "Kathmandu, Nepal (Gateway to the Himalayas)"
  ]
}

def forEachCity(_, __, ___):
    cities = ['Accra--Ghana.json',
        'Addis-Ababa--Ethiopia.json',
        'Cairo--Egypt.json',
        'Cartagena--Colombia.json',
        'Cusco--Peru.json',
        'Dakar--Senegal.json',
        'Hanoi--Vietnam.json',
        'Jaipur--India.json',
        'Kuala-Lumpur--Malaysia.json',
        'Marrakech--Morocco.json',
        'Montevideo--Uruguay.json',
        'Mumbai--India.json',
        'Nairobi--Kenya.json',
        'Santiago--Chile.json',
        'Tunis--Tunisia.json',
        'Zanzibar-City--Tanzania.json',
    ]
    path = 'data/osm_way_residential/'
    # cities = [
    #     'Toronto--Canada', 'Lisbon--Portugal', 'Boston--USA',
    #     'Amsterdam--Netherlands', 'Prague--Czech-Republic', 'Singapore--Singapore'
    #     'Tokyo--Japan', 'Barcelona--Spain', 'Madrid--Spain']
    #cities = ['Tokyo--Japan']
    return os.listdir('data/airbnb/apt')[:5]
    return [json.load(open(path + city)) for city in cities]
    return [f'data/osm_homes/Melbourne--Australia_houses.json']
    #_ = glob.glob(f'data/osm_homes/*_houses.json')
    return _
        #['Tokyo--Japan.json']
    #return os.listdir('data/airbnb/apt')


def world_map(_, __, ___):
    print('world map' + 'fix bugs?')
    cols = [s.replace('.json', '') for s in os.listdir('data/airbnb/h3_poi/')]

    return {'data': {_:json.load(open(_)) for _ in glob.glob('data/airbnb/h3_poi/*.json')},
            'hexAgonations': retrieveAggregation(cols),
            'component': '<Hexagonworld>'
    }

jupyter_functions = { #use regexes + spelling corrector + llm to match sentences w/ functions
    "for every city in ['Tokyo, Japan', 'Houston, Texas', 'Madrid, Spain']" : forEachCity,
    'find 10 houses': attempt_at_building_communities, 
    'group them into topics': groupBySimilarity,
    'for each continent': continentRadio,
    'choose a city in each': cityRadio,
    'find all airbnb in that city': getAirbnbs,
    'filter by distance to shopping store': filter_by_distance_to_shopping_store,
    'filter by 10 min train or drive to a library above 4 star': filter_by_distance_to_shopping_store,
    'plot on a map': lambda _, __, ___: .5,
    'get transcript from ': getYoutube,
    'poll': poll,
    'plant-trees': lambda _,__: 'put map here',
    'arxiv': arxiv,
    'trees_histogram' : trees_histogram,
    'twitch_comments' : twitch_comments,
    'getTopics': getTopics, 
    'trees_map': trees_map,
    'housing_intersection': 'housing_intersection',
    'for each satellite images in area find anything that matches criteria': satellite_housing,
    'given a favorite pokemon': pokemon,
    'get all twitch comments': twitch_comments,
    'map of all airbnbs': map_of_all_airbnbs,
    'i like ': filter_by_poi,
    'find best house': find_best_house,
    'make a world': world_map,
    'map of the future - all airbnbs + pois in the world': world_map
}


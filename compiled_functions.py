from collections import defaultdict
import concurrent.futures
import time 
import h3
import re
import random
import json
import glob
import asyncio
import inspect
from shapely.geometry import shape, Point
import random 
import torch
import requests
import easyocr
from fastapi import Request, FastAPI
import random
import json 
import subprocess
import json
import torch
import youtube_dl
import openai
import re
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from collections import defaultdict
from pydantic import BaseModel
import json
import os.path
import nltk
from nltk.corpus import words
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('words')

# def key_function (_):
#     return _[1]


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
    #{'library': 0, 'coffee': 0, 'bar': 0}

    apt_list = json.load(open(f'data/airbnb/apt/{city_name}.json'))[:50]

    def get_json_if_possible(apt):
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

    geocoordinates = [get_json_if_possible(apt) for apt in apt_list]
    keys = personCoefficentPreferences.keys()

    apts  = []

    import random
    for idx, _ in enumerate(geocoordinates): 
        #print(idx) optimize
        apt = {
            'url': apt_list[idx],
            'loc': geocoordinates[idx]
        } 
        for key in keys:
            coords = _
            apt[key] = random.random()
        apts.append(apt)

    from collections import defaultdict
    totals = defaultdict(int)
    for apt in apts: 
        for key in keys: 
            totals[key] += apt[key]

    for apt in apts: 
        for key in keys: 
            if totals[key] == 0: totals[key] += .01
            apt[key] = apt[key] / totals[key]
    return sorted(apts, key=lambda apt: rankApt(personCoefficentPreferences, apt))[0]



def ocrImage(fp):
    reader = easyocr.Reader(['en'])
    extract_info = reader.readtext(fp)
    from time import time
    sorted(extract_info, key=key_function)
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
    if latitude in isochroneLibraryCache:  
        return isochroneLibraryCache[latitude]
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
    #print('reading cache ', os.path.exists(fp))
    if os.path.exists(fp):
        return json.load(open(fp, 'r'))
    cache = set()
    for _ in url_list[:5]:
        response = requests.get(_)
        if response.status_code == 200:
            with open(_[-50:-1], 'wb') as f:
                f.write(response.content)
        #print('OCR', fp)
        ocr = ocrImage(_[-50:-1])
        if not ocr: continue
        coords = geoCode(ocr, location)
        if not coords: continue
        cache.add(str(coords[0]) + ':' + str(coords[1]))
    #print ('writing to' + fp)
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
    return {'data': geoCode, 'component': '<map>', 
            'geoCoordCache': geoCoordCache
            
            }

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
    
    airbnbs = [apt for apt in airbnbs if doesExist(apt)]
    gm_urls = [json.load(open(fp)) for fp in [f"data/airbnb/gm/{get_room_id(apt_url)}.json" for apt_url in airbnbs
                                              ]
                                              if os.path.exists(fp)
                                              ]
    geoCoordinates = [imageToCoords(_, documentContext['city'], get_room_id(airbnbs[idx]) ) for idx, _ in enumerate(gm_urls[:18])]

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
    #print('ARXIV')
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
    requests.get('https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v11/static/-122.4241,37.78,14.25,0,60/600x600?access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg')
    return 'for each satellite images in area find anything that matches criteria'


geoCoordCache = {}
def fetch_coffee_shops(longitude, latitude, amenities = []):
    if round(longitude, 1) in geoCoordCache: 
        #print('WE GOT THE CACHE', len(geoCoordCache[round(longitude, 1)]))
        return geoCoordCache[round(longitude, 1)]
    # if (os.path.exists(f'data/airbnb/poi/{longitude}_{latitude}_places.json')):
    #     return json.load(open(f'data/airbnb/poi/{longitude}_{latitude}_places.json', 'r'))
   
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
        print(response.status_code, longitude, latitude, amenities)
        if response.status_code == 200:
            data = response.json()
            coffee_shops = data['elements']
            places += coffee_shops
    if len(places) > 0:
        #print(places)
        geoCoordCache[round(longitude, 1)] = places
    #json.dump(places, open(f'data/airbnb/poi/{listing}_places.json', 'w'))
    return places


import os
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

#data/airbnb/h3/poi/{hex :count}
from collections import defaultdict
def fetch_coffee_shops(longitude, latitude, amenities=''):
    #if round(longitude, 2) in geoCoordCache: 
        #print('WE GOT THE CACHE', len(geoCoordCache[round(longitude, 1)]))
        #return geoCoordCache[round(longitude, 2)]
    # if (os.path.exists(f'data/airbnb/poi/{longitude}_{latitude}_places.json')):
    #     return json.load(open(f'data/airbnb/poi/{longitude}_{latitude}_places.json', 'r'))
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
    #print(len(places), longitude, latitude)

    if len(places) > 0:
        #print(places)
        geoCoordCache[round(longitude, 2)] = places
    #json.dump(places, open(f'data/airbnb/poi/{listing}_places.json', 'w'))
    return places


#delete from here to json 
def get_room_id(url):
    match = re.search(r'rooms/(\d+)', url)
    if match:
        return match.group(1)
    else:
        return None

def get_lat_long(url): 
    apt = get_room_id(url)
    data = json.load(open(f'data/airbnb/geocoordinates/{apt}.json'))
    if len(data) == 0: data = [0,0]
    else: 
        data = data[0]
        data = data.split(':')
    data = [float(data[1]), float(data[0])]
    return data

def _housing(url, h3_cells,idx ):
    shit = get_lat_long(url)
    lat = shit[0]
    lng = shit[1]
    h3_cell_id = h3.geo_to_h3(lat, lng, resolution=7)
    _coefficents = h3_cells[h3_cell_id]
    #print('_coefficents', h3_cell_id)
    ret = {
        'url': url,
        'location': shit,
        'h3_cell': h3_cell_id,  
        'coefficents': _coefficents
    }
    # for key in _coefficents: total += h3_cells[h3_cell_id][key]
    # for key in _coefficents: ret['coefficents'][key] = h3_cells[h3_cell_id][key] / total
    return ret 

def key_function(apt, preferences_poi, idx):
    dist = 0
    for key in apt['coefficents']:
        #if (apt['coefficents'][key]) == -1: apt['coefficents'][key] = 0 dont write code like this
        dist += apt['coefficents'][key] - preferences_poi[key][idx]
    return dist
    #make sure each house is in walking distance 30 minutes or train 30 minutes from each other
    #10 * 10 coefficents -> grid of checkboxes 

    
    
def make_fetch_shops_for_cell(poi_names, h3_cells):
    def fetch_shops_for_cell(hex_id):
        results = h3_cells[hex_id]
        ll = h3.h3_to_geo(hex_id)
        poi_names = ['restaurant',
                        'library',
                        'atm',
                        'vending_machine',
                        'bench',
                        'parking_space',
                        'bank',
                        'clinic',
                        'place_of_worship',
                        'research_institute']
        for key in poi_names:
            if key not in results or results[key] == 0:
                #print(key, ll[1])
                val = len(fetch_coffee_shops(ll[1], ll[0], key))
                if val == 0: val = .0000000000000001
                results[key] = val
        return (hex_id, results)
    return fetch_shops_for_cell

def aggregate_poi_in_h3_cell(h3_cells, fn):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for hex_id, results in executor.map(fn, h3_cells.keys()):
            pass
            #h3_cells[hex_id] = results
    return h3_cells


poi_names = ['restaurant',
                'library',
                'atm',
                'vending_machine',
                'bench',
                'parking_space',
                'bank',
                'clinic',
                'place_of_worship',
                'bar',
                'research_institute']
import random
preferences = {}

for i in poi_names: preferences[i] = [random.random() for _ in range(10)]
coefficents = preferences
people_names = 'fred bob sally panda velma gis mercator machester xml parsing'.split(' ')

people_preferences = {}

for person in people_names: people_preferences[person] = [random.random() for _ in range(10)]
# for key in preferences:
#     for index,val in enumerate(preferences[key]):
#         people_preferences[index].append(val)

#print('coeffecients, preferences', coefficents)
people_preferences = {
 'fred': [1, 0, 0, 0, 0, 0, 0, 0, 0, .5],
 'bob': [0, 1, 0, 0, 0, 0, 0, 0, .5, 0],
 'sally': [0, 1, 0, 0, 0, 0, .5, 0, 0, 0],
 'panda': [0, 0, 1, 0, 0, .5, 0, 0, 0, 0],
 'velma': [0, 0, 0, 1, .5, 0, 0, 0, 0, 0],
 'gis': [0, 0, 0, 0, .5, 1, 0, 0, 0, 0],
 'mercator': [0, 0, .5, 0, 1, 0, 0, 0, 0, 0],
 'machester': [0, .5, 0, 0, 0, 1, 0, 0, 0, 0],
 'xml': [0, 0, .5, 0, 0, 0, 0, 0, 1, 0],
 'parsing': [.5, 0, 0, 0, 0, 0, 0, 0, 1, 0]
 }
def max_index(lst): return lst.index(max(lst))
def second_largest_index(lst):
    index_of_max = lst.index(max(lst))
    lst_without_max = lst[:index_of_max] + lst[index_of_max+1:]
    second_largest = max(lst_without_max)
    return lst.index(second_largest)

import copy
async def attempt_at_building_communities(_, documentContext, sentence):
    all_houses = json.load(open('data/airbnb/apt/Tokyo--Japan.json'))
    geo_coords = [get_lat_long(url) for url in all_houses] #tODO flip geo coordinates or re-parse 
    people_housing_list = {}

    totals = defaultdict(int) #todo get from document context or sentence
    #this would be a good function
    h3_cells = retrieveAggregation(poi_names) #{}
    for location in geo_coords: 
        hex_id = h3.geo_to_h3(location[0], location[1], 7)
        if hex_id not in h3_cells: 
            h3_cells[hex_id] = {}
            for col in poi_names: 
                #if col not in h3_cells[hex_id]:
                    h3_cells[hex_id][col] = random.random()
     ##for each column read all h3 cells
    
    aggregate_poi_in_h3_cell(h3_cells, make_fetch_shops_for_cell(poi_names, h3_cells))
    #storeAggregation(h3_cells, poi_names)
    
    for hex_id in h3_cells:
        for key in coefficents:
            #if key in h3_cells[hex_id]:
            totals[key] = max(totals[key], h3_cells[hex_id][key])
            
    h3_cell_counts = copy.deepcopy(h3_cells)

    for hex_id in h3_cells:
        for key in coefficents:
            h3_cells[hex_id][key] = h3_cells[hex_id][key] / totals[key]
            
    #print('h3_cells', h3_cells)
    _houses = [_housing(url, h3_cells,idx) for idx, url in enumerate(all_houses)]
    json.dump(_houses, open('_houses.json', 'w+'))
    json.dump(h3_cells, open('h3_cells.json', 'w+'))
    print('_houses')
    #print('totals', totals)
    #print(_houses)
    #max per neighborhood per column 
    #each house has 
    #mutate housing - get max for each poi
    #maxs = {}
    #for key in poi_names: maxs = 0

    #print(_houses)
    # for idx, person in enumerate(people_names):
    #     people_housing_list[person] = sorted(_houses, key=lambda apt: key_function(apt, preferences, idx))
    reports = []
    for idx, person in enumerate(people_names):
        people_housing_list = sorted(_houses, key=lambda apt: key_function(apt, preferences, idx))
        house = people_housing_list[math.floor(random.random() * len(people_housing_list))]
        report = {
            'name': person,
            'house_suggestion':house['url'] ,
            'house': house,
            'reasoning_explanation': get_reasoning_explanation(people_preferences[person], house, totals, h3_cell_counts),
        }
        reports.append(report)
    #print(_houses)
    return reports


def get_reasoning_explanation(prefs, house, totals, h3_cell_counts):
    reasoning_explanation = ''
    max_idx = max_index(prefs)
    max_2nd_index = second_largest_index(prefs)
    max_key_name = poi_names[max_idx]
    hex_number = house['h3_cell']
    num_studios = h3_cell_counts[hex_number][max_key_name]
    num_studios_2nd = h3_cell_counts[hex_number][poi_names[max_2nd_index]]
    # reasoning_explanation += f'you selected {max_key_name} as the most important and there are {num_studios} studios in that region :) \n'
    # reasoning_explanation += f'you selected {poi_names[max_2nd_index]} as the most important and there are {num_studios_2nd} studios in that region :)'
    reasoning_explanation += f'1. {max_key_name}  =  {num_studios} / {totals[max_key_name]} \n'
    reasoning_explanation += f'2. {poi_names[max_2nd_index]} = {num_studios_2nd} / {totals[poi_names[max_2nd_index]]}'
    return reasoning_explanation
# first = time.time()
# _ = attempt_at_building_communities('', {}, '')
# print(_)
# time.time( )- first
import math
import asyncio
# def attempt_at_building_communities(_, documentContext, sentence):
#     from ipynb.fs.defs.geospatial import attempt_at_building_communities
#     return attempt_at_building_communities(_, documentContext, sentence)
jupyter_functions = { 
    'find 3-5 houses': attempt_at_building_communities,
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
    'find best house': find_best_house
}


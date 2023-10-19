import requests
import os, json
#import psycopg2

def find_closest_poi(poi_name, coord):
    conn = psycopg2.connect(
        dbname="osm",
        user="adnanwahab",
        host="localhost",
        password=""
    )
    cur = conn.cursor()
    min_lon = float(coord[0]) - 1
    max_lon = float(coord[0]) + 1
    min_lat = float(coord[1]) - 1
    max_lat = float(coord[1]) + 1
    LIMIT = 5
    cur.execute(f"""
SELECT osm_id,ST_X(ST_TRANSFORM(way,4674)) AS X, ST_Y(ST_TRANSFORM(way,4674)) as Y,name
from planet_osm_point where
(amenity='{poi_name}')
AND ST_TRANSFORM(way,4674)
@
ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4674) limit {LIMIT};
    """)
    rows = cur.fetchall()

    #for row in rows: print(f"OSM ID: {row[0]}, Latitude: {row[2]}, Longitude: {row[1]}")

    cur.close()
    conn.close()
    return rows


def fetch_things_in_schedule(todo, apt_location):
    # if (os.path.exists(f'data/airbnb/poi/{longitude}_{latitude}_places.json')):
    # return json.load(open(f'data/airbnb/poi/{longitude}_{latitude}_places.json', 'r'))
    latitude = float(apt_location[1])
    longitude = float(apt_location[0])
    places = []
    query = f"""
    [out:json][timeout:25];
    (
        node[amenity="{todo}"]({latitude - .1},{longitude - .1},{latitude + 1},{longitude + 1});
    );
    out body;
    """ 
    overpass_url = "https://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={'data': query})
    #print(response.status_code, longitude, latitude, amenities)
    if response.status_code == 200:
        data = response.json()
        #print(data)
        coffee_shops = data['elements']
        places += coffee_shops
    return places 

def find_closest_poi(poi_name, coord):
    ammenities = json.load(open('data/airbnb/poi.json'))
    #if poi_name in ammenities: 
    return fetch_things_in_schedule(poi_name, coord)
        

def getRoutes(city, apt_url, schedule):
    #city = 'Denver--Colorado--United-States.json' # make this dynamic
    print(city, 'getRoutes')
    apt_location = json.load(open(f'data/airbnb/apt/{city}.json'))[apt_url]
    time, routes = compute_commute_time(schedule, apt_location)
    return routes, time

def geoCode(address = "1600 Amphitheatre Parkway, Mountain View, CA"):
    accessToken = "pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg"  # Replace with your actual access token

    geocodeUrl = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}%2C%20singapore.json?access_token={accessToken}"

    response = requests.get(geocodeUrl)
    data = response.json()

    if 'features' in data and len(data['features']) > 0:
        location = data['features'][0]['geometry']['coordinates']
        #print(f"Longitude: {location[0]}, Latitude: {location[1]}")
        return location

#make this dynamic
def getRoute(start, end):
    # start = [-73.982037, 40.733542]
    # end = [-73.99916, 40.737452]

    start_lng = start[0]
    start_lat = start[1]
    end_lng = end[0]
    end_lat = end[1]

    #url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{start_lng}%2C{start_lon}%3B{end_lng}%2C{end_lon}"
    url = f'https://api.mapbox.com/directions/v5/mapbox/driving/{start_lng}%2C{start_lat}%3B{end_lng}%2C{end_lat}?alternatives=true&geometries=geojson&language=en&overview=full&steps=true&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg'
    #url = 'https://api.mapbox.com/directions/v5/mapbox/driving/-73.982037%2C40.733542%3B-73.99916%2C40.737452?alternatives=true&geometries=geojson&language=en&overview=full&steps=true&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg'

    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        return data
# _ = geoCode('20418 autumn shore drive')
# __ = geoCode('Katy Mills Mall katy texasa')
# getRoute(_, __)

apt = ("7801", [
        "-73.9561767578125",
        "40.718807220458984"
])


apt_location = apt[1]


import geopy.distance

#geopy.distance.geodesic(nearest, (float(apt['latitude']), float(apt['longitude']))).km


def distanceTo(one, two):
    return one['lat'] - float(two[0]) + float(two[1]) - one['lon']



def find_nearest(todo, apt_location):
    all_things = find_closest_poi(todo, apt_location)
    #sorted(all_things, key=lambda place: distanceTo(place, apt_location))
    return all_things[0] if len(all_things) > 0 else None


def get_travel_time(apt_location, location):
    #print(apt_location, location)
    #return 0, []
    route = getRoute(apt_location, (location[1], location[2]))
    #print(route)
    return route['routes'][0]['duration'], route

from concurrent.futures import ThreadPoolExecutor
#may want to cache or use hex-neighborhood 

def get_nearest_poi(apt_location):
    return lambda todo: find_nearest(todo),todo

def compute_commute_time(schedule, apt_location):
    schedule = schedule.copy()
    total_commute_time = 0
    routes = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for task, poi_location in executor.map(get_nearest_poi, schedule.keys()):
            schedule[task] = poi_location
    with ThreadPoolExecutor(max_workers=16) as executor:
        for task, poi_location in executor.map(get_travel_time, schedule.entries()):
                travel_time, route = get_travel_time(apt_location, schedule[todo]['location']) if schedule[todo]['location'] else (0,[])
                routes.append(route)
                total_commute_time = travel_time * schedule[todo]['days_per_week']
    return total_commute_time,routes
    #divide by 7? 
    #goal of app = get commute time down from 90 min to 20 + 15

#colorized hex map that changes for 7 cities for schedule updates -> schedule = suitability w/o sliders 
#bar charts for commute
#table has deal_score, total commute, relevant_complaints within 4 square km

#paper digest
#humor finding and ranking by youtube


#compute deal_score -> derive weighted coefficents ? tell people you will be far from train but thats okay if you got a bicycle :) 
#compute_commute_time(schedule, apt_location)
#compute 311 num_complaints -> for different schedules -> might be inverted :) 




#hex layers = num_complaints, commute_time_To_10_places, price, subjective quality -> land usage -> proximity to park


#rank apts by a list of priorities

#budget is 1250 
#minimize commute to 3 places -> only go to work once a week so you dont mind a 2 hour commute
#but you wnat to go to 5 different friends houses every day because you're a building a company on the side and you know it will take 3 years and you want the commute to be 5 min 

#someone else -> want to go to twork at 6am 
#want to go to work in 15 min walk

#someone else wants to go to work at 11am 2x a week
#20% of people may want to write a schedule in 5 sentences and then get list of apts 


print('getting routes')
# print(getRoutes('', ''))
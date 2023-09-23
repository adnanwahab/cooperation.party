# #scheduled_tasks

# import subprocess 
# import os

# location = 'tokyo--japan'

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


cities = {
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
    #"Jerusalem, Israel",
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

# for location in cities['Asia']:
#     #locations = cities[key]
#     #locations = key
#     #for location in locations:
#         #print(location)
     
#     args = [
#         "node",
#         "rpc/getAptInCity.js",
#         location
#     ]
#     location = location.replace(", ", "--")
#     if not os.path.exists(f'data/airbnb/apt/{location}.json'):
#       completed_process = subprocess.run(args)
#     #location = f'data/airbnb/apt/{location}.json'
#     args = [
#         "node",
#         "rpc/airbnb_get_img_url.js",
#         f'data/airbnb/apt/{location}.json'
#     ]
#     if not os.path.exists(f'data/airbnb/gm/{location}.json'):
#       completed_process = subprocess.run(args)




#get 10 pages of apt

#get 10 square miles from bounding box


import subprocess
import os
import concurrent.futures



import re




def run_scripts(location):
    location_formatted = re.sub(r'[ ,]', '-', location)
    print(location_formatted)
    apt_file = f'data/airbnb/apt/{location_formatted}.json'
    gm_file = f'data/airbnb/gm/{location_formatted}.json'
    
    #if not os.path.exists(apt_file):
    #for each city -> get the bounding box 
    #getAptInCity needs a bounding left-up corner and bottom right corner 
    #get max zoom level -> get w+h in lat/long.1
    #city centroid = 90,100
    #bb = 80,90-90-110
    #400 geo coordinates -> 4000 sq miles
    #4000 "searches" + click next page -> sky scraper -> 100 apts in 1 block
    #cache ocr a bit with a dictionary
    subprocess.run(["node", "rpc/getAptInCity.js", location_formatted])
    if not os.path.exists(gm_file):
        subprocess.run(["node", "rpc/airbnb_get_img_url.js", apt_file])

    for file_name in glob.glob('data/airbnb/apt/*'):
        urls = json.load(open(file_name, 'w'))

        for apt_url in urls:
            print('apt_url', apt_url, 'ocr')
            gm_list = json.load(open(f'data/airbnb/gm/{get_room_id(apt_url)}'))
            imageToCoords(gm_list, location, apt_url)
# Maximum number of concurrent threads
MAX_THREADS = 1

#with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:

import sys
print(sys.argv)
if sys.argv[1] == '33676580':
    print('running image to coords')
    apt_url = 33676580
    gm_list = json.load(open(f'data/airbnb/gm/{apt_url}.json'))
    print(gm_list)
    imageToCoords(gm_list, 'Tokyo--Japan', apt_url)
# for file_name in glob.glob('data/airbnb/apt/*'):
#     print(file_name)
#     try: urls = json.load(open(file_name, 'w'))
#     except Exception as err: 
#         print('oh noe', err, file_name) 
#         continue
#     location = file_name.replace('.json', '')
#     for apt_url in urls:
#         print('apt_url')
#         gm_list = json.load(open(f'data/airbnb/gm/{get_room_id(apt_url)}'))

  
#     imageToCoords(gm_list, location, apt_url)
for continent in cities:
  for city in cities[continent]:
    run_scripts(city)
    executor.map(run_scripts, cities[continent])

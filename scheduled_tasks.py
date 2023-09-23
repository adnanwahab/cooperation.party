# #scheduled_tasks

# import subprocess 
# import os

# location = 'tokyo--japan'

city_coords = {
    "New York City, USA": [
        40.7128,
        -74.006
    ],
    "San Francisco, USA": [
        37.7749,
        -122.4194
    ],
    "Vancouver, Canada": [
        49.2827,
        -123.1207
    ],
    "New Orleans, USA": [
        29.9511,
        -90.0715
    ],
    "Los Angeles, USA": [
        34.0522,
        -118.2437
    ],
    "Chicago, USA": [
        41.8781,
        -87.6298
    ],
    "Toronto, Canada": [
        43.6532,
        -79.3832
    ],
    "Mexico City, Mexico": [
        19.4326,
        -99.1332
    ],
    "Montreal, Canada": [
        45.5017,
        -73.5673
    ],
    "Boston, USA": [
        42.3601,
        -71.0589
    ],
    "Miami, USA": [
        25.7617,
        -80.1918
    ],
    "Austin, USA": [
        30.2672,
        -97.7431
    ],
    "Quebec City, Canada": [
        46.8139,
        -71.2082
    ],
    "Seattle, USA": [
        47.6062,
        -122.3321
    ],
    "Nashville, USA": [
        36.1627,
        -86.7816
    ],
    "Tokyo, Japan": [
        35.6895,
        139.6917
    ],
    "Kyoto, Japan": [
        35.0116,
        135.7681
    ],
    "Bangkok, Thailand": [
        13.7563,
        100.5018
    ],
    "Hong Kong, China": [
        22.3193,
        114.1694
    ],
    "Singapore": [
        1.3521,
        103.8198
    ],
    "Seoul, South Korea": [
        37.5665,
        126.978
    ],
    "Beijing, China": [
        39.9042,
        116.4074
    ],
    "Dubai, UAE": [
        25.276987,
        55.296249
    ],
    "Taipei, Taiwan": [
        25.033,
        121.5654
    ],
    "Istanbul, Turkey": [
        41.0082,
        28.9784
    ],
    "Hanoi, Vietnam": [
        21.0285,
        105.8544
    ],
    "Mumbai, India": [
        19.076,
        72.8777
    ],
    "Kuala Lumpur, Malaysia": [
        3.139,
        101.6869
    ],
    "Jaipur, India": [
        26.9124,
        75.7873
    ],
    "Rio de Janeiro, Brazil": [
        -22.9068,
        -43.1729
    ],
    "Buenos Aires, Argentina": [
        -34.6037,
        -58.3816
    ],
    "Cartagena, Colombia": [
        10.391,
        -75.4794
    ],
    "Lima, Peru": [
        -12.0464,
        -77.0428
    ],
    "Santiago, Chile": [
        -33.4489,
        -70.6693
    ],
    "Cusco, Peru": [
        -13.5319,
        -71.9675
    ],
    "Medellín, Colombia": [
        6.2476,
        -75.5709
    ],
    "Quito, Ecuador": [
        -0.1807,
        -78.4678
    ],
    "Montevideo, Uruguay": [
        -34.9011,
        -56.1911
    ],
    "Bogota, Colombia": [
        4.71,
        -74.0721
    ],
    "Cape Town, South Africa": [
        -33.9249,
        18.4241
    ],
    "Marrakech, Morocco": [
        31.6295,
        -7.9811
    ],
    "Cairo, Egypt": [
        30.8025,
        31.2357
    ],
    "Dakar, Senegal": [
        14.6928,
        -17.4467
    ],
    "Zanzibar City, Tanzania": [
        -6.1659,
        39.2026
    ],
    "Accra, Ghana": [
        5.6037,
        -0.1869
    ],
    "Addis Ababa, Ethiopia": [
        9.03,
        38.74
    ],
    "Victoria Falls, Zimbabwe/Zambia": [
        -17.9243,
        25.8572,
        null
    ],
    "Nairobi, Kenya": [
        -1.286389,
        36.817223
    ],
    "Tunis, Tunisia": [
        36.8065,
        10.1815
    ],
    "Sydney, Australia": [
        -33.8688,
        151.2093
    ],
    "Melbourne, Australia": [
        -37.8136,
        144.9631
    ],
    "Auckland, New Zealand": [
        -36.8485,
        174.7633
    ],
    "Wellington, New Zealand": [
        -41.2865,
        174.7762
    ],
    "Brisbane, Australia": [
        -27.4698,
        153.0251
    ],
    "Honolulu, Hawaii, USA": [
        21.3069,
        -157.8583
    ],
    "Bali, Indonesia": [
        -8.3405,
        115.092,
        null
    ],
    "Santorini, Greece": [
        36.3932,
        25.4615
    ],
    "Maldives (Male)": [
        4.1755,
        73.5093
    ],
    "Phuket, Thailand": [
        7.8804,
        98.3923
    ],
    "Ibiza, Spain": [
        38.9067,
        1.4206
    ],
    "Seychelles (Victoria)": [
        -4.6191,
        55.4513
    ],
    "Havana, Cuba": [
        23.1136,
        -82.3666
    ],
    "Punta Cana, Dominican Republic": [
        18.582,
        -68.4055
    ],
    "Dubrovnik, Croatia": [
        42.6507,
        18.0944
    ],
    "Ljubljana, Slovenia": [
        46.0569,
        14.5058
    ],
    "Tallinn, Estonia": [
        59.437,
        24.7536
    ],
    "Riga, Latvia": [
        56.9496,
        24.1052
    ],
    "Sarajevo, Bosnia and Herzegovina": [
        43.8563,
        18.4131
    ],
    "Vilnius, Lithuania": [
        54.6872,
        25.2797
    ],
    "Tbilisi, Georgia": [
        41.7151,
        44.8271
    ],
    "Yerevan, Armenia": [
        40.1792,
        44.4991
    ],
    "Baku, Azerbaijan": [
        40.4093,
        49.8671
    ],
    "Belgrade, Serbia": [
        44.7866,
        20.4489
    ],
    "Skopje, North Macedonia": [
        41.9973,
        21.428
    ],
    "Banff, Canada": [
        51.1784,
        -115.5708
    ],
    "Queenstown, New Zealand": [
        -45.0312,
        168.6626
    ],
    "Reykjavik (as a gateway to Icelandic nature)": [
        64.1466,
        -21.9426
    ],
    "Ushuaia, Argentina (Gateway to Antarctica)": [
        -54.8019,
        -68.303
    ],
    "Kathmandu, Nepal (Gateway to the Himalayas)": [
        27.7172,
        85.324
    ]
}



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
    # if not os.path.exists(gm_file):
    #     subprocess.run(["node", "rpc/airbnb_get_img_url.js", apt_file])

    # for file_name in glob.glob('data/airbnb/apt/*'):
    #     if len(open(file_name, 'w').read() < 1): continue
    #     urls = json.load(open(file_name, 'r'))


    #     for apt_url in urls:
    #         print('apt_url', apt_url, 'ocr')
    #         gm_list = json.load(open(f'data/airbnb/gm/{get_room_id(apt_url)}'))
    #         imageToCoords(gm_list, location, apt_url)
# Maximum number of concurrent threads
MAX_THREADS = 5

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

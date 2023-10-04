#2 cities
#moscow 

city_location = {"New York City, USA":" 40.7128, -74.0060","San Francisco, USA":" 37.7749, -122.4194","Vancouver, Canada":" 49.2827, -123.1207","New Orleans, USA":" 29.9511, -90.0715","Los Angeles, USA":" 34.0522, -118.2437","Chicago, USA":" 41.8781, -87.6298","Toronto, Canada":" 43.6532, -79.3832","Mexico City, Mexico":" 19.4326, -99.1332","Montreal, Canada":" 45.5017, -73.5673","Boston, USA":" 42.3601, -71.0589","Miami, USA":" 25.7617, -80.1918","Austin, USA":" 30.2672, -97.7431","Quebec City, Canada":" 46.8139, -71.2082","Seattle, USA":" 47.6062, -122.3321","Nashville, USA":" 36.1627, -86.7816","Tokyo, Japan":" 35.6895, 139.6917","Kyoto, Japan":" 35.0116, 135.7681","Bangkok, Thailand":" 13.7563, 100.5018","Hong Kong, China":" 22.3193, 114.1694","Singapore":" 1.3521, 103.8198","Seoul, South Korea":" 37.5665, 126.9780","Beijing, China":" 39.9042, 116.4074","Dubai, UAE":" 25.276987, 55.296249","Taipei, Taiwan":" 25.0330, 121.5654","Istanbul, Turkey":" 41.0082, 28.9784","Hanoi, Vietnam":" 21.0285, 105.8544","Mumbai, India":" 19.0760, 72.8777","Kuala Lumpur, Malaysia":" 3.1390, 101.6869","Jaipur, India":" 26.9124, 75.7873","Rio de Janeiro, Brazil":" -22.9068, -43.1729","Buenos Aires, Argentina":" -34.6037, -58.3816","Cartagena, Colombia":" 10.3910, -75.4794","Lima, Peru":" -12.0464, -77.0428","Santiago, Chile":" -33.4489, -70.6693","Cusco, Peru":" -13.5319, -71.9675","Medellín, Colombia":" 6.2476, -75.5709","Quito, Ecuador":" -0.1807, -78.4678","Montevideo, Uruguay":" -34.9011, -56.1911","Bogota, Colombia":" 4.7100, -74.0721","Cape Town, South Africa":" -33.9249, 18.4241","Marrakech, Morocco":" 31.6295, -7.9811","Cairo, Egypt":" 30.8025, 31.2357","Dakar, Senegal":" 14.6928, -17.4467","Zanzibar City, Tanzania":" -6.1659, 39.2026","Accra, Ghana":" 5.6037, -0.1869","Addis Ababa, Ethiopia":" 9.0300, 38.7400","Victoria Falls, Zimbabwe Zambia":" -17.9243, 25.8572","Nairobi, Kenya":" -1.286389, 36.817223","Tunis, Tunisia":" 36.8065, 10.1815","Sydney, Australia":" -33.8688, 151.2093","Melbourne, Australia":" -37.8136, 144.9631","Auckland, New Zealand":" -36.8485, 174.7633","Wellington, New Zealand":" -41.2865, 174.7762","Brisbane, Australia":" -27.4698, 153.0251","Honolulu, Hawaii, USA":" 21.3069, -157.8583","Bali, Indonesia":" -8.3405, 115.0920","Santorini, Greece":" 36.3932, 25.4615","Maldives (Male)":" 4.1755, 73.5093","Phuket, Thailand":" 7.8804, 98.3923","Ibiza, Spain":" 38.9067, 1.4206","Seychelles (Victoria)":" -4.6191, 55.4513","Havana, Cuba":" 23.1136, -82.3666","Punta Cana, Dominican Republic":" 18.5820, -68.4055","Dubrovnik, Croatia":" 42.6507, 18.0944","Ljubljana, Slovenia":" 46.0569, 14.5058","Tallinn, Estonia":" 59.4370, 24.7536","Riga, Latvia":" 56.9496, 24.1052","Sarajevo, Bosnia and Herzegovina":" 43.8563, 18.4131","Vilnius, Lithuania":" 54.6872, 25.2797","Tbilisi, Georgia":" 41.7151, 44.8271","Yerevan, Armenia":" 40.1792, 44.4991","Baku, Azerbaijan":" 40.4093, 49.8671","Belgrade, Serbia":" 44.7866, 20.4489","Skopje, North Macedonia":" 41.9973, 21.4280","Banff, Canada":" 51.1784, -115.5708","Queenstown, New Zealand":" -45.0312, 168.6626","Reykjavik (as a gateway to Icelandic nature)":" 64.1466, -21.9426","Ushuaia, Argentina (Gateway to Antarctica)":" -54.8019, -68.3030","Kathmandu, Nepal (Gateway to the Himalayas)":" 27.7172, 85.3240"}
#africa-latest.osm.pbf  asia-latest.osm.pbf  australia-oceania-latest.osm.pbf  central-america-latest.osm.pbf  europe-latest.osm.pbf  north-america-latest.osm.pbf  south-america-latest.osm.pbf
def getCityList():
    return {
  "europe": [
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
  "north-america": [
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
  "asia": [
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
  "south-america": [
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
  "africa": [
    "Cape Town, South Africa",
    "Marrakech, Morocco",
    "Cairo, Egypt",
    "Dakar, Senegal",
    "Zanzibar City, Tanzania",
    "Accra, Ghana",
    "Addis Ababa, Ethiopia",
    "Victoria Falls, Zimbabwe-Zambia",
    "Nairobi, Kenya",
    "Tunis, Tunisia"
  ],
  "australia-oceania": [
    "Sydney, Australia",
    "Melbourne, Australia",
    "Auckland, New Zealand",
    "Wellington, New Zealand",
    "Brisbane, Australia"
  ],
#   "Others/Islands": [
#     "Honolulu, Hawaii, USA",
#     "Bali, Indonesia",
#     "Santorini, Greece",
#     "Maldives (Male)",
#     "Phuket, Thailand",
#     "Ibiza, Spain",
#     "Seychelles (Victoria)",
#     "Havana, Cuba",
#     "Punta Cana, Dominican Republic",
#     "Dubrovnik, Croatia"
#   ],
#   "Lesser-known Gems": [
#     "Ljubljana, Slovenia",
#     "Tallinn, Estonia",
#     "Riga, Latvia",
#     "Sarajevo, Bosnia and Herzegovina",
#     "Vilnius, Lithuania",
#     "Tbilisi, Georgia",
#     "Yerevan, Armenia",
#     "Baku, Azerbaijan",
#     "Belgrade, Serbia",
#     "Skopje, North Macedonia"
#   ],
#   "For Nature Lovers": [
#     "Banff, Canada",
#     "Queenstown, New Zealand",
#     "Reykjavik (as a gateway to Icelandic nature)",
#     "Ushuaia, Argentina (Gateway to Antarctica)",
#     "Kathmandu, Nepal (Gateway to the Himalayas)"
#   ]
}




#aus = getCityList()['Australia and Oceania']


import subprocess

requests = []
def fn(parameters):
    #print(' '.join(parameters))
    result = subprocess.run(parameters, capture_output=True, text=True)
    print("Return code:", result.returncode)
    print("stdout:\n{}".format(result.stdout))
    print("stdout:\n{}".format(result.stderr))

    
def makeRequests(continent, city):
    if city not in city_location: return print(city + 'not found')
    location = city_location[city].split(',')
    location = [float(location[0].strip()), float(location[1].strip())]
    location.reverse()
    bbox = [location[0] -1, location[1] -1, location[0] + 1, location[1] + 1]
    city_name = city.replace(',','-').replace(r' ', '-')
    #request = f'osmium extract --bbox {bbox} australia-oceania-latest.osm.pbf -o {city}.osm'
    #request = f'osmium extract --bbox {bbox} data/planet-latest.osm -o - | osmium tags-filter - n/building=house -o data/{city_name}.osm.pbf'
    #request = 'osmium extract --bbox 139.5,35.5,140.1,36.0 planet-latest.osm   -o _.osm'
    request = [
        'osmium', 'extract',
        '--overwrite',
        '--bbox', ','.join(map(str, bbox)),
        f'osm_pbf/{continent}-latest.osm.pbf',
        '-o', f'osm_homes/{city_name}.osm'
    ]
    requests.append(' '.join(request))
    #print(f"'{' '.join(request)}'")
    #print(' ')
    #print(f'osmium tags-filter  osm_homes/{city_name}.osm n/building=house -o osm_homes/{city_name}_houses.osm')
    #print(' ')

    #fn(request)
    #requests.append(request)

for continent in getCityList():
    for city in getCityList()[continent]:
        makeRequests(continent, city)
requests.reverse()
# import concurrent
# with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#     for p in executor.map(fn, requests):
#         print('_')
          
          
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_cmd(cmd):
    print(f"Starting: {cmd}")
    process = subprocess.run(cmd, shell=True)
    #process.communicate()
    if process.returncode == 0:
        print(f"Done: {cmd}")
    else:
        print(f"Error running: {cmd}")
    return cmd


# with ThreadPoolExecutor(max_workers=9) as executor:
#     results = list(executor.map(run_cmd, requests))

print('\n'.join(requests))
print("All commands executed!")

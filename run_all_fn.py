import concurrent.futures
import subprocess
import json
import os

#cities = json.load(open('data/airbnb/cities.json'))

most_dense = [
    "Mumbai",
    "Jakarta",
    "Seoul",
    "Cairo",
    "Dhaka",
    "Kolkata",
    "Buenos Aires",
    "Bandung",
    "Quezon City",
    "Chittagong",
    "Paris",
    "Manila",
    "Damascus",
    "Caloocan",
    "Barcelona",
    "Kathmandu",
    "Ciudad Nezahualcóyotl",
    "Howrah",
    "Freetown",
    "Port-au-Prince",
    "Asmara",
    "Pasig",
    "Colombo",
    "Bacoor",
    "La Plata",
    "Macau",
    "Makati",
    "Las Piñas",
    "Cimahi",
    "Marikina",
    "Pasay",
    "Mandaluyong",
    "Malabon",
    "Mandaue",
    "San Pedro, Laguna",
    "L'Hospitalet de Llobregat",
    "Navotas",
    "Bnei Brak",
    "Geneva",
    "General Mariano Alvarez",
    "Malé",
    "Schaerbeek",
    "Bat-Yam",
    "San Juan, Metro Manila",
    "Boulogne-Billancourt",
    "Santa Coloma de Gramenet",
    "Rosario, Cavite",
    "Kallithea",
    "Sint-Jans-Molenbeek",
    "Asnières-sur-Seine",
    "Courbevoie",
    "Modi'in Illit",
    "Warabi",
    "Nea Smyrni",
    "Issy-les-Moulineaux",
    "Union City",
    "Pateros",
    "Levallois-Perret",
    "Clichy",
    "Hoboken",
    "Neuilly-sur-Seine",
    "Giv'atayim",
    "Portici",
    "Montrouge",
    "Saint-Gilles",
    "Vincennes",
    "West New York",
    "El'ad",
    "Mislata",
    "Monaco",
    "Charenton-le-Pont",
    "La Garenne-Colombes",
    "Vanves",
    "Saint-Josse-ten-Noode",
    "Neapoli, Thessaloniki",
    "Les Lilas",
    "Saint-Mandé",
    "Sliema",
    "Koekelberg",
    "Gentilly",
    "Le Pré-Saint-Gervais",
    "Kotsiubynske",
    "Senglea"
]

national_capitals = [
    "Beijing",
    "Tokyo",
    "Moscow",
    "Kinshasa",
    "Jakarta",
    "Cairo",
    "Seoul",
    "Mexico City",
    "London",
    "Dhaka",
    "Lima",
    "Tehran",
    "Bangkok",
    "Hanoi",
    "Baghdad",
    "Riyadh",
    "Hong Kong",
    "Bogotá",
    "Santiago",
    "Ankara",
    "Singapore",
    "Kabul",
    "Nairobi",
    "Amman",
    "Algiers",
    "Berlin",
    "Madrid",
    "Addis Ababa",
    "Kuwait City",
    "Guatemala City",
    "Pretoria",
    "Kyiv",
    "Buenos Aires",
    "Pyongyang",
    "Tashkent",
    "Rome",
    "Quito",
    "Yaoundé",
    "Lusaka",
    "Khartoum",
    "Brasília",
    "Taipei (de facto)",
    "Sanaa",
    "Luanda",
    "Ouagadougou",
    "Accra",
    "Mogadishu",
    "Baku",
    "Phnom Penh",
    "Caracas",
    "Paris",
    "Havana",
    "Harare",
    "Damascus",
    "Minsk",
    "Vienna",
    "Warsaw",
    "Manila",
    "Bamako",
    "Kuala Lumpur",
    "Bucharest",
    "Budapest",
    "Brazzaville",
    "Belgrade",
    "Kampala",
    "Conakry",
    "Ulaanbaatar",
    "Tegucigalpa",
    "Dakar",
    "Prague",
    "Niamey",
    "Montevideo",
    "Sofia",
    "Muscat",
    "Antananarivo",
    "Astana",
    "Abuja",
    "Tbilisi",
    "Nouakchott",
    "Doha",
    "Tripoli",
    "Naypyidaw",
    "Kigali",
    "Maputo",
    "Santo Domingo",
    "Yerevan",
    "Bishkek",
    "Freetown",
    "Managua",
    "Ottawa",
    "Islamabad",
    "Monrovia",
    "Abu Dhabi",
    "Lilongwe",
    "Port-au-Prince",
    "Stockholm",
    "Asmara",
    "Jerusalem[a]",
    "Vientiane",
    "N'Djamena",
    "Amsterdam",
    "Bangui",
    "Panama City",
    "Dushanbe",
    "Kathmandu",
    "Lomé",
    "Ashgabat",
    "Chişinău",
    "Zagreb",
    "Libreville",
    "Oslo",
    "Macau",
    "Washington, D.C.",
    "Kingston",
    "Helsinki",
    "Tunis",
    "Copenhagen",
    "Athens",
    "Riga",
    "Djibouti (city)",
    "Dublin",
    "Rabat",
    "Vilnius",
    "San Salvador",
    "Tirana",
    "Skopje",
    "Juba",
    "Asunción",
    "Lisbon",
    "Bissau",
    "Bratislava",
    "Tallinn",
    "Canberra",
    "Windhoek",
    "Dodoma",
    "Port Moresby",
    "Yamoussoukro",
    "Beirut",
    "Sucre",
    "San Juan",
    "San José",
    "Maseru",
    "Nicosia",
    "Malabo",
    "Ljubljana",
    "Dili",
    "Sarajevo",
    "Nassau",
    "Gaborone",
    "Porto-Novo",
    "New Delhi",
    "Paramaribo",
    "Laayoune (claimed)Tifariti (de facto)\n",
    "Wellington",
    "Manama",
    "Pristina",
    "Podgorica",
    "Brussels",
    "Praia",
    "Port Louis",
    "Willemstad",
    "Gitega",
    "Bern (de facto)",
    "Tiraspol",
    "Malé",
    "Reykjavík",
    "Luxembourg City",
    "Georgetown",
    "Thimphu",
    "Moroni",
    "Bridgetown",
    "Sri Jayawardenepura Kotte",
    "Bandar Seri Begawan",
    "Mbabane",
    "Nouméa",
    "Suva",
    "Honiara",
    "Stepanakert",
    "Banjul",
    "São Tomé",
    "Tarawa",
    "Port Vila",
    "Saipan",
    "Apia",
    "Ramallah (de facto)",
    "Monaco",
    "Saint Helier",
    "Port of Spain",
    "George Town",
    "Gibraltar",
    "St. George's",
    "Oranjestad",
    "Douglas",
    "Majuro",
    "Nukuʻalofa",
    "Victoria",
    "Papeete",
    "Andorra la Vella",
    "Tórshavn",
    "St. John's",
    "Belmopan",
    "Castries",
    "Saint Peter Port",
    "Nuuk",
    "Roseau",
    "Basseterre",
    "Kingstown",
    "Road Town",
    "Mariehamn",
    "Charlotte Amalie",
    "Palikir",
    "Funafuti",
    "Valletta",
    "Vaduz",
    "Saint-Pierre",
    "Avarua",
    "City of San Marino",
    "Cockburn Town",
    "Pago Pago",
    "Marigot",
    "Gustavia",
    "Stanley",
    "Longyearbyen",
    "Philipsburg",
    "Flying Fish Cove",
    "Hagåtña",
    "Mata Utu",
    "Hamilton",
    "Yaren (de facto)",
    "Jamestown",
    "Alofi",
    "Atafu",
    "Vatican City (city-state)",
    "Brades (de facto)Plymouth (de jure)",
    "Kingston",
    "West Island",
    "Adamstown",
    "King Edward Point",
    "Ngerulmud"
]

most_populated_cities = [
  "Tokyo",
  "Delhi",
  "Shanghai",
  "São_Paulo",
  "Mexico_City",
  "Cairo",
  "Mumbai",
  "Beijing",
  "Dhaka",
  "Osaka",
  "New_York",
  "Karachi",
  "Buenos_Aires",
  "Chongqing",
  "Istanbul",
  "Kolkata",
  "Manila",
  "Lagos",
  "Rio_de_Janeiro",
  "Tianjin",
  "Kinshasa",
  "Guangzhou",
  "Los_Angeles",
  "Moscow",
  "Shenzhen",
  "Lahore",
  "Bangalore",
  "Paris",
  "Bogotá",
  "Jakarta",
  "Chennai",
  "Lima",
  "Bangkok",
  "Seoul",
  "Nagoya",
  "Hyderabad",
  "London",
  "Tehran",
  "Chicago",
  "Chengdu",
  "Nanjing",
  "Wuhan",
  "Ho_Chi_Minh_City",
  "Luanda",
  "Ahmedabad",
  "Kuala_Lumpur",
  "Xi'an",
  "Hong_Kong",
  "Dongguan",
  "Hangzhou",
  "Foshan",
  "Shenyang",
  "Riyadh",
  "Baghdad",
  "Santiago",
  "Surat",
  "Madrid",
  "Suzhou",
  "Pune",
  "Harbin",
  "Houston",
  "Dallas",
  "Toronto",
  "Dar_es_Salaam",
  "Miami",
  "Belo_Horizonte",
  "Singapore",
  "Philadelphia",
  "Atlanta",
  "Fukuoka",
  "Khartoum",
  "Barcelona",
  "Johannesburg",
  "Saint_Petersburg",
  "Qingdao",
  "Dalian",
  "Washington,_D.C.",
  "Yangon",
  "Alexandria",
  "Jinan",
  "Guadalajara"
]

satellite_cities = [
    "Giza (informal)",
    "New Taipei City (agglo)",
    "Yokohama",
    "Ekurhuleni (agglo)",
    "Incheon",
    "Quezon City",
    "Bekasi",
    "Ghaziabad",
    "Taoyuan",
    "Depok",
    "Tangerang",
    "Viana",
    "Omdurman",
    "La Matanza",
    "Thane",
    "Pimpri-Chinchwad",
    "Caloocan",
    "Ecatepec de Morelos",
    "Matola",
    "Philadelphia",
    "Karaj",
    "Narayanganj",
    "South Tangerang",
    "Kawasaki",
    "Kobe City",
    "Gurgaon",
    "Zapopan",
    "Faridabad",
    "Guarulhos",
    "Saitama City",
    "Cacuaco",
    "Belas",
    "Sharjah",
    "Kalyan-Dombivali",
    "Suwon",
    "Shubra El-Kheima",
    "Vasai-Virar",
    "Gazipur",
    "Ulsan",
    "Navi Mumbai",
    "Changwon",
    "Cazenga",
    "Ciudad Nezahualcóyotl",
    "Howrah",
    "Goyang",
    "Yongin",
    "Biên Hòa",
    "Bogor",
    "San Jose",
    "Callao",
    "Khartoum North",
    "Chiba City",
    "Seongnam",
    "El Alto",
    "São Gonçalo",
    "Hawalli",
    "Santo Domingo Este",
    "Salé",
    "Antipolo",
    "Taguig",
    "Bucheon",
    "Hwaseong",
    "Cheongju",
    "São Bernardo do Campo",
    "Naucalpan de Juárez",
    "Sakai",
    "Soacha",
    "Seberang Perai",
    "Farwaniya",
    "Mira-Bhayander",
    "Nova Iguaçu",
    "Pasig",
    "Kajang",
    "Duque de Caxias",
    "Osasco",
    "Santo André",
    "La Plata",
    "Hempstead",
    "Ansan",
    "Noida/Greater Noida",
    "Klang",
    "Tlajomulco de Zúñiga",
    "Sagamihara",
    "Emfuleni",
    "Mississauga",
    "Valenzuela",
    "Bhiwandi",
    "Parañaque",
    "Namyangju",
    "Subang Jaya",
    "Chimalhuacán",
    "Dasmariñas",
    "Tolyatti",
    "Lomas de Zamora",
    "Jaboatão dos Guararapes",
    "Tlaquepaque",
    "Tlalnepantla de Baz",
    "Soledad",
    "Contagem",
    "Bacoor",
    "Apodaca",
    "San Jose del Monte",
    "Gold Coast"
]

cities = most_dense  + satellite_cities + national_capitals + most_populated_cities
cities = list(set(cities))

# cities = {
#     "New-York-City--USA": [
#         40.7128,
#         -74.006
#     ],
#     "San-Francisco--USA": [
#         37.7749,
#         -122.4194
#     ],
#     "Vancouver--Canada": [
#         49.2827,
#         -123.1207
#     ],
#     "New-Orleans--USA": [
#         29.9511,
#         -90.0715
#     ],
#     "Los-Angeles--USA": [
#         34.0522,
#         -118.2437
#     ],
#     "Chicago--USA": [
#         41.8781,
#         -87.6298
#     ],
#     "Toronto--Canada": [
#         43.6532,
#         -79.3832
#     ],
#     "Mexico-City--Mexico": [
#         19.4326,
#         -99.1332
#     ],
#     "Montreal--Canada": [
#         45.5017,
#         -73.5673
#     ],
#     "Boston--USA": [
#         42.3601,
#         -71.0589
#     ],
#     "Miami--USA": [
#         25.7617,
#         -80.1918
#     ],
#     "Austin--USA": [
#         30.2672,
#         -97.7431
#     ],
#     "Quebec-City--Canada": [
#         46.8139,
#         -71.2082
#     ],
#     "Seattle--USA": [
#         47.6062,
#         -122.3321
#     ],
#     "Nashville--USA": [
#         36.1627,
#         -86.7816
#     ],
#     "Tokyo--Japan": [
#         35.6895,
#         139.6917
#     ],
#     "Kyoto--Japan": [
#         35.0116,
#         135.7681
#     ],
#     "Bangkok--Thailand": [
#         13.7563,
#         100.5018
#     ],
#     "Hong-Kong--China": [
#         22.3193,
#         114.1694
#     ],
#     "Singapore": [
#         1.3521,
#         103.8198
#     ],
#     "Seoul--South-Korea": [
#         37.5665,
#         126.978
#     ],
#     "Beijing--China": [
#         39.9042,
#         116.4074
#     ],
#     "Dubai--UAE": [
#         25.276987,
#         55.296249
#     ],
#     "Taipei--Taiwan": [
#         25.033,
#         121.5654
#     ],
#     "Istanbul--Turkey": [
#         41.0082,
#         28.9784
#     ],
#     "Hanoi--Vietnam": [
#         21.0285,
#         105.8544
#     ],
#     "Mumbai--India": [
#         19.076,
#         72.8777
#     ],
#     "Kuala-Lumpur--Malaysia": [
#         3.139,
#         101.6869
#     ],
#     "Jaipur--India": [
#         26.9124,
#         75.7873
#     ],
#     "Rio-de-Janeiro--Brazil": [
#         -22.9068,
#         -43.1729
#     ],
#     "Buenos-Aires--Argentina": [
#         -34.6037,
#         -58.3816
#     ],
#     "Cartagena--Colombia": [
#         10.391,
#         -75.4794
#     ],
#     "Lima--Peru": [
#         -12.0464,
#         -77.0428
#     ],
#     "Santiago--Chile": [
#         -33.4489,
#         -70.6693
#     ],
#     "Cusco--Peru": [
#         -13.5319,
#         -71.9675
#     ],
#     "Medellín--Colombia": [
#         6.2476,
#         -75.5709
#     ],
#     "Quito--Ecuador": [
#         -0.1807,
#         -78.4678
#     ],
#     "Montevideo--Uruguay": [
#         -34.9011,
#         -56.1911
#     ],
#     "Bogota--Colombia": [
#         4.71,
#         -74.0721
#     ],
#     "Cape-Town--South-Africa": [
#         -33.9249,
#         18.4241
#     ],
#     "Marrakech--Morocco": [
#         31.6295,
#         -7.9811
#     ],
#     "Cairo--Egypt": [
#         30.8025,
#         31.2357
#     ],
#     "Dakar--Senegal": [
#         14.6928,
#         -17.4467
#     ],
#     "Zanzibar-City--Tanzania": [
#         -6.1659,
#         39.2026
#     ],
#     "Accra--Ghana": [
#         5.6037,
#         -0.1869
#     ],
#     "Addis-Ababa--Ethiopia": [
#         9.03,
#         38.74
#     ],
#     "Victoria-Falls--Zimbabwe/Zambia": [
#         -17.9243,
#         25.8572
#     ],
#     "Nairobi--Kenya": [
#         -1.286389,
#         36.817223
#     ],
#     "Tunis--Tunisia": [
#         36.8065,
#         10.1815
#     ],
#     "Sydney--Australia": [
#         -33.8688,
#         151.2093
#     ],
#     "Melbourne--Australia": [
#         -37.8136,
#         144.9631
#     ],
#     "Auckland--New-Zealand": [
#         -36.8485,
#         174.7633
#     ],
#     "Wellington--New-Zealand": [
#         -41.2865,
#         174.7762
#     ],
#     "Brisbane--Australia": [
#         -27.4698,
#         153.0251
#     ],
#     "Honolulu--Hawaii--USA": [
#         21.3069,
#         -157.8583
#     ],
#     "Bali--Indonesia": [
#         -8.3405,
#         115.092
#     ],
#     "Santorini--Greece": [
#         36.3932,
#         25.4615
#     ],
#     "Maldives-(Male)": [
#         4.1755,
#         73.5093
#     ],
#     "Phuket--Thailand": [
#         7.8804,
#         98.3923
#     ],
#     "Ibiza--Spain": [
#         38.9067,
#         1.4206
#     ],
#     "Seychelles-(Victoria)": [
#         -4.6191,
#         55.4513
#     ],
#     "Havana--Cuba": [
#         23.1136,
#         -82.3666
#     ],
#     "Punta-Cana--Dominican-Republic": [
#         18.582,
#         -68.4055
#     ],
#     "Dubrovnik--Croatia": [
#         42.6507,
#         18.0944
#     ],
#     "Ljubljana--Slovenia": [
#         46.0569,
#         14.5058
#     ],
#     "Tallinn--Estonia": [
#         59.437,
#         24.7536
#     ],
#     "Riga--Latvia": [
#         56.9496,
#         24.1052
#     ],
#     "Sarajevo--Bosnia-and-Herzegovina": [
#         43.8563,
#         18.4131
#     ],
#     "Vilnius--Lithuania": [
#         54.6872,
#         25.2797
#     ],
#     "Tbilisi--Georgia": [
#         41.7151,
#         44.8271
#     ],
#     "Yerevan--Armenia": [
#         40.1792,
#         44.4991
#     ],
#     "Baku--Azerbaijan": [
#         40.4093,
#         49.8671
#     ],
#     "Belgrade--Serbia": [
#         44.7866,
#         20.4489
#     ],
#     "Skopje--North-Macedonia": [
#         41.9973,
#         21.428
#     ],
#     "Banff--Canada": [
#         51.1784,
#         -115.5708
#     ],
#     "Queenstown--New-Zealand": [
#         -45.0312,
#         168.6626
#     ],
#     "Reykjavik-(as-a-gateway-to-Icelandic-nature)": [
#         64.1466,
#         -21.9426
#     ],
#     "Ushuaia--Argentina-(Gateway-to-Antarctica)": [
#         -54.8019,
#         -68.303
#     ],
#     "Kathmandu--Nepal-(Gateway-to-the-Himalayas)": [
#         27.7172,
#         85.324
#     ]
# }

import random#
#TODO - remove shuffle

#cities = list(cities.keys())
random.shuffle(cities)

def fn(city):
    return subprocess.run(['node', 'rpc/getAptInCity.js', city])

# def fn2(city):
#     return subprocess.run(['node', 'rpc/airbnb_get_img_url.js', f'{city}'])

# from compiled_functions import get_lat_long

# def fn3(city):
#     all_houses = json.load(open('data/airbnb/apt/'+city+'.json'))
#     if len(all_houses) is 0: return []
#     geo_coords = [get_lat_long(url, city) for url in all_houses]

with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
    for results in executor.map(fn, cities):
            print(results)

import json

# with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#     for results in executor.map(fn2, cities):
#         print(results)


# with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#     for results in executor.map(fn3, cities):
#         print(results)

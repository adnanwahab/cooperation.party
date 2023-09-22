#scheduled_tasks

import subprocess 
import os

location = 'tokyo--japan'

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

for location in cities['Asia']:
    #locations = cities[key]
    #locations = key
    #for location in locations:
        #print(location)
     
    args = [
        "node",
        "rpc/getAptInCity.js",
        location
    ]
    location = location.replace(", ", "--")
    # if not os.path.exists(f'data/airbnb/apt/{location}.json'):
    #     #print('hellos')
    #     5 + 5
    #   #completed_process = subprocess.run(args)
    #location = f'data/airbnb/apt/{location}.json'
    # print('location', location)
    args = [
        "node",
        "rpc/airbnb_get_img_url.js",
        f'data/airbnb/apt/{location}.json'
    ]
    completed_process = subprocess.Popen(args)




#get 10 pages of apt

#get 10 square miles from bounding box

#
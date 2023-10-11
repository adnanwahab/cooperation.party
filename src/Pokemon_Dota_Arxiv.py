#!/usr/bin/env python
# coding: utf-8

# In[ ]:


sentences = [item for sublist in sentences2 for item in sublist]
encodings = model.encode(sentences, convert_to_tensor=True, device='cpu')
clusters = util.community_detection(encodings, min_community_size=1, threshold=0.55)
#[sentences[i] for sentences in clusters for i, s in enumerate(sentences)]

#[sentences[sentenceIdx] for cluster in clusters for sentenceIdx in cluster]

clusters
def sentenceIDToSentence(num):
    return sentences[num]

sentenceClusters = []
for s in clusters:
    sentenceCluster = []
    for id, sentenceId in enumerate(s):
        sentenceCluster.append(sentences[sentenceId])
    sentenceClusters.append(sentenceCluster)
sentenceClusters
# print(clusters)
# print(sentences)

#sentences
#[sentences[sentenceIdx] for cluster in clusters for sentenceIdx in cluster]


# In[ ]:


[fl[i] for sentences in clusters for i, s in enumerate(sentences)]

#[fl[i] for clusters in clusters for i, s in enumerate(s)]

#[fl[i] for i, s in enumerate(items_in_cluster) for items_in_cluster in clusters]


# In[ ]:


import asyncio
import json
import sys
from pyppeteer import launch

async def delay(time):
    await asyncio.sleep(time)

async def get_html():
    channel_name = 'elajjaz'
    print(f"hi, {channel_name}")

    browser = await launch(headless=True)
    page = await browser.newPage()

    await page.setViewport({"width": 1920, "height": 1080})
    await page.goto(f"https://www.twitch.tv/{channel_name}", {"waitUntil": "networkidle2"})

    await delay(1)

    await page.waitForFunction("""
    () => {
        const el = document.querySelector('div[data-a-target="chat-welcome-message"].chat-line__status');
        return !!el;
    }
    """, polling="raf")

    selector = '.chat-line__message, .chat-line__status, div[data-a-target="chat-line-message"]'
    await page.waitForSelector(selector)

    text_elements = await page.querySelectorAll(selector)
    texts = []

    for element in text_elements:
        content = await page.evaluate('(element) => element.textContent', element)
        texts.append(content.strip())

    await browser.close()

    print(f"texts: {texts}")

    with open(f"twitch-{channel_name}.json", "w") as f:
        json.dump(texts, f)

    return texts
async def main():
    texts = await get_html()
    print(f"Scraped texts: {texts}")
loop = asyncio.get_event_loop()
loop.create_task(main())
# for i in range(10):
#     #task = asyncio.create_task(some_coro(param=i))

#     loop = asyncio.get_event_loop()
#     loop.create_task(main())
#     delay(1000)


# In[ ]:


import subprocess

#subprocess.run(['node', './streaming-output.js'])
import asyncio

async def run(_):
    print('___', _)
    proc = await asyncio.create_subprocess_shell(
        _,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await proc.communicate()

    print(f'[{cmd!r} exited with {proc.returncode}]')
    if stdout:
        print(f'[stdout]\n{stdout.decode()}')
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')

#asyncio.run()

loop = asyncio.get_event_loop()
loop.create_task(run('node ./streaming-output.js'))


# In[ ]:


import re


text = "get all twitch comments from ['zackrawrr', 'elajjaz', 'arthars']"

pattern = r"\[([^\]]+)\]"

match = re.search(pattern, text)

match[0][1:-1].replace('\'', '').split(',')


# In[ ]:


import requests
from bs4 import BeautifulSoup

# The URL of the page you want to scrape
webpage_url = 'https://example.com/some_page_with_pdfs'

# Fetch the content of the webpage
response = requests.get(webpage_url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Assuming the PDF links have a specific identifier or unique attribute
# Modify this part to suit the website structure
pdf_links = soup.find_all('a', {'class': 'pdf-link-class'})

# Download the first PDF (you can loop over all PDFs if you want)
if pdf_links:
    pdf_url = pdf_links[0]['href']


# In[ ]:


import PyPDF2
from PIL import Image
import pdfreader
from pypdf import PdfReader
import requests
import pypdf
from bs4 import BeautifulSoup

def get_paper_google_scholar(query):
    res = requests.get(f'https://scholar.google.com/scholar?start=0&q={query}&hl=en&as_sdt=0,44)')
    text = res.text
    
    _ = BeautifulSoup(text)
    
    return _.find_all('.gs_or_ggsm')

def pdf_image_extract(fp):
    reader = PdfReader(fp)
    diagrams = []
    for page in reader.pages:
        for image in page.images:
            with open(image.name, "wb") as fp:
                fp.write(image.data)
                diagrams.append(image)
    return diagrams



def get_all_papers_related_to_topic():
    return 123
             
import glob
_ = [pdf_image_extract(fp) for fp in glob.glob('./*.pdf')]
   

#get_paper_google_scholar('IPC')


# In[ ]:


_[0][0].data


# In[ ]:


# from io import BytesIO
# for i in _:
#     for j in i:
#         image_display = Image.open(BytesIO(j.data))
#         image_display.show()


# In[ ]:


# fn_cache = {}

# def cacheThisFunction(func):
#     def _(*args):
#         key = func.__name__ + args.join(',')
#         if key in fn_cache: return fn_cache[key]
#         val = func(*args)
#         fn_cache[key] = val
#     return _

# @cacheThisFunction
# def _(_, __):
#     return _ + __

# _(0, 90008)


# In[ ]:


_ = (12312,31,231,23)


# In[ ]:


import random
# k = json['_k']
# v = json['_v']

cities = ['tokyo', 'houston', 'moscow', 'cairo', 'mumbai', 'delhi', 'shanghai', 'beijing', 'dhaka', 'osaka', 'chongqing', 'istanbul']
def rankApt(personCoefficentPreferences, apt):
    diff = 0
    print(apt, personCoefficentPreferences)
    for key in apt:
        diff += abs(apt[key] - personCoefficentPreferences[key])
    return diff 
cityAptChoice = {}
personCoefficentPreferences = {'commuteDistance': 0, 'library': 0, 'bar': 0, 'coffee': 0}

def makeApt():
    props = ['commuteDistance', 'library', 'bar', 'coffee'] 
    coeffs = {}
    for prop in props: coeffs[prop] = random.random()
    return coeffs

cities = {
    'tokyo': [makeApt() for i in range(5)],
    'houston': [makeApt() for i in range(5)],
    'moscow': [makeApt() for i in range(5)],
}

for city_name in cities:
   apt_list = cities[city_name]
   #print(apt_list)
   sorted(apt_list, key=lambda apt: rankApt(personCoefficentPreferences, apt))
   cityAptChoice[city_name] = apt_list
#10,000 cities
#1 week stays - 4 per city
# group size of 30,000 
#work on parsing and make the whole thing cool
#{japan: airbnbURL, india: airbnbURL, canada: airbnbURL}
#return cityAptChoice


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:







import random
import math

def getCounter(typ):
    type_counters = {
    "Normal": ["Fighting"],
    "Fire": ["Water", "Rock", "Ground"],
    "Water": ["Electric", "Grass"],
    "Electric": ["Ground"],
    "Grass": ["Fire", "Flying", "Bug", "Poison"],
    "Ice": ["Fire", "Fighting", "Steel", "Rock"],
    "Fighting": ["Flying", "Psychic", "Fairy"],
    "Poison": ["Ground", "Psychic"],
    "Ground": ["Water", "Ice", "Grass"],
    "Flying": ["Electric", "Ice", "Rock"],
    "Psychic": ["Bug", "Ghost", "Dark"],
    "Bug": ["Fire", "Flying", "Rock"],
    "Rock": ["Water", "Grass", "Fighting", "Steel", "Ground"],
    "Ghost": ["Ghost", "Dark"],
    "Dragon": ["Ice", "Dragon", "Fairy"],
    "Dark": ["Fighting", "Bug", "Fairy"],
    "Steel": ["Fire", "Fighting", "Ground"],
    "Fairy": ["Poison", "Steel"]
    }
    pokemon_types = [
    ("Bulbasaur", "Grass", "Poison"),
    ("Ivysaur", "Grass", "Poison"),
    ("Venusaur", "Grass", "Poison"),
    ("Charmander", "Fire", None),
    ("Charmeleon", "Fire", None),
    ("Charizard", "Fire", "Flying"),
    ("Squirtle", "Water", None),
    ("Wartortle", "Water", None),
    ("Blastoise", "Water", None),
    ("Caterpie", "Bug", None),
    ("Metapod", "Bug", None),
    ("Butterfree", "Bug", "Flying"),
    ("Weedle", "Bug", "Poison"),
    ("Kakuna", "Bug", "Poison"),
    ("Beedrill", "Bug", "Poison"),
    ("Pidgey", "Normal", "Flying"),
    ("Pidgeotto", "Normal", "Flying"),
    ("Pidgeot", "Normal", "Flying"),
    ("Rattata", "Normal", None),
    ("Raticate", "Normal", None),
    ("Spearow", "Normal", "Flying"),
    ("Fearow", "Normal", "Flying"),
    ("Ekans", "Poison", None),
    ("Arbok", "Poison", None),
    ("Pikachu", "Electric", None),
    ("Raichu", "Electric", None),
    ("Sandshrew", "Ground", None),
    ("Sandslash", "Ground", None),
    ("Nidoran♀", "Poison", None),
    ("Nidorina", "Poison", None),
    ("Nidoqueen", "Poison", "Ground"),
    ("Nidoran♂", "Poison", None),
    ("Nidorino", "Poison", None),
    ("Nidoking", "Poison", "Ground"),
    ("Clefairy", "Fairy", None),
    ("Clefable", "Fairy", None),
    ("Vulpix", "Fire", None),
    ("Ninetales", "Fire", None),
    ("Jigglypuff", "Normal", "Fairy"),
    ("Wigglytuff", "Normal", "Fairy"),
    ("Zubat", "Poison", "Flying"),
    ("Golbat", "Poison", "Flying"),
    ("Oddish", "Grass", "Poison"),
    ("Gloom", "Grass", "Poison"),
    ("Vileplume", "Grass", "Poison"),
    ("Paras", "Bug", "Grass"),
    ("Parasect", "Bug", "Grass"),
    ("Venonat", "Bug", "Poison"),
    ("Venomoth", "Bug", "Poison"),
    ("Diglett", "Ground", None),
    ("Dugtrio", "Ground", None),
    ("Meowth", "Normal", None),
    ("Persian", "Normal", None),
    ("Psyduck", "Water", None),
    ("Golduck", "Water", None),
    ("Mankey", "Fighting", None),
    ("Primeape", "Fighting", None),
    ("Growlithe", "Fire", None),
    ("Arcanine", "Fire", None),
    ("Poliwag", "Water", None),
    ("Poliwhirl", "Water", None),
    ("Poliwrath", "Water", "Fighting"),
    ("Abra", "Psychic", None),
    ("Kadabra", "Psychic", None),
    ("Alakazam", "Psychic", None),
    ("Machop", "Fighting", None),
    ("Machoke", "Fighting", None),
    ("Machamp", "Fighting", None),
    ("Bellsprout", "Grass", "Poison"),
    ("Weepinbell", "Grass", "Poison"),
    ("Victreebel", "Grass", "Poison"),
    ("Tentacool", "Water", "Poison"),
    ("Tentacruel", "Water", "Poison"),
    ("Geodude", "Rock", "Ground"),
    ("Graveler", "Rock", "Ground"),
    ("Golem", "Rock", "Ground"),
    ("Ponyta", "Fire", None),
    ("Rapidash", "Fire", None),
    ("Slowpoke", "Water", "Psychic"),
    ("Slowbro", "Water", "Psychic"),
    ("Magnemite", "Electric", "Steel"),
    ("Magneton", "Electric", "Steel"),
    ("Farfetch'd", "Normal", "Flying"),
    ("Doduo", "Normal", "Flying"),
    ("Dodrio", "Normal", "Flying"),
    ("Seel", "Water", None),
    ("Dewgong", "Water", "Ice"),
    ("Grimer", "Poison", None),
    ("Muk", "Poison", None),
    ("Shellder", "Water", None),
    ("Cloyster", "Water", "Ice"),
    ("Gastly", "Ghost", "Poison"),
    ("Haunter", "Ghost", "Poison"),
    ("Gengar", "Ghost", "Poison"),
    ("Onix", "Rock", "Ground"),
    ("Drowzee", "Psychic", None),
    ("Hypno", "Psychic", None),
    ("Krabby", "Water", None),
    ("Kingler", "Water", None),
    ("Voltorb", "Electric", None),
    ("Electrode", "Electric", None),
    ("Exeggcute", "Grass", "Psychic"),
    ("Exeggutor", "Grass", "Psychic"),
    ("Cubone", "Ground", None),
    ("Marowak", "Ground", None),
    ("Hitmonlee", "Fighting", None),
    ("Hitmonchan", "Fighting", None),
    ("Lickitung", "Normal", None),
    ("Koffing", "Poison", None),
    ("Weezing", "Poison", None),
    ("Rhyhorn", "Ground", "Rock"),
    ("Rhydon", "Ground", "Rock"),
    ("Chansey", "Normal", None),
    ("Tangela", "Grass", None),
    ("Kangaskhan", "Normal", None),
    ("Horsea", "Water", None),
    ("Seadra", "Water", None),
    ("Goldeen", "Water", None),
    ("Seaking", "Water", None),
    ("Staryu", "Water", None),
    ("Starmie", "Water", "Psychic"),
    ("Mr. Mime", "Psychic", "Fairy"),
    ("Scyther", "Bug", "Flying"),
    ("Jynx", "Ice", "Psychic"),
    ("Electabuzz", "Electric", None),
    ("Magmar", "Fire", None),
    ("Pinsir", "Bug", None),
    ("Tauros", "Normal", None),
    ("Magikarp", "Water", None),
    ("Gyarados", "Water", "Flying"),
    ("Lapras", "Water", "Ice"),
    ("Ditto", "Normal", None),
    ("Eevee", "Normal", None),
    ("Vaporeon", "Water", None),
    ("Jolteon", "Electric", None),
    ("Flareon", "Fire", None),
    ("Porygon", "Normal", None),
    ("Omanyte", "Rock", "Water"),
    ("Omastar", "Rock", "Water"),
    ("Kabuto", "Rock", "Water"),
    ("Kabutops", "Rock", "Water"),
    ("Aerodactyl", "Rock", "Flying"),
    ("Snorlax", "Normal", None),
    ("Articuno", "Ice", "Flying"),
    ("Zapdos", "Electric", "Flying"),
    ("Moltres", "Fire", "Flying"),
    ("Dratini", "Dragon", None),
    ("Dragonair", "Dragon", None),
    ("Dragonite", "Dragon", "Flying"),
    ("Mewtwo", "Psychic", None),
    ("Mew", "Psychic", None),
    # Generation 2
    ("Chikorita", "Grass", None),
    ("Bayleef", "Grass", None),
    ("Meganium", "Grass", None),
    ("Cyndaquil", "Fire", None),
    ("Quilava", "Fire", None),
    ("Typhlosion", "Fire", None),
    ("Totodile", "Water", None),
    ("Croconaw", "Water", None),
    ("Feraligatr", "Water", None),
    ("Sentret", "Normal", None),
    ("Furret", "Normal", None),
    ("Hoothoot", "Normal", "Flying"),
    ("Noctowl", "Normal", "Flying"),
    ("Ledyba", "Bug", "Flying"),
    ("Ledian", "Bug", "Flying"),
    ("Spinarak", "Bug", "Poison"),
    ("Ariados", "Bug", "Poison"),
    ("Crobat", "Poison", "Flying"),
    ("Chinchou", "Water", "Electric"),
    ("Lanturn", "Water", "Electric"),
    ("Pichu", "Electric", None),
    ("Cleffa", "Fairy", None),
    ("Igglybuff", "Normal", "Fairy"),
    ("Togepi", "Fairy", None),
    ("Togetic", "Fairy", "Flying"),
    ("Natu", "Psychic", "Flying"),
    ("Xatu", "Psychic", "Flying"),
    ("Mareep", "Electric", None),
    ("Flaaffy", "Electric", None),
    ("Ampharos", "Electric", None),
    ("Bellossom", "Grass", None),
    ("Marill", "Water", "Fairy"),
    ("Azumarill", "Water", "Fairy"),
    ("Sudowoodo", "Rock", None),
    ("Politoed", "Water", None),
    ("Hoppip", "Grass", "Flying"),
    ("Skiploom", "Grass", "Flying"),
    ("Jumpluff", "Grass", "Flying"),
    ("Aipom", "Normal", None),
    ("Sunkern", "Grass", None),
    ("Sunflora", "Grass", None),
    ("Yanma", "Bug", "Flying"),
    ("Wooper", "Water", "Ground"),
    ("Quagsire", "Water", "Ground"),
    ("Espeon", "Psychic", None),
    ("Umbreon", "Dark", None),
    ("Murkrow", "Dark", "Flying"),
    ("Slowking", "Water", "Psychic"),
    ("Misdreavus", "Ghost", None),
    ("Unown", "Psychic", None),
    ("Wobbuffet", "Psychic", None),
    ("Girafarig", "Normal", "Psychic"),
    ("Pineco", "Bug", None),
    ("Forretress", "Bug", "Steel"),
    ("Dunsparce", "Normal", None),
    ("Gligar", "Ground", "Flying"),
    ("Steelix", "Steel", "Ground"),
    ("Snubbull", "Fairy", None),
    ("Granbull", "Fairy", None),
    ("Qwilfish", "Water", "Poison"),
    ("Scizor", "Bug", "Steel"),
    ("Shuckle", "Bug", "Rock"),
    ("Heracross", "Bug", "Fighting"),
    ("Sneasel", "Dark", "Ice"),
    ("Teddiursa", "Normal", None),
    ("Ursaring", "Normal", None),
    ("Slugma", "Fire", None),
    ("Magcargo", "Fire", "Rock"),
    ("Swinub", "Ice", "Ground"),
    ("Piloswine", "Ice", "Ground"),
    ("Corsola", "Water", "Rock"),
    ("Remoraid", "Water", None),
    ("Octillery", "Water", None),
    ("Delibird", "Ice", "Flying"),
    ("Mantine", "Water", "Flying"),
    ("Skarmory", "Steel", "Flying"),
    ("Houndour", "Dark", "Fire"),
    ("Houndoom", "Dark", "Fire"),
    ("Kingdra", "Water", "Dragon"),
    ("Phanpy", "Ground", None),
    ("Donphan", "Ground", None),
    ("Porygon2", "Normal", None),
    ("Stantler", "Normal", None),
    ("Smeargle", "Normal", None),
    ("Tyrogue", "Fighting", None),
    ("Hitmontop", "Fighting", None),
    ("Smoochum", "Ice", "Psychic"),
    ("Elekid", "Electric", None),
    ("Magby", "Fire", None),
    ("Miltank", "Normal", None),
    ("Blissey", "Normal", None),
    ("Raikou", "Electric", None),
    ("Entei", "Fire", None),
    ("Suicune", "Water", None),
    ("Larvitar", "Rock", "Ground"),
    ("Pupitar", "Rock", "Ground"),
    ("Tyranitar", "Rock", "Dark"),
    ("Lugia", "Psychic", "Flying"),
    ("Ho-oh", "Fire", "Flying"),
    ("Celebi", "Psychic", "Grass")
    ]
    counters = type_counters[typ]
    #once a type has been countered
    poss = [pokemon[0] for pokemon in pokemon_types 
            if pokemon[1] in counters 
            or pokemon[2] in counters
           ]
    return poss[math.floor(random.random() * len(poss))]



def generate_team(player_choice='mew'):
    type_counters = {
    "Normal": ["Fighting"],
    "Fire": ["Water", "Rock", "Ground"],
    "Water": ["Electric", "Grass"],
    "Electric": ["Ground"],
    "Grass": ["Fire", "Flying", "Bug", "Poison"],
    "Ice": ["Fire", "Fighting", "Steel", "Rock"],
    "Fighting": ["Flying", "Psychic", "Fairy"],
    "Poison": ["Ground", "Psychic"],
    "Ground": ["Water", "Ice", "Grass"],
    "Flying": ["Electric", "Ice", "Rock"],
    "Psychic": ["Bug", "Ghost", "Dark"],
    "Bug": ["Fire", "Flying", "Rock"],
    "Rock": ["Water", "Grass", "Fighting", "Steel", "Ground"],
    "Ghost": ["Ghost", "Dark"],
    "Dragon": ["Ice", "Dragon", "Fairy"],
    "Dark": ["Fighting", "Bug", "Fairy"],
    "Steel": ["Fire", "Fighting", "Ground"],
    "Fairy": ["Poison", "Steel"]
    }
    types = list(type_counters.keys())
    elite_four = types[6:12]
    team = [player_choice]
    for typ in elite_four:
        team.append(getCounter(typ))
    return team


for i in range(6):
    print(generate_team())
    print('\n')
#look at move types
#downlaod transcripts of competitive pokemon
#find like move counters -> get program to actually understand metagame
#or get user to understand meta game -> 


# In[ ]:


cities = [
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

#write these by hand -> use h3 -> fine tune codeLLama + other ones 
def proximity_to_ocean():
    return 123123
        
def historical():
    return 555

def hipster():
    return 123

def laid_back():
    return 555

def commerical():
    return 1231


def getAirbnbGeoCoordinate(): return 1231231
    #get all 
    #https://maps.googleapis.com/maps/vt?pb=!1m5!1m4!1i14!2i11708!3i6832!4i256!2m3!1e0!2sm!3i662403149!3m17!2sen!3sUS!5e18!12m4!1e68!2m2!1sset!2sRoadmap!12m3!1e37!2m1!1ssmartmaps!12m4!1e26!2m2!1sstyles!2zcy50OjF8cy5lOmwudC5mfHAuYzojZmY4Zjk3ODcscy50OjF8cy5lOmwudC5zfHAudzoxLHMudDoxN3xzLmU6bC50LmZ8cC5jOiNmZjRiNGU0YyxzLnQ6MTd8cy5lOmwudC5zfHAudzoxLHMudDoxOXxzLmU6bHxwLnY6b2ZmfHo6NXxwLnY6b258ejo2fHAudjpvbnx6Ojd8cC52Om9uLHMudDoxOXxzLmU6bC50LmZ8cC5jOiNmZjRiNGU0Y3x6OjV8cC5jOiNmZjU3NjA1MnxwLnY6b258ejoxMnxwLmM6I2ZmNmE2ZDcwLHMudDoxOXxzLmU6bC50LnN8cC52Om9mZnxwLnc6MSxzLnQ6MjB8cy5lOmwudHxwLnY6b24scy50OjIwfHMuZTpsLnQuZnxwLmM6I2ZmYWNiMmI5LHMudDoyMHxzLmU6bC50LnN8cC53OjIscy50OjE4fHMuZTpsLnQuZnxwLmM6I2ZmNGI0ZTRjfHo6N3xwLmM6I2ZmNjM2ZTdlLHMudDoxOHxzLmU6bC50LnN8cC53OjEscy50OjgxfHMuZTpnLmZ8cC5jOiNmZmY4ZjRmMSxzLnQ6ODF8cy5lOmwudHxwLnY6b2ZmLHMudDoxMjk3fHMuZTpnLmZ8cC5jOiNmZmU4ZWFlZHx6OjE3fHAudjpvZmZ8ejoxOHxwLmM6I2ZmZjNlY2U3fHAudjpvbnx6OjIwfHAudjpvbixzLnQ6MTI5N3xzLmU6Zy5zfHAuYzojZmZkYWRjZTJ8cC52Om9ufHo6MTd8cC52Om9mZnx6OjE4fHAudjpvZmZ8ejoxOXxwLmM6I2ZmYmRiZGJjfHAudjpvbnx6OjIwfHAuYzojZmZiYWJhYmF8cC52Om9uLHMudDoxMjk5fHMuZTpnLmZ8cC5jOiNmZmZmZTBjY3x6OjE0fHAuYzojZmZmYWU2ZGJ8ejoxNXxwLmM6I2ZmZmJlYWUwLHMudDo4MnxzLmU6Zy5mfHAubDozNnxwLnM6MyxzLnQ6ODJ8cy5lOmwudHxwLnY6b2ZmLHMudDoyfHMuZTpnLmZ8cC5jOiNmZmQ2ZWNjNyxzLnQ6MnxzLmU6bHxwLnY6b2ZmLHMudDozN3xzLmU6Zy5mfHAuYzojZmZmOGY1ZWUscy50OjM2fHMuZTpnLmZ8cC5jOiNmZmY4ZjVlZSxzLnQ6NDB8cy5lOmcuZnxwLmM6I2ZmZGNmMmNkLHMudDozOHxzLmU6Zy5mfHAuYzojZmZmNmY0ZWUscy50OjM1fHMuZTpnLmZ8cC5jOiNmZmY4ZjVlZSxzLnQ6Mzl8cy5lOmcuZnxwLmM6I2ZmZGNmMmNkLHMudDozfHAudjpvZmYscy50OjUwfHMuZTpnLmZ8cC5jOiNmZmY5ZmFmYnxwLnY6b2ZmfHo6MTB8cC52Om9mZnx6OjExfHAuYzojZmZmY2ZjZmN8cC52Om9ufHo6MTJ8cC5jOiNmZmZmZmZmZnxwLnY6b258ejoxNHxwLnY6b24scy50OjUwfHMuZTpnLnN8ejoxM3xwLmM6I2ZmYjViMmIwfHAudjpvbnx6OjE0fHAuYzojZmZkZGQ4ZDV8cC52Om9ufHo6MTV8cC5jOiNmZmU0ZTBkZHxwLnY6b24scy50OjUwfHMuZTpsLnR8ejoxM3xwLnY6b2ZmfHo6MTR8cC52Om9uLHMudDo1MHxzLmU6bC50LmZ8cC5jOiNmZmEzYTNhMixzLnQ6NDl8cy5lOmcuZnxwLmM6I2ZmZmZmZmZmfHAudjpvZmZ8ejo2fHAuYzojZmZlZmZiZWF8cC52Om9mZnx6Ojd8cC5jOiNmZmZmZmZmZnxwLnY6b258ejo4fHAuYzojZmZmZmZmZmZ8cC52Om9ufHo6OXxwLmM6I2ZmZmZmZmZmfHAudjpvbnx6OjEzfHAuYzojZmZmZmZmZmYscy50OjQ5fHMuZTpnLnN8cC5jOiNmZmQ0ZDhkZHxwLnY6b2ZmfHo6N3xwLmM6I2ZmZTllNmUyfHAudjpvbnx6Ojh8cC5jOiNmZmU5ZTZlMnxwLnY6b258ejo5fHAuYzojZmZlOWU2ZTJ8cC52Om9ufHo6MTB8cC5jOiNmZmUwZGNkN3xwLnY6b258ejoxMXxwLmM6I2ZmZTNkZGQ5fHAudjpvbnxwLnc6MXx6OjEyfHAuYzojZmZjZmM4YzR8cC52Om9ufHo6MTR8cC5jOiNmZmJmYmNiYnxwLnc6MSxzLnQ6NDl8cy5lOmwuaXxwLnY6b2ZmfHo6MTF8ejoxM3xwLnY6b2ZmfHo6MTV8cC5sOjE1fHAudjpvbixzLnQ6NDl8cy5lOmwudC5mfHAuYzojZmY5YmE0YjAscy50OjUxfHMuZTpnLmZ8cC5jOiNmZmYzZjRmNnx6OjEzfHAuYzojZmZmZmZmZmZ8cC52Om9ufHo6MTR8cC5jOiNmZmZmZmZmZnxwLnc6MXx6OjE1fHAuYzojZmZmZmZmZmZ8cC53OjJ8ejoxNnxwLmM6I2ZmZmZmZmZmfHAudzozfHo6MTd8cC53OjR8ejoxOHxwLnc6NSxzLnQ6NTF8cy5lOmcuc3xwLmM6I2ZmZDNkMmNmfHo6MTN8cC52Om9mZnx6OjE2fHAuYzojZmZmNWY1ZjV8ejoxN3xwLnY6b2ZmLHMudDo1MXxzLmU6bC50fHAudjpvZmZ8ejoxNHxwLnY6b2ZmfHo6MTZ8cC52Om9uLHMudDo1MXxzLmU6bC50LmZ8cC5jOiNmZmEzYTNhMixzLnQ6ODE4fHMuZTpnfHAudzoxfHo6MTd8cC53OjIscy50OjgxOHxzLmU6Zy5mfHAuYzojZmZjNGQ4YjZ8cC52Om9uLHMudDo2NXxwLnc6MSxzLnQ6NjV8cy5lOmwudC5mfHAuYzojZmY3ODY4NmIscy50OjY1fHMuZTpsLnQuc3xwLmM6I2ZmZmZmZmZmfHAudzo0LHMudDoxMDQyfHAudjpvZmYscy50OjEwNDJ8cy5lOmcuZnxwLmM6I2ZmNzNiYWQ0LHMudDoxMDQxfHMuZTpnLmZ8cC5jOiNmZmM0YWJhZnxwLnY6b2ZmfHo6MTN8cC5jOiNmZmRjZDZkN3xwLnY6b258ejoxNXxwLmM6I2ZmY2ZiZmMyfHAudjpvbnx6OjE2fHAuYzojZmZjNGFiYWZ8ejoxOHxwLmM6I2ZmODc3Mzc2LHMudDoxMDQxfHMuZTpnLnN8cC5jOiNmZmUyZDVkN3xwLnY6b2ZmfHAudzoxfHo6MTV8cC52Om9uLHMudDo2NnxzLmU6bHxwLnY6b24scy50OjY2fHMuZTpsLml8cC52Om9mZnx6OjE1fHAubDoyNHxwLnM6LTEwMHxwLnY6b258ejoxNnxwLmw6MTl8cC5zOi0xMDB8ejoxN3xwLmw6OXxwLnM6LTEwMCxzLnQ6NjZ8cy5lOmwudC5mfHAuYzojZmY2MzZlN2V8cC52Om9uLHMudDoxMDU5fHMuZTpnLmZ8cC5sOi0xfHAuczotMTAwLHMudDoxMDU4fHMuZTpsfHAudjpvZmZ8ejoxNnxwLnY6b24scy50OjEwNTh8cy5lOmwuaXxwLnY6b2ZmfHo6MTd8cC52Om9uLHMudDoxMDU3fHMuZTpnLmZ8cC5jOiNmZjQyODVmNHxwLnY6b24scy50OjEwNTd8cy5lOmwudHxwLnY6b2ZmfHo6MTZ8cC52Om9mZnx6OjE3fHAudjpvbixzLnQ6MTA1N3xzLmU6bC50LmZ8cC5jOiNmZjc4Nzg3OCxzLnQ6NnxzLmU6Zy5mfHAuYzojZmZiM2U2ZjQscy50OjZ8cy5lOmcuc3xwLnY6b24scy50OjZ8cy5lOmwudC5mfHAuYzojZmY0ZDkyYTMscy50OjZ8cy5lOmwudC5zfHAudjpvZmY!4e0!5m1!5f2!23i1379903!23i1376099&key=AIzaSyCrpUPhpbPzRI4hYC7xE02WKsrxQv0HClI&token=98506
    #OCR the image using python



#twitter - historical 
#yelp - maybe
#4square
#airbnb


#user puppeteer.js -> get map tile -> 


#better curate travel with data -> allow user to customize what data they find important or relevant


#realistically -> get list of 12 cities
#run functions on those 12 cities to get what critiera you can actually sort them by
#once apt are sorted, the user's chosen weight to each coefficent is whats used to calculate the filter 


#for each city / get list of places -> get all reviews -> find words that are commonly repeated
#for lots of tweets in each city -> 1m tweets 
#the idea is if you can write comments fast -> computer can generate working code as fast in a few years 
#why is codex so fast -> but codellama is slow ? 
#codex is not open source -> gpt4 is expensive after fine tuning -> not sure how much it can be customized 

criteria_functions = ['proximity_to_ocean', 'proximity_to_supermarket', 'family_oriented', 'historical' , 'hipster', 'laid_back', 'commerical']
criteria = ['proximity_to_ocean', 'proximity_to_supermarket', 'family_oriented', 'historical' , 'hipster', 'laid_back', 'commerical']
#https://pudding.cool/2018/03/neighborhoods/

import os
def getAllAirbnb(cityPicks, chosenCriteria):
    cities = cityPicks
    os.subprocess('node fetchAirbnb ' + ' '.join(cities)) #writes a json to /data/airbnb_city.json
    
    #list of list of json objects representing an appartment
    cities = {city: json.load(f'/data/airbnb_{city}') for city in cities}
    for city in cities:
        for apt in city:
            for idx, criteria in enumerate(commerical):
                criteria_functions[idx](apt)
    return cities


#https://en.wikipedia.org/wiki/List_of_largest_cities


# In[ ]:


url = 'https://www.airbnb.com/rooms/46167540?adults=1&children=0&enable_m3_private_room=true&infants=0&pets=0&check_in=2023-09-17&check_out=2023-09-22&source_impression_id=p3_1694700080_UjFiBEV3n8iEBFMu&previous_page_section_name=1000&federated_search_id=c852781b-8f44-4187-a236-0aeb4c288fa8'


import requests
from bs4 import BeautifulSoup

response = requests.get(url)
response.text


if response.status_code == 200:
    html_content = response.text
else:
    print(f"Failed to retrieve the website with status code: {response.status_code}")
    exit()

# Initialize a BeautifulSoup object
soup = BeautifulSoup(html_content, 'html.parser')

# Find all image tags
img_tags = soup.find_all('*')

# Loop to print out all image sources
for img in img_tags:
    print(img)
    
    
#with open('airbnb.html')response.text
print(soup)


# In[ ]:


# from PIL import Image
# import matplotlib
# import pytesseract

# # If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# # Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# # Simple image to string
# print(pytesseract.image_to_string(Image.open('test.png')))

# # In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# # NOTE: In this case you should provide tesseract supported images or tesseract will return error
# print(pytesseract.image_to_string('test.png'))

# # List of available languages
# print(pytesseract.get_languages(config=''))

# # French text image to string
# print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))

# # Batch processing with a single file containing the list of multiple image file paths
# print(pytesseract.image_to_string('images.txt'))

# # Timeout/terminate the tesseract job after a period of time
# try:
#     print(pytesseract.image_to_string('test.jpg', timeout=2)) # Timeout after 2 seconds
#     print(pytesseract.image_to_string('test.jpg', timeout=0.5)) # Timeout after half a second
# except RuntimeError as timeout_error:
#     # Tesseract processing is terminated
#     pass

# # Get bounding box estimates
# print(pytesseract.image_to_boxes(Image.open('test.png')))

# # Get verbose data including boxes, confidences, line and page numbers
# print(pytesseract.image_to_data(Image.open('test.png')))

# # Get information about orientation and script detection
# print(pytesseract.image_to_osd(Image.open('test.png')))

# # Get a searchable PDF
# pdf = pytesseract.image_to_pdf_or_hocr('test.png', extension='pdf')
# with open('test.pdf', 'w+b') as f:
#     f.write(pdf) # pdf type is bytes by default

# # Get HOCR output
# hocr = pytesseract.image_to_pdf_or_hocr('test.png', extension='hocr')

# # Get ALTO XML output
# xml = pytesseract.image_to_alto_xml('test.png')
# Support for OpenCV image/NumPy array objects

# import cv2

# img_cv = cv2.imread(r'/<path_to_image>/digits.png')

# # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# # we need to convert from BGR to RGB format/mode:
# img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
# print(pytesseract.image_to_string(img_rgb))
# # OR
# img_rgb = Image.frombytes('RGB', img_cv.shape[:2], img_cv, 'raw', 'BGR', 0, 0)
# print(pytesseract.image_to_string(img_rgb))


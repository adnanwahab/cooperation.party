#!/usr/bin/env python
# coding: utf-8

# In[76]:


import requests

url = 'https://www.airbnb.com/rooms/12936?source_impression_id=p3_1696720714_gm3QqayS6GfiSfBc'

res = requests.get(url)

res.text


# In[84]:


columns

#neighborhood ID = hex
#find which variables correlate - number_of_reviews
#del minimum_nights, last_review, host_name, host_id, host_listings_count
#filter price


# In[73]:


import os
import csv
import json
import re 
for fp in os.listdir('../data/csv'):
    with open('../data/csv/' + fp, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        file = {}
        for row in reader:
            _id = row['id']
            del row['id']
            file[_id] = row
            
        fp = re.sub(r' |,', '-', fp).replace('.csv', '.json')
        print(fp)
        json.dump(file, open('../data/columns/' + fp, 'w+'))


# In[10]:


get_ipython().system(' mkdir ../data/columns')


# In[73]:


import os
import csv
import json
import re 
for fp in os.listdir('../data/csv'):
    with open('../data/csv/' + fp, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        file = {}
        for row in reader:
            _id = row['id']
            del row['id']
            file[_id] = row
            
        fp = re.sub(r' |,', '-', fp).replace('.csv', '.json')
        print(fp)
        json.dump(file, open('../data/columns/' + fp, 'w+'))
            #print(row['first_name'], row['last_name'])


# In[89]:


import os

#which factors correlate with proice

# substract neighborhood
# reviews per month

def convertToLinearRegression(row):
    result = {}
    props = row['name'].split(' · ')
    if (len(props) > 2):
        rooms = props[2][0]
    else: rooms = 0
    #lavatory = props[4][0]
    #print(rooms)
    result['room_type'] = 1 if row['room_type'] == 'Entire home/apt' else 2
    result['rooms'] = rooms
    result['price'] = row['price'] #indep variable
    result['number_of_reviews'] = row['price'] #indep variable
    result['availability_365'] = row['availability_365'] #indep variable
    result['hex_id'] = 0 #indep variable
    return result
    
for fp in os.listdir('../data/columns')[:1]:
    columns = json.load(open('../data/columns/' + fp))
    results = []
    for row in columns: 
        results.append(convertToLinearRegression(columns[row]))


# In[110]:


from torch import nn 
from fastprogress.fastprogress import progress_bar
from fastprogress.fastprogress import master_bar 
dev = 'cuda:0'
dev = 'cpu'
def good_deals_per_city(city_columns, name):
    print(name)
    def makeIndep(item):
        result = []
        for key in item:
            if key != 'price':
                #print(key, item[key])
                alphabet = 'Z X C V B N M A S D F G H J K L Q W E R T Y U I O P'.split(' ')
                if (item[key] in alphabet): item[key] = 0
                result.append(int(item[key]))
        return result

    t_dep = torch.tensor([int(result['price']) for result in city_columns]).float()
    t_indep = torch.tensor([makeIndep(result) for result in city_columns])
    def test_prediction(test_predictions):
        ctrl = test_predictions.sum(1).tolist()[0]
        isFalse = len([sum(row) for idx, row in enumerate(test_predictions.tolist()) if sum(row) <= ctrl and t_dep[idx] == 0])
        isTrue = len([sum(row) for idx, row in enumerate(test_predictions.tolist()) if sum(row) > ctrl and t_dep[idx] == 1])
        allFalse = len([sum(row) for idx, row in enumerate(test_predictions.tolist()) if t_dep[idx] == 0])
        allTrue = len([sum(row) for idx, row in enumerate(test_predictions.tolist()) if t_dep[idx] == 1])
        return (isFalse / allFalse, isTrue / allTrue, isFalse, isTrue)
    def plot_loss(l):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 4))
        legends = []
        plt.plot(l) 
        plt.plot([0, len([i for k,i in enumerate(rowGeneExpression.values()) if dependent_variables[k]])], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
        plt.legend(legends);

    #numerical_values = df.select_dtypes(include=[int, float]).values.tolist()
    #t_indep = torch.Tensor(numerical_values).to(dev)
    # t_indep = t_indep / vals
    # λλλλλ.requires_grad_(True)
    #3 variations, test, t_indep and t_indep+embedding
    resultant_tensor = t_indep
    #encodedOutput.requires_grad_(True)
    #resultant_tensor = torch.cat((t_indep.to(dev),tensor.to(dev)), 1)
    #resultant_tensor = λλλλλ
    #resultant_tensor = tensor
    #t_indep = t_indep / t_indep.max()
    t_dep = t_dep / t_dep.max()
    t_indep = t_indep / t_indep.max()

    dim = t_indep.shape[1]

    model = torch.nn.Sequential(
        torch.nn.Linear(dim,dim),
        nn.ReLU(),
        nn.Linear(dim,dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.Sigmoid()
    ).to(dev)

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=.1, 
        weight_decay=0.01
    )
    # model = model.to(device)
    # input_tensor = input_tensor.to(device)

    n_iterations = 1000
    loss_track = []
    accuracy_track = []
    no_entropy = []
    loss_function = torch.nn.BCELoss()
    def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss):
        """ dynamically print the loss plot during the training/validation loop.
            expects epoch to start from 1.
        """
        x = range(1, epoch+1)
        y = np.concatenate((train_loss, valid_loss))
        graphs = [[x,train_loss], [x,valid_loss]]
        x_margin = 0.2
        y_margin = 0.05
        x_bounds = [1-x_margin, epochs+x_margin]
        y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]

        mb.update_graph(graphs, x_bounds, y_bounds)
    mb = master_bar(range(1))
    def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss):
        """ dynamically print the loss plot during the training/validation loop.
            expects epoch to start from 1.
        """
        x = range(1, epoch+1)
        y = np.concatenate((train_loss, valid_loss))
        print(x,y)
        graphs = [[x,train_loss], [x,valid_loss]]
        x_margin = 0.2
        y_margin = 0.05
        x_bounds = [1-x_margin, epochs+x_margin]
        y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]
        print(x_bounds, y_bounds)
        mb.update_graph(graphs, x_bounds, y_bounds)

    #diff expected price
    #1 million airbnbs -> give each a 
    #adapt NN to 30% of airbnbs -> 600,000 pricing estimates 
    #plot this on a notebook
    #in these cities you can find good deals -> travelers + immigrants
    #in these cities you can buy a home and make an airbnb and it will make lots of money -> SMBs
    def pricing_prediction():
        for i in mb:    
        #for j in progress_bar(range(2000), parent=mb):
            for j in progress_bar(range(2000)):
                loss = loss_function(model(t_indep).sum(1).sigmoid(), t_dep.to(dev))
                optimizer.zero_grad()  # 3
                loss.backward(retain_graph=True)  # 4
                optimizer.step()  # 5
                if j == 1 or j % 50 == 0:
                    test_predictions = model(t_indep)
                    #print(loss.item(), test_predictions.sum().item() / 8)
                    #print(test_prediction(test_predictions))
                loss_track.append(loss.item())
                accuracy_track.append(test_predictions.sum().item() / 8)
    #             print(loss_track[-1])
    #             print(accuracy_track[-1])

                #no_entropy += [test_predictions.sum().item() / 8]
                #         k = 100 * i + j
                #         x = np.arange(0, 2*k*np.pi/1000, 0.01)
                #         y1, y2 = np.cos(x), np.sin(x)
                #         graphs = [[x,y1], [x,y2]]
                #         x_bounds = [0, 2*np.pi]
                #         y_bounds = [-1,1]
                #         mb.update_graph(graphs, x_bounds, y_bounds)
                #         print(loss_track, accuracy_track)
                #print(loss_track, accuracy_track)
                #plot_loss_update(j, n_iterations, mb, loss_track, accuracy_track)
                #for batch in progress_bar(range(2), parent=mb): sleep(0.2)
    return model(t_indep)
    #get avg price of neighborhood
    #try to get coefficents of price vs factors -> 5 room = 5k, 4 room = 4k
    pricing_prediction()


# In[7]:


complaints = [
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Austin/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Baltimore/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Boston/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Chicago/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Dallas/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Denver/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Houston/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Las-Vegas/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Los-Angeles/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Louisville/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Memphis/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Miami/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Milwaukee/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Minneapolis/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Nashville/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-New-Orleans/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Oakland/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Philadelphia/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Phoenix/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Sacramento/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-San-Antonio/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-San-Diego/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-San-Francisco/",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Washington,-D.C",
    "https://andrew-friedman.github.io/jkan/datasets/311-City-of-Washington-DC/",
    "https://andrew-friedman.github.io/jkan/datasets/311-Kansas-City-Government/",
    "https://andrew-friedman.github.io/jkan/datasets/311-New-York-City-Government/",
    "https://andrew-friedman.github.io/jkan/datasets/311_requests_boston/",
    "https://andrew-friedman.github.io/jkan/datasets/311_requests_chicago/",
    "https://andrew-friedman.github.io/jkan/datasets/sample-dataset/",
    "https://andrew-friedman.github.io/jkan/datasets/sample-dataset1/"
]

from bs4 import BeautifulSoup
import requests
import re
cities_with_311 = {}
for uri in complaints:
    res = requests.get(uri)
    # Parse the HTML content
    soup = BeautifulSoup(res.text, 'html.parser')

    # Find elements with property="dcat:accessURL"
    elements = soup.select('[property="dcat:accessURL"]')
    
    # Print the found elements
    for element in elements:
        if 'CSV' in element.text:
            if element.text == 'Air Monitoring Stations CSV': continue
            #if element.text == 'Boston': continue
            key = element.text
            if key == '311 Service Requests CSV': continue
            key = re.sub(r' |\.|,', '-', key)
            key = key.replace('-CSV', '')
            cities_with_311[key] = element.attrs['href']


# In[10]:


for _ in cities_with_311.keys(): print(_)


# In[ ]:


for key in cities_with_311:
    res = requests.get(cities_with_311[key])
    text = res.text
    print(key)
    json.dump(text, open('../data_sets/' + key, 'w+'))


# In[ ]:


urls = {
'City-of-Boston': 'https://data.boston.gov/dataset/311-service-requests#:~:text=Preview-,DOWNLOAD,-311%20SERVICE%20REQUESTS%20%2D%202022'
'new-york': 'https://data.cityofnewyork.us/api/views/erm2-nwe9/rows.csv?accessType=DOWNLOAD',
'los-angeles': 'https://data.lacity.org/api/views/97z7-y5bt/rows.csv?accessType=DOWNLOAD',
'City-of-Washington--D-C': '',
    
'City-of-Phoenix': '',
    
'City-of-Miami':,
    
'Los-Angeles':'https://data.lacity.org/api/views/pvft-t768/rows.csv?accessType=DOWNLOAD'
}


for key in urls:
    res = requests.get(urls)
    text = res.text
    print(key)
    json.dump(text, open('../data_sets/' + key, 'w+'))


# In[20]:


get_ipython().system(' tail ../data_sets/poorly_labeled/*')


# In[30]:


all_csv = os.listdir('../data_sets/clearly_labeled_311/')
                     
all_csv


# In[35]:


all_apt


# In[37]:


get_ipython().system(' pip install pydeck')


# In[42]:


import pydeck as pdk
import pandas as pd

CPU_GRID_LAYER_DATA = (
    "https://raw.githubusercontent.com/uber-common/" "deck.gl-data/master/website/sf-bike-parking.json"
)
df = pd.read_json(CPU_GRID_LAYER_DATA)

# Define a layer to display on a map

layer = pdk.Layer(
    "GridLayer", df, pickable=True, extruded=True, cell_size=200, elevation_scale=4, get_position="COORDINATES",
)

view_state = pdk.ViewState(latitude=37.7749295, longitude=-122.4194155, zoom=11, bearing=0, pitch=45)

# Render
r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{position}\nCount: {count}"},)
r.to_html("grid_layer.html")


# In[24]:


print('123')


# In[4]:


def compute_avg_311(apt_json):
    for city in apt_json:
        h3_index_list = {}
        for apt in apt_json[city]:
            h3_index = h3.geo_to_h3(apt['latitude'], apt['longitude'], 7)
            apt['h3_index'] = h3_index
            if h3_index not in h3_index_list: h3_index_list[h3_index_list] = []
            h3_index_list[h3_index].append(apt)
        for h3_index in h3_index_list:
            total = 0
            for apt in h3_index_list[h3_index]:
                total += apt['price']
            total /= len(h3_index_list[h3_index])
            for apt in h3_index_list[h3_index]: apt['avg_price'] = average

def compute_travel_time(apt_json, coefficents):
    for city in apt_json:
        h3_index_list = {}
        for apt in apt_json[city]:
            h3_index = h3.geo_to_h3(apt['latitude'], apt['longitude'], 7)
            apt['h3_index'] = h3_index
            if h3_index not in h3_index_list: h3_index_list[h3_index] = []
            h3_index_list[h3_index].append(apt)
        for h3_index in h3_index_list:
            total = 0
            for apt in h3_index_list[h3_index]:
                total += apt['price']
            average /= len(h3_index_list[h3_index])
            for apt in h3_index_list[h3_index]: apt['avg_price'] = average


# In[4]:


import os, json, h3
all_apt = os.listdir('../data/airbnb/apt/')

# for _ in all_csv:
#     _ = _.split('-')
#         for token in _:
#//subjective metrics of land-usage, https://morphocode.com/the-making-of-morphocode-explorer/

def compute_deal_ranking(apt_json):
    for city in apt_json:
        h3_index_list = {}
        for key in apt_json[city]:
            apt = apt_json[city][key]
            h3_index = h3.geo_to_h3(float(apt['latitude']), float(apt['longitude']), 4)
            apt['h3_index'] = h3_index
            if h3_index not in h3_index_list: h3_index_list[h3_index] = []
            h3_index_list[h3_index].append(apt)
        for h3_index in h3_index_list:
            total = 0
            for apt in h3_index_list[h3_index]:
                total += float(apt['price'])
            average = total / len(h3_index_list[h3_index])
            for apt in h3_index_list[h3_index]: apt['avg_price'] = average
    return apt_json
                

def rankAptByFactors():
    print('rankAptByFactors')
    cities = os.listdir('../data/airbnb/apt')
    apt_json = {
        city: json.load(open(f'../data/columns/{city}')) for city in cities[:1]
    }
#     apt_json = compute_deal_ranking(apt_json)
#     _ = {"yoga": .5, "rock-climbing": .3, "park": 1 }
#     apt_json = compute_travel_time(apt_json, [])
    apt_json = compute_deal_ranking(apt_json)    
    #client side filters - so only recompute if change document
    #onchange document .3 milliseconds for 1-5million airbnbs
    #put OSM locally -> use a script to download and update post gres
    return apt_json
def getH3By311(csv):
    #convert csv to key-value map
    #convert lat lng to h3 index
    #add complaint_type to h3 index
    return {
        'complaint_type': { 'h3_index': 0}
    }
# simplified_list1 = [item.split('--')[0] for item in all_csv]
# simplified_list2 = [item.split('--')[0] for item in all_apt]
# intersection = set(simplified_list1).intersection(set(simplified_list2))
rankAptByFactors()


# In[13]:


os.listdir('../data/columns')
get_ipython().system(' mv ../data/columns/Tokyo--Kantō--Japan.json ../data/columns/Tokyo--Japan.json')


# In[36]:


intersection
#could add new york


# In[22]:


import pandas as pd
all_csv = os.listdir(filepath)


# In[18]:


import os 
import csv
import pandas as pd
os.listdir(filepath)
filepath = '../data_sets/poorly_labeled/'

os.listdir(filepath)

count = 0
for file_name in os.listdir(filepath):
    count = 0
    
def get_first_row():
    with open(filepath+file_name, newline='') as csvfile:
        for line in csvfile:
            count += 1
            print(line)
            if count > 2: continue
            print(line)
#         reader = csv.DictReader(csvfile)
#         for col in reader: print(col)


# In[215]:


parsed_cities = os.listdir('../data/columns')


has_311_and_airbnb = []
for city in cities_with_311:
    tokens = city.split(' ')
    for token in tokens:
        for other_city in parsed_cities:
            if token in other_city and 'CSV' in city and 'Air Monitoring' not in city:
                has_311_and_airbnb.append(city)


# In[217]:


has_311_and_airbnb =list(set(has_311_and_airbnb))


# In[220]:


has_311_and_airbnb


# In[143]:





# In[182]:


#makeIndep()
from collections import defaultdict
import h3
h3AveragePrice = defaultdict(int)
h3Counts = defaultdict(int)


_ = []
for i, fp in enumerate(os.listdir('../data/columns')[20:79]):
    columns = json.load(open('../data/columns/' + fp))
    results = []
    for row in columns:
        row = columns[row]
        h3Index = h3.geo_to_h3(float(row['latitude']), float(row['longitude']), 7)
        h3AveragePrice[h3Index] += float(row['price'])
        h3Counts[h3Index] += 1
        
for key in h3AveragePrice:
    h3AveragePrice[key] /= h3Counts[key]
    
        #results.append(convertToLinearRegression(columns[row]))
    #_.append(good_deals_per_city(results, fp))

goodDeals = defaultdict(list)
count = 0
for i, fp in enumerate(os.listdir('../data/columns')[20:79]):
    columns = json.load(open('../data/columns/' + fp))
    results = []
    for row in columns:
        row = columns[row]
        h3Index = h3.geo_to_h3(float(row['latitude']), float(row['longitude']), 7)
        if float(row['price']) < (.5 * h3AveragePrice[h3Index]):
            goodDeals[fp].append(row) 
            count += 1
count


# In[183]:


has_311_and_airbnb


# In[200]:


goodDeals.keys()

count = 0
# for city in goodDeals:
#     print(city, len(goodDeals[city]))
    
    

_= list(dict(goodDeals).items())

inorder = sorted(_, key= lambda _: -len(_[1]))


# In[201]:


for i in inorder:
    print(i[0], len(i[1]))


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [25, 40, 30, 55]

# Create the bar chart
plt.bar(categories, values)

# Customize the plot
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')

# Display the bar chart
plt.show()


# In[131]:


def getGoodDeals(tensor):
    
    return True

#[_.sum(dim=1).shape for _ in _]

avg = _[0].sum(dim=1).sum() / _[0].sum(dim=1).shape[0]
for i in _[0]: print(i.sum() > avg)


# In[ ]:


import defaultdict from collections
for i, fp in enumerate(os.listdir('../data/columns')[20:115]):
    columns = json.load(open('../data/columns/' + fp))
    results = []
    neighborhoodAvg = defaultdict(int)
    for row in columns: 
        results.append(convertToLinearRegression(columns[row]))
    for result in results:
        neighborhoodAvg[result]


# In[144]:


columns


# In[64]:


t_indep / t_indep.max()


# In[60]:


t_dep


# In[105]:


os.listdir('../data/columns').index('Shanghai--Shanghai--China.json')


# In[45]:




#[city for city in cities2 ]

# cities2 
# os.path.exists(f'data/airbnb/apt/{cities2[1]}.json')
import os
os.listdir('../data')
fn_cache = {}
def cacheThisFunction(func):
    def _(*args):
        print(func.__name__)
        return func(*args)
        _args = json.dumps(args)
        #','.join(list(args))
        key = hash(func.__name__ + _args)
        if key in fn_cache: return fn_cache[key]
        val = func(*args)
        fn_cache[key] = val
        json.dump(fn_cache, open('../data/cache/fn_cache.json', 'w+'))
        return val
    return _


# In[46]:


def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_whee():
    print("Whee!")
    
    
say_whee()


# In[51]:


import json
import requests
from shapely.geometry import shape, Point


        
def isochroneAssertion(geojson_data, point_to_check):
    longitude = point_to_check[0]
    latitude = point_to_check[1]
    point_to_check = Point(longitude, latitude)
    for feature in geojson_data['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point_to_check): return True
    else : return False

@cacheThisFunction
def getApt_by_travel_time(location, coords):
    apt = json.load(open(f'../data/airbnb/apt/{location}.json'))
    coords = locations[location]
    contours_minutes = 15
    lng = coords['longitude']
    lat = coords['latitude']
    isochrone_url = f'https://api.mapbox.com/isochrone/v1/mapbox/walking/{lng}%2C{lat}?contours_minutes={contours_minutes}&polygons=true&denoise=0&generalize=0&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg'
    geojson_data = requests.get(isochrone_url).json()
    apt2 = [_ for _ in apt if isochroneAssertion(geojson_data, apt[_])]
    print(location, len(apt), len(apt2))
    return {'data': apt2, 'isocrhrone': 'isochrone'}
    #print(len(apt), location)
    #apt2 = [_ for _ in apt if isochrone(apt[_])]
    #print(len(apt), len(apt2))

for location in locations:
    getApt_by_travel_time(location , locations[location])


# In[48]:





# In[5]:


locationNames = [
    "Amsterdam, North Holland, The Netherlands",
    "Antwerp, Flemish Region, Belgium",
    "Asheville, North Carolina, United States",
    "Athens, Attica, Greece",
    "Austin, Texas, United States",
    "Bangkok, Central Thailand, Thailand",
    "Barcelona, Catalonia, Spain",
    "Barossa Valley, South Australia, Australia",
    "Barwon South West, Vic, Victoria, Australia",
    "Beijing, Beijing, China",
    "Belize, Belize, Belize",
    "Bergamo, Lombardia, Italy",
    "Berlin, Berlin, Germany",
    "Bologna, Emilia-Romagna, Italy",
    "Bordeaux, Nouvelle-Aquitaine, France",
    "Boston, Massachusetts, United States",
    "Bozeman, Montana, United States",
    "Bristol, England, United Kingdom",
    "Broward County, Florida, United States",
    "Brussels, Brussels, Belgium",
    "Buenos Aires, Ciudad Autónoma de Buenos Aires, Argentina",
    "Cambridge, Massachusetts, United States",
    "Cape Town, Western Cape, South Africa",
    "Chicago, Illinois, United States",
    "Clark County, NV, Nevada, United States",
    "Columbus, Ohio, United States",
    "Copenhagen, Hovedstaden, Denmark",
    "Crete, Crete, Greece",
    "Dallas, Texas, United States",
    "Denver, Colorado, United States",
    "Dublin, Leinster, Ireland",
    "Edinburgh, Scotland, United Kingdom",
    "Euskadi, Euskadi, Spain",
    "Florence, Toscana, Italy",
    "Fort Worth, Texas, United States",
    "Geneva, Geneva, Switzerland",
    "Ghent, Flemish Region, Belgium",
    "Girona, Catalonia, Spain",
    "Greater Manchester, England, United Kingdom",
    "Hawaii, Hawaii, United States",
    "Hong Kong, Hong Kong, China",
    "Istanbul, Marmara, Turkey",
    "Jersey City, New Jersey, United States",
    "Lisbon, Lisbon, Portugal",
    "London, England, United Kingdom",
    "Los Angeles, California, United States",
    "Lyon, Auvergne-Rhone-Alpes, France",
    "Madrid, Comunidad de Madrid, Spain",
    "Malaga, Andalucía, Spain",
    "Mallorca, Islas Baleares, Spain",
    "Melbourne, Victoria, Australia",
    "Menorca, Islas Baleares, Spain",
    "Mexico City, Distrito Federal, Mexico",
    "Mid North Coast, New South Wales, Australia",
    "Milan, Lombardy, Italy",
    "Montreal, Quebec, Canada",
    "Mornington Peninsula, Victoria, Australia",
    "Munich, Bavaria, Germany",
    "Naples, Campania, Italy",
    "Nashville, Tennessee, United States",
    "New Brunswick, New Brunswick, Canada",
    "New Orleans, Louisiana, United States",
    "New York City, New York, United States",
    "Newark, New Jersey, United States",
    "Northern Rivers, New South Wales, Australia",
    "Oakland, California, United States",
    "Oslo, Oslo, Norway",
    "Pacific Grove, California, United States",
    "Paris, Île-de-France, France",
    "Pays Basque, Pyrénées-Atlantiques, France",
    "Portland, Oregon, United States",
    "Porto, Norte, Portugal",
    "Prague, Prague, Czech Republic",
    "Puglia, Puglia, Italy",
    "Quebec City, Quebec, Canada",
    "Rhode Island, Rhode Island, United States",
    "Riga, Riga, Latvia",
    "Rio de Janeiro, Rio de Janeiro, Brazil",
    "Rome, Lazio, Italy",
    "Rotterdam, South Holland, The Netherlands",
    "Salem, OR, Oregon, United States",
    "San Diego, California, United States",
    "San Francisco, California, United States",
    "San Mateo County, California, United States",
    "Santa Clara County, California, United States",
    "Santa Cruz County, California, United States",
    "Santiago, Región Metropolitana de Santiago, Chile",
    "Seattle, Washington, United States",
    "Sevilla, Andalucía, Spain",
    "Shanghai, Shanghai, China",
    "Sicily, Sicilia, Italy",
    "Singapore, Singapore, Singapore",
    "South Aegean, South Aegean, Greece",
    "Stockholm, Stockholms län, Sweden",
    "Sydney, New South Wales, Australia",
    "Taipei, Northern Taiwan, Taiwan",
    "Tasmania, Tasmania, Australia",
    "The Hague, South Holland, The Netherlands",
    "Thessaloniki, Central Macedonia, Greece",
    "Tokyo, Kantō, Japan",
    "Toronto, Ontario, Canada",
    #"Trentino, Trentino-Alto Adige/Südtirol, Italy",
    "Twin Cities MSA, Minnesota, United States",
    "Valencia, Valencia, Spain",
    "Vancouver, British Columbia, Canada",
    "Vaud, Vaud, Switzerland",
    "Venice, Veneto, Italy",
    "Victoria, British Columbia, Canada",
    "Vienna, Vienna, Austria",
    "Washington, D.C., District of Columbia, United States",
    "Western Australia, Western Australia, Australia",
    "Winnipeg, Manitoba, Canada",
    "Zurich, Zürich, Switzerland",
    "Ireland",
    "Malta",
    "New Zealand"
]


# In[6]:


shit = [
    "http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2023-09-03/visualisations/listings.csv",
    "http://data.insideairbnb.com/belgium/vlg/antwerp/2023-09-23/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/nc/asheville/2023-09-13/visualisations/listings.csv",
    "http://data.insideairbnb.com/greece/attica/athens/2023-09-21/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/tx/austin/2023-09-10/visualisations/listings.csv",
    "http://data.insideairbnb.com/thailand/central-thailand/bangkok/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/spain/catalonia/barcelona/2023-09-06/visualisations/listings.csv",
    "http://data.insideairbnb.com/australia/sa/barossa-valley/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/australia/vic/barwon-south-west-vic/2023-09-23/visualisations/listings.csv",
    "http://data.insideairbnb.com/china/beijing/beijing/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/belize/bz/belize/2023-09-23/visualisations/listings.csv",
    "http://data.insideairbnb.com/italy/lombardia/bergamo/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/germany/be/berlin/2023-09-16/visualisations/listings.csv",
    "http://data.insideairbnb.com/italy/emilia-romagna/bologna/2023-09-13/visualisations/listings.csv",
    "http://data.insideairbnb.com/france/nouvelle-aquitaine/bordeaux/2023-09-10/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ma/boston/2023-09-16/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/mt/bozeman/2023-09-04/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-kingdom/england/bristol/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/fl/broward-county/2023-09-21/visualisations/listings.csv",
    "http://data.insideairbnb.com/belgium/bru/brussels/2023-09-17/visualisations/listings.csv",
    "http://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ma/cambridge/2023-09-23/visualisations/listings.csv",
    "http://data.insideairbnb.com/south-africa/wc/cape-town/2023-09-23/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/il/chicago/2023-09-12/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/nv/clark-county-nv/2023-09-16/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/oh/columbus/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/denmark/hovedstaden/copenhagen/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/greece/crete/crete/2023-09-23/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/tx/dallas/2023-09-12/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/co/denver/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/ireland/leinster/dublin/2023-09-07/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-kingdom/scotland/edinburgh/2023-09-11/visualisations/listings.csv",
    "http://data.insideairbnb.com/spain/pv/euskadi/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/italy/toscana/florence/2023-09-13/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/tx/fort-worth/2023-09-07/visualisations/listings.csv",
    "http://data.insideairbnb.com/switzerland/geneva/geneva/2023-09-23/visualisations/listings.csv",
    "http://data.insideairbnb.com/belgium/vlg/ghent/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/spain/catalonia/girona/2023-09-25/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-kingdom/england/greater-manchester/2023-09-21/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/hi/hawaii/2023-09-10/visualisations/listings.csv",
    "http://data.insideairbnb.com/china/hk/hong-kong/2023-09-17/visualisations/listings.csv",
    "http://data.insideairbnb.com/turkey/marmara/istanbul/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/nj/jersey-city/2023-09-18/visualisations/listings.csv",
    "http://data.insideairbnb.com/portugal/lisbon/lisbon/2023-09-11/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-kingdom/england/london/2023-09-06/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ca/los-angeles/2023-09-03/visualisations/listings.csv",
    "http://data.insideairbnb.com/france/auvergne-rhone-alpes/lyon/2023-09-10/visualisations/listings.csv",
    "http://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2023-09-07/visualisations/listings.csv",
    "http://data.insideairbnb.com/spain/andaluc%C3%ADa/malaga/2023-09-25/visualisations/listings.csv",
    "http://data.insideairbnb.com/spain/islas-baleares/mallorca/2023-09-11/visualisations/listings.csv",
    "http://data.insideairbnb.com/australia/vic/melbourne/2023-09-04/visualisations/listings.csv",
    "http://data.insideairbnb.com/spain/islas-baleares/menorca/2023-09-25/visualisations/listings.csv",
    "http://data.insideairbnb.com/mexico/df/mexico-city/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/australia/nsw/mid-north-coast/2023-09-06/visualisations/listings.csv",
    "http://data.insideairbnb.com/italy/lombardy/milan/2023-09-13/visualisations/listings.csv",
    "http://data.insideairbnb.com/canada/qc/montreal/2023-09-02/visualisations/listings.csv",
    "http://data.insideairbnb.com/australia/vic/mornington-peninsula/2023-09-11/visualisations/listings.csv",
    "http://data.insideairbnb.com/germany/bv/munich/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/italy/campania/naples/2023-09-13/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/tn/nashville/2023-09-16/visualisations/listings.csv",
    "http://data.insideairbnb.com/canada/nb/new-brunswick/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/la/new-orleans/2023-09-03/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ny/new-york-city/2023-10-01/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/nj/newark/2023-09-25/visualisations/listings.csv",
    "http://data.insideairbnb.com/australia/nsw/northern-rivers/2023-09-12/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ca/oakland/2023-09-18/visualisations/listings.csv",
    "http://data.insideairbnb.com/norway/oslo/oslo/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ca/pacific-grove/2023-09-25/visualisations/listings.csv",
    "http://data.insideairbnb.com/france/ile-de-france/paris/2023-09-04/visualisations/listings.csv",
    "http://data.insideairbnb.com/france/pyr%C3%A9n%C3%A9es-atlantiques/pays-basque/2023-09-12/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/or/portland/2023-09-17/visualisations/listings.csv",
    "http://data.insideairbnb.com/portugal/norte/porto/2023-09-11/visualisations/listings.csv",
    "http://data.insideairbnb.com/czech-republic/prague/prague/2023-09-17/visualisations/listings.csv",
    "http://data.insideairbnb.com/italy/puglia/puglia/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/canada/qc/quebec-city/2023-09-02/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ri/rhode-island/2023-09-25/visualisations/listings.csv",
    "http://data.insideairbnb.com/latvia/riga/riga/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/brazil/rj/rio-de-janeiro/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/italy/lazio/rome/2023-09-07/visualisations/listings.csv",
    "http://data.insideairbnb.com/the-netherlands/south-holland/rotterdam/2023-09-18/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/or/salem-or/2023-09-18/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ca/san-diego/2023-09-18/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ca/san-francisco/2023-09-02/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ca/san-mateo-county/2023-09-18/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ca/santa-clara-county/2023-09-18/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/ca/santa-cruz-county/2023-09-25/visualisations/listings.csv",
    "http://data.insideairbnb.com/chile/rm/santiago/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/wa/seattle/2023-09-18/visualisations/listings.csv",
    "http://data.insideairbnb.com/spain/andaluc%C3%ADa/sevilla/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/china/shanghai/shanghai/2023-09-22/visualisations/listings.csv",
    "http://data.insideairbnb.com/italy/sicilia/sicily/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/singapore/sg/singapore/2023-09-23/visualisations/listings.csv",
    "http://data.insideairbnb.com/greece/south-aegean/south-aegean/2023-09-16/visualisations/listings.csv",
    "http://data.insideairbnb.com/sweden/stockholms-l%C3%A4n/stockholm/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/australia/nsw/sydney/2023-09-04/visualisations/listings.csv",
    "http://data.insideairbnb.com/taiwan/northern-taiwan/taipei/2023-09-25/visualisations/listings.csv",
    "http://data.insideairbnb.com/australia/tas/tasmania/2023-09-03/visualisations/listings.csv",
    "http://data.insideairbnb.com/the-netherlands/south-holland/the-hague/2023-09-18/visualisations/listings.csv",
    "http://data.insideairbnb.com/greece/central-macedonia/thessaloniki/2023-09-21/visualisations/listings.csv",
    "http://data.insideairbnb.com/japan/kant%C5%8D/tokyo/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/canada/on/toronto/2023-09-03/visualisations/listings.csv",
    #"http://data.insideairbnb.com/italy/trentino-alto-adige-s%C3%BCdtirol/trentino/2023-09-25/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/mn/twin-cities-msa/2023-09-17/visualisations/listings.csv",
    "http://data.insideairbnb.com/spain/vc/valencia/2023-09-16/visualisations/listings.csv",
    "http://data.insideairbnb.com/canada/bc/vancouver/2023-09-06/visualisations/listings.csv",
    "http://data.insideairbnb.com/switzerland/vd/vaud/2023-10-01/visualisations/listings.csv",
    "http://data.insideairbnb.com/italy/veneto/venice/2023-09-03/visualisations/listings.csv",
    "http://data.insideairbnb.com/canada/bc/victoria/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/austria/vienna/vienna/2023-09-07/visualisations/listings.csv",
    "http://data.insideairbnb.com/united-states/dc/washington-dc/2023-09-13/visualisations/listings.csv",
    "http://data.insideairbnb.com/australia/wa/western-australia/2023-09-21/visualisations/listings.csv",
    "http://data.insideairbnb.com/canada/mb/winnipeg/2023-09-11/visualisations/listings.csv",
    "http://data.insideairbnb.com/switzerland/z%C3%BCrich/zurich/2023-09-24/visualisations/listings.csv",
    "http://data.insideairbnb.com/ireland/2023-09-06/visualisations/listings.csv",
    "http://data.insideairbnb.com/malta/2023-09-17/visualisations/listings.csv",
    "http://data.insideairbnb.com/new-zealand/2023-09-02/visualisations/listings.csv"
]


for i, url in enumerate(shit):
    res = requests.get(url)
    location = locationNames[i]
    print(location, url)
    with open('../data/csv/' + location +'.csv', 'w+') as f: f.write(res.text)


# In[32]:


dev = 'cuda:0'
def test_prediction(test_predictions):
    ctrl = test_predictions.sum(1).tolist()[0]
    isFalse = len([sum(row) for idx, row in enumerate(test_predictions.tolist()) if sum(row) <= ctrl and t_dep[idx] == 0])
    isTrue = len([sum(row) for idx, row in enumerate(test_predictions.tolist()) if sum(row) > ctrl and t_dep[idx] == 1])
    allFalse = len([sum(row) for idx, row in enumerate(test_predictions.tolist()) if t_dep[idx] == 0])
    allTrue = len([sum(row) for idx, row in enumerate(test_predictions.tolist()) if t_dep[idx] == 1])
    return (isFalse / allFalse, isTrue / allTrue, isFalse, isTrue)
def plot_loss(l):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 4))
    legends = []
    plt.plot(l) 
    plt.plot([0, len([i for k,i in enumerate(rowGeneExpression.values()) if dependent_variables[k]])], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
    plt.legend(legends);

#numerical_values = df.select_dtypes(include=[int, float]).values.tolist()
#t_indep = torch.Tensor(numerical_values).to(dev)
# t_indep = t_indep / vals
# λλλλλ.requires_grad_(True)
#3 variations, test, t_indep and t_indep+embedding
resultant_tensor = t_indep
encodedOutput.requires_grad_(True)
#resultant_tensor = torch.cat((t_indep.to(dev),tensor.to(dev)), 1)
#resultant_tensor = λλλλλ
#resultant_tensor = tensor
vals, indices = resultant_tensor.max(dim=0)
#resultant_tensor = resultant_tensor / vals
resultant_tensor = resultant_tensor.to(dev)
test_indep = torch.tensor([[t_dep[k].item() for i in enumerate(range(resultant_tensor.shape[1]))] for k, i in enumerate(range(resultant_tensor.shape[0]))])
dim = resultant_tensor.shape[1]

model = torch.nn.Sequential(
    torch.nn.Linear(dim,dim),
    nn.ReLU(),
    nn.Linear(dim,dim),
    nn.ReLU(),
    nn.Linear(dim, dim),
    nn.Sigmoid()
).to(dev)

optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=.1, 
    weight_decay=0.01
)
# model = model.to(device)
# input_tensor = input_tensor.to(device)

n_iterations = 1000
loss_track = []
accuracy_track = []
no_entropy = []
loss_function = torch.nn.BCELoss()
def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss):
    """ dynamically print the loss plot during the training/validation loop.
        expects epoch to start from 1.
    """
    x = range(1, epoch+1)
    y = np.concatenate((train_loss, valid_loss))
    graphs = [[x,train_loss], [x,valid_loss]]
    x_margin = 0.2
    y_margin = 0.05
    x_bounds = [1-x_margin, epochs+x_margin]
    y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]

    mb.update_graph(graphs, x_bounds, y_bounds)
mb = master_bar(range(1))
def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss):
    """ dynamically print the loss plot during the training/validation loop.
        expects epoch to start from 1.
    """
    x = range(1, epoch+1)
    y = np.concatenate((train_loss, valid_loss))
    print(x,y)
    graphs = [[x,train_loss], [x,valid_loss]]
    x_margin = 0.2
    y_margin = 0.05
    x_bounds = [1-x_margin, epochs+x_margin]
    y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]
    print(x_bounds, y_bounds)
    mb.update_graph(graphs, x_bounds, y_bounds)

#diff expected price
#1 million airbnbs -> give each a 
#adapt NN to 30% of airbnbs -> 600,000 pricing estimates 
#plot this on a notebook
#in these cities you can find good deals -> travelers + immigrants
#in these cities you can buy a home and make an airbnb and it will make lots of money -> SMBs
def pricing_prediction():
    for i in mb:    
    #for j in progress_bar(range(2000), parent=mb):
        for j in progress_bar(range(2000)):
            loss = loss_function(model(resultant_tensor).sum(1).sigmoid(), t_dep.to(dev))
            optimizer.zero_grad()  # 3
            loss.backward(retain_graph=True)  # 4
            optimizer.step()  # 5
            if j == 1 or j % 50 == 0:
                test_predictions = model(resultant_tensor)
                #print(loss.item(), test_predictions.sum().item() / 8)
                print(test_prediction(test_predictions))
            loss_track.append(loss.item())
            accuracy_track.append(test_predictions.sum().item() / 8)
            #no_entropy += [test_predictions.sum().item() / 8]
            #         k = 100 * i + j
            #         x = np.arange(0, 2*k*np.pi/1000, 0.01)
            #         y1, y2 = np.cos(x), np.sin(x)
            #         graphs = [[x,y1], [x,y2]]
            #         x_bounds = [0, 2*np.pi]
            #         y_bounds = [-1,1]
            #         mb.update_graph(graphs, x_bounds, y_bounds)
            #         print(loss_track, accuracy_track)
            #print(loss_track, accuracy_track)
            #plot_loss_update(j, n_iterations, mb, loss_track, accuracy_track)
            #for batch in progress_bar(range(2), parent=mb): sleep(0.2)
#get avg price of neighborhood
#try to get coefficents of price vs factors -> 5 room = 5k, 4 room = 4k


# In[3]:


get_ipython().system('mkdir ../data/csv')


# In[2]:


import requests
print(12312)


# In[200]:


# import csv
# import json
def denormalize(file_name):
    print(file_name)
    with open('data/shit/' + file_name, mode='r') as file:
        csv_reader = csv.DictReader(file)
        csv_reader = [row for row in csv_reader]
        #listings = [row['id'] for row in csv_reader] 
        listings = {row['id']:[row['longitude'], row['latitude']] for row in csv_reader}
        result = re.sub(r' |,', '-', file_name)
        print(len(listings), len(geoCoordinates))
        json.dump(listings, open('data/airbnb/apt/' + result.replace('.csv', '.json'), 'w+') , indent=4)
#         for i, listingID in enumerate(listings):
#                 json.dump(geoCoordinates[i], open(f'data/airbnb/geocoordinates/{listingID}.json', 'w+'))

#denormalize(os.listdir('data/shit')[0])
for file_name in os.listdir('data/shit'): denormalize(file_name)


# In[142]:


import random
def makeUrl():
    _ = (.1 * random.random())
    ne_lat = 35.82484552438841 + _
    ne_lng =139.61669328586441 + _
    sw_lng = 139.36326113397735 + _
    sw_lat = 35.45779629356468 + _
    url = ['https://www.airbnb.com/s/Tokyo--Japan/homes?tab_id=home_tab&',
    'refinement_paths%5B%5D=%2Fhomes&flexible_trip_lengths%5B%5D=one_wee',
    'k&monthly_start_date=2023-11-01&monthly_length=3&price_filter_input_type=0',
    '&price_filter_num_nights=5&channel=EXPLORE&query=Tokyo%2C%20Japan&',
    'date_picker_type=calendar&',
    'source=structured_search_input_header&search_type=user_map_move',
    f'&ne_lat={ne_lat}&ne_lng={ne_lng}&sw_lat={sw_lat}',
    f'&sw_lng={sw_lng}&zoom=11.227277836691444&zoom_level=11.227277836691444&search_by_map=true']
    
    return ''.join(url)

import re
def getUrl(s):
    match = re.search(r"rooms/(\d+)", s)
    if match:
        room_id = match.group(1)
        return room_id
    return ''

def get_all_rooms(links):
    s = [getUrl(string) for string in _]
    return list(set([_ for _ in s if len(_) > 0]))

def process_page(bs):
    all_links = bs.find_all('a')
    _ = [string.attrs.get('href') for string in all_links]
    return get_all_rooms(_)

r = requests.get(makeUrl())
bs = bs4.BeautifulSoup(r.text)
print(process_page(bs))

r = requests.get(makeUrl())
bs = bs4.BeautifulSoup(r.text)
print(process_page(bs))


# In[66]:


s = [_.match(r'/rooms/([0-9]*)') for _ in s]
s


# In[34]:


import requests




r.text

import openai
#openai.organization = "org-Yz814AiRJVl9JGvXXiL9ZXPl"
openai.api_key = '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'


# In[33]:



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


# In[11]:


import bs4

bs = bs4.BeautifulSoup(r.text)


bs.find_all('a')


# In[76]:


all_houses = [json.load(open(_)) for _ in glob.glob('osm_residential/*.json')]
all_houses


# In[74]:


all_houses = [json.load(open(_)) for _ in glob.glob('osm_homes/*.json')]
all_houses


# In[1]:


import json
from lxml import etree

def extract_houses_from_osm(osm_file):
    houses = []

    # Parse the OSM file using lxml
    if len(open(osm_file).read()) == 0: return []
    tree = etree.parse(osm_file)
    root = tree.getroot()
    #print(tree)
    # Iterate over every node and way in the OSM XML
    for element in root:
        #print(element.tag)
        if element.tag == 'node' or element.tag == 'way':
            for tag in element.findall('tag'):
                k = tag.get('k')
                # Assuming houses are denoted with building=residential in OSM data
                if k == 'building':
                    houses.append(element)

    print(len(houses))
    return houses

def convert_to_json(elements):
    json_data = []
    
    for element in elements:
        json_element = {
            'id': element.get('id'),
            'type': element.tag,
            'tags': {}
        }
        
        for tag in element.findall('tag'):
            json_element['tags'][tag.get('k')] = tag.get('v')
        
        if element.tag == 'node':
            json_element['lat'] = element.get('lat')
            json_element['lon'] = element.get('lon')
        
        json_data.append(json_element)
    
    return json_data
import glob
for osm_file in glob.glob('osm_way_residential/*.osm'):
    if 'pbf' in osm_file: continue
    if '-way-house' in osm_file: continue
    
    houses = extract_houses_from_osm(osm_file)
    houses_json = convert_to_json(houses)
    with open(osm_file.replace('.osm','.json'), 'w') as outfile:
        print('writing osm file ' + osm_file)
        json.dump(houses_json, outfile, indent=2)

print("Houses extracted and saved to houses.json")


# In[6]:


import os
os.listdir('airbnb/apt')


# In[8]:


city_location = {"New York City, USA":" 40.7128, -74.0060","San Francisco, USA":" 37.7749, -122.4194","Vancouver, Canada":" 49.2827, -123.1207","New Orleans, USA":" 29.9511, -90.0715","Los Angeles, USA":" 34.0522, -118.2437","Chicago, USA":" 41.8781, -87.6298","Toronto, Canada":" 43.6532, -79.3832","Mexico City, Mexico":" 19.4326, -99.1332","Montreal, Canada":" 45.5017, -73.5673","Boston, USA":" 42.3601, -71.0589","Miami, USA":" 25.7617, -80.1918","Austin, USA":" 30.2672, -97.7431","Quebec City, Canada":" 46.8139, -71.2082","Seattle, USA":" 47.6062, -122.3321","Nashville, USA":" 36.1627, -86.7816","Tokyo, Japan":" 35.6895, 139.6917","Kyoto, Japan":" 35.0116, 135.7681","Bangkok, Thailand":" 13.7563, 100.5018","Hong Kong, China":" 22.3193, 114.1694","Singapore":" 1.3521, 103.8198","Seoul, South Korea":" 37.5665, 126.9780","Beijing, China":" 39.9042, 116.4074","Dubai, UAE":" 25.276987, 55.296249","Taipei, Taiwan":" 25.0330, 121.5654","Istanbul, Turkey":" 41.0082, 28.9784","Hanoi, Vietnam":" 21.0285, 105.8544","Mumbai, India":" 19.0760, 72.8777","Kuala Lumpur, Malaysia":" 3.1390, 101.6869","Jaipur, India":" 26.9124, 75.7873","Rio de Janeiro, Brazil":" -22.9068, -43.1729","Buenos Aires, Argentina":" -34.6037, -58.3816","Cartagena, Colombia":" 10.3910, -75.4794","Lima, Peru":" -12.0464, -77.0428","Santiago, Chile":" -33.4489, -70.6693","Cusco, Peru":" -13.5319, -71.9675","Medellín, Colombia":" 6.2476, -75.5709","Quito, Ecuador":" -0.1807, -78.4678","Montevideo, Uruguay":" -34.9011, -56.1911","Bogota, Colombia":" 4.7100, -74.0721","Cape Town, South Africa":" -33.9249, 18.4241","Marrakech, Morocco":" 31.6295, -7.9811","Cairo, Egypt":" 30.8025, 31.2357","Dakar, Senegal":" 14.6928, -17.4467","Zanzibar City, Tanzania":" -6.1659, 39.2026","Accra, Ghana":" 5.6037, -0.1869","Addis Ababa, Ethiopia":" 9.0300, 38.7400","Victoria Falls, Zimbabwe-Zambia":" -17.9243, 25.8572","Nairobi, Kenya":" -1.286389, 36.817223","Tunis, Tunisia":" 36.8065, 10.1815","Sydney, Australia":" -33.8688, 151.2093","Melbourne, Australia":" -37.8136, 144.9631","Auckland, New Zealand":" -36.8485, 174.7633","Wellington, New Zealand":" -41.2865, 174.7762","Brisbane, Australia":" -27.4698, 153.0251","Honolulu, Hawaii, USA":" 21.3069, -157.8583","Bali, Indonesia":" -8.3405, 115.0920","Santorini, Greece":" 36.3932, 25.4615","Maldives (Male)":" 4.1755, 73.5093","Phuket, Thailand":" 7.8804, 98.3923","Ibiza, Spain":" 38.9067, 1.4206","Seychelles (Victoria)":" -4.6191, 55.4513","Havana, Cuba":" 23.1136, -82.3666","Punta Cana, Dominican Republic":" 18.5820, -68.4055","Dubrovnik, Croatia":" 42.6507, 18.0944","Ljubljana, Slovenia":" 46.0569, 14.5058","Tallinn, Estonia":" 59.4370, 24.7536","Riga, Latvia":" 56.9496, 24.1052","Sarajevo, Bosnia and Herzegovina":" 43.8563, 18.4131","Vilnius, Lithuania":" 54.6872, 25.2797","Tbilisi, Georgia":" 41.7151, 44.8271","Yerevan, Armenia":" 40.1792, 44.4991","Baku, Azerbaijan":" 40.4093, 49.8671","Belgrade, Serbia":" 44.7866, 20.4489","Skopje, North Macedonia":" 41.9973, 21.4280","Banff, Canada":" 51.1784, -115.5708","Queenstown, New Zealand":" -45.0312, 168.6626","Reykjavik (as a gateway to Icelandic nature)":" 64.1466, -21.9426","Ushuaia, Argentina (Gateway to Antarctica)":" -54.8019, -68.3030","Kathmandu, Nepal (Gateway to the Himalayas)":" 27.7172, 85.3240"}


# In[25]:


#2 cities
#moscow 
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


# In[11]:


# import os

# os.listdir('osm_homes/')
# glob.glob(f'osm_homes/*_houses.json')


# In[12]:


# import os 
# houses = json.load(open(f'osm_homes/Melbourne--Australia_houses.json'))
# houses


# In[82]:


for continent in getCityList(): 
    print(f'osmium tags-filter osm_pbf/{continent}-latest.osm.pbf w/building=residential -o {continent}-way-house.osm &')


# In[48]:


for continent in getCityList():
    print(f'osmium tags-filter osm_pbf/{continent}.osm.pbf n/building=residential -o osm_pbf/{continent}_residential.osm')


# In[86]:


import subprocess

requests = []
def fn(parameters):
    #print(' '.join(parameters))
    print(parameters)
    result = subprocess.run(parameters, capture_output=True, text=True)
    #print("Return code:", result.returncode)
    #print("stdout:\n{}".format(result.stdout))
    #print("stdout:\n{}".format(result.stderr))
    
    


    
def makeRequests(continent, city):
    if city not in city_location: return #print(city + 'not found')
    location = city_location[city].split(',')
    location = [float(location[0].strip()), float(location[1].strip())]
    location.reverse()
    bbox = [location[0] -.5, location[1] -.5, location[0] + .5, location[1] + .5]
    city_name = city.replace(',','-').replace(r' ', '-')
    #request = f'osmium extract --bbox {bbox} australia-oceania-latest.osm.pbf -o {city}.osm'
    #request = f'osmium extract --bbox {bbox} data/planet-latest.osm -o - | osmium tags-filter - n/building=house -o data/{city_name}.osm.pbf'
    #request = 'osmium extract --bbox 139.5,35.5,140.1,36.0 planet-latest.osm   -o _.osm'
    request = [
        'osmium', 'extract',
        '--overwrite',
        '--bbox', ','.join(map(str, bbox)),
        f'osm_way_residential/{continent}-way-house.osm'
#        f'osm_pbf/-houses.osm.pbf',
        ' -o', f' osm_way_residential/{city_name}.osm'
    ]
    requests.append(' '.join(request))

    #print(f"'{' '.join(request)}'")
    #print(' ')
    #print(' ')

    #fn(request)
    #requests.append(request)

for continent in getCityList():
    for city in getCityList()[continent]:
        makeRequests(continent, city)
requests.reverse()
# for request in requests:
#     fn(request)
# import concurrent
# with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#     for p in executor.map(fn, requests):
#         print('_')
          
          
# import subprocess
# from concurrent.futures import ThreadPoolExecutor

# def run_cmd(cmd):
#     print(f"Starting: {cmd}")
#     process = subprocess.Popen(cmd, shell=True)
#     process.communicate()
#     if process.returncode == 0:
#         print(f"Done: {cmd}")
#     else:
#         print(f"Error running: {cmd}")
#     return cmd
# with ThreadPoolExecutor(max_workers=5) as executor:
#     results = list(executor.map(run_cmd, requests))
print("All commands executed!")
buffer = ''
for i in requests: print(i + ' &')


# In[32]:


requests


# In[ ]:


# import osmium as osm
import sys

class HouseHandler(osm.SimpleHandler):
    def __init__(self):
        super(HouseHandler, self).__init__()
        self.houses = []

    def area(self, a):
        # This method is triggered whenever an area (like a building) is found in the OSM data
        if a.tags.get('building') == 'house':
            self.houses.append(a)

    def output_houses(self):
        for house in self.houses:
            print(house.id, house.tags.get('name', 'Unnamed'))

# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("Usage: python extract_houses.py <OSM FILE>")
#         sys.exit(1)

osm_file = 'data/planet-latest.osm'

handler = HouseHandler()
handler.apply_file(osm_file, locations=True)

handler.output_houses()


# In[10]:


geoCoordCache
from collections import defaultdict
import concurrent.futures
import time 
import h3
from compiled_functions import getAirbnbs
import re
import random
import json


# In[203]:


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


# In[204]:


import requests
import copy
geoCoordCache = {}


# In[205]:


poi_names = [
    "grit_bin",
    "parking_entrance",
    "pub",
    "cafe",
    "bar",
    "pub",
    "bar",
    "restaurant",
    "biergarten",
    "pub",
    "bar",
    "restaurant",
    "pub",
    "bar",
    "restaurant",
    "cafe",
    "restaurant",
    "bar",
    "fast_food",
    "fast_food",
    "restaurant",
    "food_court",
    "ice_cream",
    "pub",
    "bar",
    "pub",
    "restaurant",
    "fast_food",
    "college",
    "dancing_school",
    "driving_school",
    "kindergarten",
    "language_school",
    "library",
    "surf_school",
    "toy_library",
    "research_institute",
    "training",
    "music_school",
    "school",
    "traffic_park",
    "university",
    "bicycle_parking",
    "bicycle_repair_station",
    "bicycle_rental",
    "boat_rental",
    "boat_sharing",
    "bus_station",
    "station",
    "car_rental",
    "car_sharing",
    "car_wash",
    "compressed_air",
    "vehicle_inspection",
    "charging_station",
    "driver_training",
    "ferry_terminal",
    "fuel",
    "service",
    "grit_bin",
    "motorcycle_parking",
    "parking",
    "service",
    "parking_aisle",
    "parking_entrance",
    "site",
    "parking",
    "parking",
    "parking_space",
    "site",
    "parking",
    "parking",
    "taxi",
    "atm",
    "bank",
    "bureau_de_change",
    "baby_hatch",
    "clinic",
    "dentist",
    "doctors",
    "hospital",
    "nursing_home",
    "social_facility",
    "nursing_home",
    "pharmacy",
    "social_facility",
    "veterinary",
    "arts_centre",
    "brothel",
    "casino",
    "cinema",
    "community_centre",
    "conference_centre",
    "events_venue",
    "exhibition_centre",
    "fountain",
    "gambling",
    "bookmaker",
    "lottery",
    "casino",
    "adult_gaming_centre",
    "love_hotel",
    "music_venue",
    "nightclub",
    "amenity=stripclub",
    "planetarium",
    "public_bookcase",
    "social_centre",
    "stripclub",
    "studio",
    "swingerclub",
    "theatre",
    "cinema",
    "courthouse",
    "fire_station",
    "police",
    "post_box",
    "post_depot",
    "post_office",
    "prison",
    "ranger_station",
    "townhall",
    "bbq",
    "picnic_site",
    "firepit",
    "bench",
    "dog_toilet",
    "dressing_room",
    "drinking_water",
    "give_box",
    "mailroom",
    "parcel_locker",
    "shelter",
    "shower",
    "telephone",
    "toilets",
    "water_point",
    "watering_place",
    "sanitary_dump_station",
    "recycling",
    "container",
    "centre",
    "waste_basket",
    "waste_disposal",
    "waste_transfer_station",
    "animal_boarding",
    "animal_breeding",
    "animal_shelter",
    "animal_training",
    "baking_oven",
    "bakehouse",
    "clock",
    "crematorium",
    "dive_centre",
    "funeral_hall",
    "grave_yard",
    "cemetery",
    "hunting_stand",
    "internet_cafe",
    "kitchen",
    "kneipp_water_cure",
    "lounger",
    "marketplace",
    "monastery",
    "photo_booth",
    "place_of_mourning",
    "place_of_worship",
    "place_of_worship",
    "the article",
    "public_bath",
    "public_building",
    "government",
    "refugee_site",
    "vending_machine",
    "Taginfo",
    "embassy",
    "hospital",
    "hospital"
][10:12]


# In[218]:


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

def _housing(url):
    shit = get_lat_long(url)
    lat = shit[0]
    lng = shit[1]
    h3_cell_id = h3.geo_to_h3(lat, lng, resolution=7)
    _coefficents = {}
    ret = {
        'url': url,
        'location': shit,
        'h3_cell': h3_cell_id,  
        'coefficents': _coefficents
    }
    total = 0
    for key in coefficents: total += h3_cells[h3_cell_id][key]
    for key in coefficents: ret['coefficents'][key] = h3_cells[h3_cell_id][key] / total
    return ret 

def key_function(apt, preferences_poi, idx):
    dist = 0
    for key in apt['coefficents']:
        dist += apt['coefficents'][key] - preferences_poi[key][idx]
    return dist
    #make sure each house is in walking distance 30 minutes or train 30 minutes from each other
    #10 * 10 coefficents -> grid of checkboxes 

    
    
def make_fetch_shops_for_cell(poi_names, h3_cells):
    def fetch_shops_for_cell(hex_id):
        results = h3_cells[hex_id]
        ll = h3.h3_to_geo(hex_id)
        #     poi_names = ['restaurant',
        #                  'library',
        #                  'atm',
        #                  'vending_machine',
        #                  'bench',
        #                  'parking_space',
        #                  'bank',
        #                  'clinic',
        #                  'place_of_worship',
        #                  'research_institute']
        for key in poi_names:
            if key not in results or results[key] == 0:
                print(key, ll[1])
                val = len(fetch_coffee_shops(ll[1], ll[0], key))
                if val == 0: val = -1
                results[key] = val
        return (hex_id, results)
    return fetch_shops_for_cell

def aggregate_poi_in_h3_cell(h3_cells, fn):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for hex_id, results in executor.map(fn, h3_cells.keys()):
            pass
            #h3_cells[hex_id] = results
    return h3_cells

def attempt_at_building_communities(_, documentContext, sentence):
    all_houses = json.load(open('data/airbnb/apt/Tokyo--Japan.json'))
    geo_coords = [get_lat_long(url) for url in all_houses] #tODO flip geo coordinates or re-parse 
    people_names = 'fred bob sally panda velma gis mercator machester xml parsing'.split(' ')
    people_housing_list = {}

    totals = defaultdict(int)
#     poi_names = ['restaurant',
#                  'library',
#                  'atm',
#                  'vending_machine',
#                  'bench',
#                  'parking_space',
#                  'bank',
#                  'clinic',
#                  'place_of_worship',
#                  'bar',
#                  'research_institute'] #todo get from document context or sentence
    
    #this would be a good function
    h3_cells = retrieveAggregation(poi_names) #{}
    for location in geo_coords: 
        hex_id = h3.geo_to_h3(location[0], location[1], 7)
        if hex_id not in h3_cells: 
            h3_cells[hex_id] = {}
            for col in poi_names: 
                if col not in h3_cells[hex_id]:
                    h3_cells[hex_id][col] = 0
     ##for each column read all h3 cells
    
    aggregate_poi_in_h3_cell(h3_cells, make_fetch_shops_for_cell(poi_names, h3_cells))
    storeAggregation(h3_cells, poi_names)
    
    for hex_id in h3_cells:
        for key in coefficents:
            if key in h3_cells[hex_id]:
                totals[hex_id+key] += h3_cells[hex_id][key]
            
    h3_cell_counts = copy.deepcopy(h3_cells)

    for hex_id in h3_cells:
        for key in coefficents:
            h3_cells[hex_id][key] / max(1, totals[hex_id+key])
    
    _houses = [_housing(url) for url in all_houses]
    for idx, person in enumerate(people_names):
        people_housing_list[person] = sorted(_houses, key=lambda apt: key_function(apt, preferences, idx))
    reports = []
    for idx, person in enumerate(people_names):
        reasoning_explanation = ''
        house = people_housing_list[person][0]
        most = sorted(house['coefficents'].items(), key=lambda _: _[1])[-1]
        #max_key = max(house['coefficents'].values(), key=house['coefficents'].values().get)
        #print(most)
        max_key = most
        hex_number = house['h3_cell']
        num_studios = h3_cell_counts[hex_number][max_key[0]]
        reasoning_explanation += f'you selected {max_key[0]} as the most important and there are {num_studios} studios in that region :)'
        report = {
            'name': person,
            'house_suggestion':people_housing_list[person][0]['url'] ,
            'reasoning_explanation': reasoning_explanation,
        }
        reports.append(report)
    
    return reports
first = time.time()
_ = attempt_at_building_communities('', {}, '')
print(_)
time.time( )- first


# In[221]:


dir(h3)


# In[189]:




example = {'restaurant': 18,
  'library': 2,
  'atm': 0,
  'vending_machine': 1,
  'bench': 35,
  'parking_space': 0,
  'bank': 5,
  'clinic': 1,
  'place_of_worship': 16,
  'research_institute': 0}



# In[161]:


storeAggregation(h3_cells, example.keys())


# In[158]:


hash(json.dumps(retrieveAggregation(example.keys())))


# In[208]:


retrieveAggregation(example.keys())


# In[215]:


people_names = 'fred bob sally panda velma gis mercator machester xml parsing'.split(' ')

people_preferences = {}

for person in people_names: people_preferences[person] = [0 for _ in range(10)]


# In[216]:


people_preferences


# In[212]:




count = 0
for house in json.load(open('_houses.json')):
    for col in house['coefficents']:
        if col == 0: count += 1
            
print(count)


# In[209]:


h3_cells


# In[45]:


_houses


# In[27]:


preferences


# In[29]:



h3_cells


# In[ ]:


num_houses


# In[5]:


import os

os

os.listdir('./data')


# In[ ]:


# perform migrations -> 0 data
# 7 million airbnbs
# for each city in airbnb ?
#     get geocoordinates of borders of city + 2 miles?
#        teseelate that into boxes?
#                        


#airbnb, yelp, etc


# In[3]:


from PIL import Image
import requests
from io import BytesIO
longitude =-95.731003
latitude = 29.746700
zoom =16.25
tilt = 0
rotation = 0

res = requests.get(f'https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v11/static/{longitude},{latitude},{zoom},{tilt}, {rotation}/600x600?access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg')


image = Image.open(BytesIO(res.content))
image


# In[3]:


async def _():
    pass

import inspect


inspect.iscoroutinefunction(_)


def __():
    pass

inspect.iscoroutinefunction(__)


# In[43]:


import glob

coords = glob.glob('*_geoCoordinates.json')

import json

json.load(open(coords[0]))


# In[53]:


# #given a geolocation -> output like 5 metrics 
# #hipsterness -> mentions of hipster per square mile 
# #


# #reverse - frequency of words in twitter for each city in the top 100

# # installation
# !pip install easyocr

# import easyocr

# reader = easyocr.Reader(['en'])
# extract_info = reader.readtext(img_path1)

# for el in extract_info:
#    print(el)


#characterize neighborhoods as hipster, density of _ foods, 
#for every building in negihborhood -> get classification
#mueseum, history, culture, quiet, family, loud, nightlife, parks, architecture

# safetey
#transportation

#proximity to waterfront
#sum up complaints in each area -> find complaints above norm not just density of complaint

#word cloud for reviews of google places api

#curb appeal
#https://metropolismoving.com/blog/neighborhoods-in-manhattan-explained/

#https://developer.twitter.com/en/docs/twitter-api/enterprise/rules-and-filtering/enterprise-operators#listofoperators
#https://developer.twitter.com/en/docs/twitter-api/v1/geo/places-near-location/api-reference/get-geo-search

twit_url = 'https://api.twitter.com/1.1/geo/search.json'
res = requests.post(twit_url, json={
  "lat": -0.8957793,
  "long": 119.8679974,
})

#find all bars, restauarnts and shit near an airbnb - automatically sort them
#find all blogs about place -> find words of interest


#create data sets but especially geospatial

latitude =  -0.8957793
longitude = 119.8679974


# In[60]:


l=list(cache)
l


# In[68]:





# In[53]:


coords = [-71.126085, 42.25230]
import requests
longitude = coords[0]
latitude = coords[1]
latitude, longitude = 139.766828, 35.668613

contours_minutes = 2
isochrone_url = f'https://api.mapbox.com/isochrone/v1/mapbox/driving-traffic/{latitude}%2C{longitude}?contours_minutes={contours_minutes}&polygons=true&denoise=0&generalize=0&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg'
geojson_data = requests.get(isochrone_url).json()


# In[54]:


#print(coffee_shops, 'coffee_shops')
#print('geojson_data', geojson_data)
data = []
from shapely.geometry import shape, Point

for shop in coffee_shops: 
    if 'lat' not in shop or 'lon' not in shop: 
        #print(shop)
        continue
    point_to_check = Point(shop['lon'], shop['lat'])
    for feature in geojson_data['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point_to_check):
            data.append(shop)
len(data)


# In[55]:


coffee_shops = fetch_coffee_shops(longitude, latitude)
coffee_shops


# In[ ]:


polygon.contain()


# In[49]:




polygon = shape(geojson_data['features'][0]['geometry'])
polygon


# In[47]:


geojson_data['features'][0]['geometry']


# In[13]:


coords_1 = [52.2296756, 21.0122287]
coords_2 = str(coords_1[0]) + ':' + str(coords_1[1])
coords_3 = coords_2.split(':')
coords_4 = [float(coords_3[0]), float(coords_3[1])]
coords_4


# In[77]:


import geopy.distance

coords_2 = (52.406374, 16.9251681)

 

def geoDistance(one, two):
    return geopy.distance.geodesic(one, two).km


#for each airbnb -> key by distance to closest shopping place
#all all shopping types to select box -> which lets you pick what you care about
#bookstore, bike store, clothes -> give each a coefficent from -1 to 1 ?


# In[73]:


all_json[0]


# In[40]:


import requests
address = '1600 Amphitheatre Parkway, Mountain View, CA'
address = 'SUNDAR NAGAR sneer ae'
# Replace YOUR_ACCESS_TOKEN with your Mapbox access token
#geoCode('HYDE PARK NN 5208')
#avoid this https://www.wsj.com/lifestyle/travel/vacation-hotels-flights-online-booking-fees-29262506
#traveling in a poopy way is an unseen problem


# In[64]:


import glob
import json 
image_urls = glob.glob('./*.json')
import pytesseract
##put city as name of file

def parseUrl(url):
    import re
    url = "https://www.airbnb.com/rooms/21363658?adults=1&children=0&enable_m3_private_room=true&infants=0&pets=0&check_in=2023-10-15&check_out=2023-10-20&source_impression_id=p3_1694806265_9ywrM3JxkoiBHIp%2F&previous_page_section_name=1000&federated_search_id=c9de1c68-04a4-4dce-8f81-60648418af40"
    match = re.search(r'/rooms/(\d+)', url)
    return match.group(1)

#apt has 6 images -> get coordinates -> search in mb -> get distance from APT to closest one
# def givenListOfImagesInJson(src='0.json'):
#     imgs = json.load(open(src))
#     for _ in image_urls:
#         gm = json.load(open(_))
#         coordinates = [saveImage(_) for _ in gm]
from PIL import Image

# Open an image file
image = Image.open('_.png')

# Display the image
#image.show()

#!pip install easyocr

import easyocr


# fp='gm_395995.json'
# images=json.load(open(fp))
# #imageToCoords(images)

# response = requests.get(images[0])
# response.content
# with open('_.png', 'wb') as f:
#         f.write(response.content)
# pytesseract.image_to_string('_.png', lang='eng')


# In[14]:


#_ =[json.load(open(_)) for _ in glob.glob('./gm*.json')]
import json
import requests
import easyocr
def imageToCoords(url_list, location='jaipur,india'):
    cache = set()

    for _ in url_list[:5]:
        if len(_) < 10: continue
        response = requests.get(_)
        if response.status_code == 200:
            with open(_[-50:-1], 'wb') as f:
                f.write(response.content)

        ocr = ocrImage(_[-50:-1])
        if not ocr: continue
        coords = geoCode(ocr)
        print(coords)
        if not coords: continue
        cache.add(str(coords[0]) + ':' + str(coords[1]))
    return list(cache)

def key_function (_):
    return _[1]

def ocrImage(fp):
    reader = easyocr.Reader(['en'])
    extract_info = reader.readtext(fp)
    #print(extract_info)
    from time import time
    sorted(extract_info, key=key_function)
    #if 0 not in extract_info: return print('wtf', extract_info)
    #print(extract_info)
    if (not extract_info): return False
    return extract_info[0][1]   

def geoCode(address = "1600 Amphitheatre Parkway, Mountain View, CA"):
    accessToken = "pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg"  # Replace with your actual access token

    geocodeUrl = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}%2C%20singapore.json?access_token={accessToken}"

    response = requests.get(geocodeUrl)
    data = response.json()

    if 'features' in data and len(data['features']) > 0:
        location = data['features'][0]['geometry']['coordinates']
        #print(f"Longitude: {location[0]}, Latitude: {location[1]}")
        return location

geocords = [imageToCoords(_) for _ in _] 
#each _ is an appartment -> 
#have to ocr 6 iamges or find a better ocr function

#ocr('_.png')
_ = json.load(open('gm_54155787.json'))
imageToCoords(_)


# In[ ]:


geoCoordinates = [[75.775256, 26.897392],
[75.818982, 26.915458],
[75.822858, 26.915904],
[75.632758, 26.697753],
[6.9561, 50.944057],
[75.787137, 26.933939],
[75.804092, 26.815534],
[75.822858, 26.915904],
[75.78857, 26.919951],
[75.818982, 26.915458],
[75.775256, 26.897392],
[75.818982, 26.915458],
[75.632758, 26.697753],
[75.822858, 26.915904],
[6.9561, 50.944057],
[75.818982, 26.915458],
[6.9561, 50.944057],
[75.739386, 26.936794],
[75.804092, 26.815534],
[30.203709, 59.987311],
[75.784422, 26.8726],
[75.203905, 26.464273],]


def getPlacesOfInterest(aptGeoLocation):
#     print(aptGeoLocation)
#     aptGeoLocation = aptGeoLocation.split(':')
#     aptGeoLocation =  [float(aptGeoLocation[0]), float(aptGeoLocation[1])]
    all_json = []
    latitude = aptGeoLocation[0]
    longitude = aptGeoLocation[1]
    url = f""" https://api.mapbox.com/search/searchbox/v1/category/shopping?access_token=pk.eyJ1Ijoic2VhcmNoLW1hY2hpbmUtdXNlci0xIiwiYSI6ImNrNnJ6bDdzdzA5cnAza3F4aTVwcWxqdWEifQ.RFF7CVFKrUsZVrJsFzhRvQ&language=en&limit=20&proximity={longitude}%2C%20{latitude}"""
    _ = requests.get(url).json()
    print(_)
    for place in _['features']:
        #print(place)
        all_json.append(place)
    poi = []
    for place in all_json:
        coords = place['geometry']['coordinates']
        categories = place['properties']['poi_category']
        poi.append([coords, categories])
        #print(place)
    sorted(poi, key=lambda _: geoDistance(_[0], aptGeoLocation))
    return geoDistance(poi[0][0], aptGeoLocation)


distance = [getPlacesOfInterest(geoCoordinate) for geoCoordinate in geoCoordinates]


# In[82]:


shit = [([[193, 25], [255, 25], [255, 39], [193, 39]], 'REC COL', 0.8501251496426608), ([[1, 75], [57, 75], [57, 91], [1, 91]], 'AWALA', 0.9991425927777027), ([[0, 94], [24, 94], [24, 102], [0, 102]], '4iG', 0.1567160244474485), ([[191, 99], [245, 99], [245, 117], [191, 117]], 'BLOCK', 0.9795149986694393), ([[199, 115], [233, 115], [233, 129], [199, 129]], 'Elilich', 0.11260652166395571), ([[222.71963120067105, 38.463557440805246], [254.6924656801795, 35.93437346062772], [256.280368799329, 53.536442559194754], [224.3075343198205, 55.06562653937228]], 'REC', 0.982265947512128), ([[215.21913119055696, 53.375304952445575], [253.8788534316657, 47.52290813709577], [255.78086880944304, 62.624695047554425], [217.1211465683343, 67.47709186290423]], 'TlF', 0.0781317186115469), ([[150.21913119055696, 82.37530495244557], [189.91785703596014, 78.60308884931453], [191.78086880944304, 91.62469504755443], [151.08214296403986, 95.39691115068547]], 'Vaishali ~', 0.3651379495284656), ([[189.57006641960766, 76.09713948117607], [217.9953500135553, 85.90367580513981], [213.42993358039234, 98.90286051882393], [185.0046499864447, 89.09632419486019]], 'Mg', 0.38071738696253066)]
shit



shit[0]


# In[ ]:


#distance to POI 
#add columns for hipsterish, craft beer variety, properties of an area that were hard to google
#some areas -> really cool art
#some areas -> really great parks
#some areas -> things people said on twitter -> 5 characteristics of neighborhood 
#proximity to water -> graph search -> search through geoJSON features 
async def getAirbnbForCityAndGetDistanceToPoI(location):
    await main(location)
    #saveImage -> get lat long from city for each address
    #for each address -> get POI
    #then calc distance for each address to POI
    #add more interesting + useful things
    
await getAirbnbForCityAndGetDistanceToPoI('cario,egypt')


# In[52]:


list(cache)


# In[1]:


docs ='https://reflect.site/g/awahab/win-kaggle-w-english/22ab82085c9143b194b81c6111266145'

import requests

r= requests.get(docs)
#r.text

from bs4 import BeautifulSoup

_ = BeautifulSoup(r.text)
#_.text


# In[8]:


subprocess.run(['node', 'airbnb_get_img_url.js', 'jaipur--india_apt.json'])


# In[18]:


import requests
import json

def fetch_cafes(minlat, minlon, maxlat, maxlon):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
        node["amenity"="cafe"]({minlat},{minlon},{maxlat},{maxlon});
        way["amenity"="cafe"]({minlat},{minlon},{maxlat},{maxlon});
        relation["amenity"="cafe"]({minlat},{minlon},{maxlat},{maxlon});
    );
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data

# Example usage:
minlat, minlon, maxlat, maxlon = 40.7128, -74.0060, 40.7138, -73.0050  # Example coordinates for New York City
cafes = fetch_cafes(minlat, minlon, maxlat, maxlon)
print(json.dumps(cafes, indent=4))


# In[17]:


getCityList().keys()


# In[16]:


#2 cities
#moscow 

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


locations = getCityList()

    
    


import json
import asyncio
from pyppeteer import launch


async def delay(seconds):
    await asyncio.sleep(seconds)

links = []
headless = True

async def get_img_url(apt_listing, idx):
    print(apt_listing, idx)
    browser = await launch(headless=headless)
    page = await browser.newPage()
    await page.setViewport({'width': 1920, 'height': 1080})
    await page.goto(apt_listing)

    await page.waitForSelector('section')

    await page.evaluate('''() => {
        window.scrollBy(0, 100000);
    }''')

    await delay(3)

    await page.evaluate('''() => {
        //window.scrollBy(0, 100000);
        //document.querySelectorAll('.hnwb2pb.dir.dir-ltr')[3]?.scrollIntoView()
        //document?.querySelector('.cj0q2ib.sne7mb7.rp6dtyx.c1y4i074.dir.dir-ltr').click()
    }''')

    await delay(1)

    selector = '.gm-style img'
    img_urls = await page.querySelectorAllEval(selector, 'img => img.map(i => i.src).filter(src => src.indexOf("maps.googleapis.com") !== -1)')

    print('saving to ' + f'{idx}.json')
    with open(f'{idx}.json', 'w') as f:
        json.dump(img_urls, f)

    #print(img_urls)
    #await browser.close()
    return img_urls

async def get_apt(url, location):
    browser = await launch(headless=headless)
    page = await browser.newPage()

    await page.goto(url)
    qs = '.cy5jw6o.dir.dir-ltr a'

    try:
        await page.waitForSelector(qs)
    except:
      return []

    # Placeholder for your logic to extract tweets, adapt as needed
    tweets = await page.querySelectorAllEval(qs, '''tweets => {
        return tweets.map(tweet => ({
            link: tweet.href
        }));
    }''')

    #print(tweets)
    #link += tweets

    await browser.close()
    with open(f'{location}_apt.json', 'w') as f:
        json.dump(tweets, f)
    return tweets

#if __name__ == "__main__":
#fn = input("Enter function to execute (get_apt/get_img_url): ")
#if fn == 'get_apt':
#    asyncio.get_event_loop().run_until_complete(get_apt())
#elif fn == 'get_img_url':
    #apt_listing = input("Enter the apartment listing URL: ")
apt_listing = 'https://www.airbnb.com/rooms/586166376600248074?adults=1&category_tag=Tag%3A8522&children=0&enable_m3_private_room=true&infants=0&pets=0&photo_id=1457367815&search_mode=flex_destinations_search&check_in=2023-09-14&check_out=2023-09-19&source_impression_id=p3_1694719767_8kxnr2mVDAfpYdMo&previous_page_section_name=1000&federated_search_id=62cba934-7640-4911-b5c5-cb37f6420f19'
#asyncio.get_event_loop().run_until_complete(get_img_url(apt_listing))

#for city in list of cities
#search airbnb
#cache listing_url in json
#for listing_url -> get geo-coordinate

#check availability weekly -> 7 million = 
#86,400 seconds => 1 computer 

import asyncio
#loop = asyncio.get_event_loop()
#loop.create_task(get_img_url(apt_listing))
def getAllApt_(location):
    return ['hello world']

#make server async -> 10 ?? 
#use nodejs subprocess -> 2 min 
async def getAllAptForLocation(location):
    location = location.replace(',','--')
    url = f'https://www.airbnb.com/s/{location}/homes' 
    task = asyncio.create_task(get_apt(url, location)) #get 18 apt urls 
    return await task

async def main(location):
    # run my_coroutine as a task
    task = getAllAptForLocation(location)

    url = await task
    #return print('task done', len(url))
    for idx, url in enumerate(url):
        task = asyncio.create_task(get_img_url(url['link'], location))
        #task = asyncio.create_task(get_img_url(apt_listing))
        await task

# run main() in the current event loop
# for location in locations['Asia']:
#     print('lets go to ' + location)
#     await main(location)


# In[43]:


#get every property on airbnb
#get every house in the world
#get every appartment in the world
#re-index every 24 hours

#make it simple to "suitability-analysis" each one with 10-20 columns or whatever is available in that city
#make an "admin UI" that is viewable by all so everyone can see all the different data for geospatial 




#also do timeseries
#also do tabular data

#geospatial ETA - 1 week
url


# In[68]:


#Image(url=i).
import requests



#pytesseract.image_to_string()
#Image(filename='./local_image.jpg')


# In[13]:


import os

#import os
os.listdir()

#print('local_image.jpg' in os.listdir())

#'local_image.jpg' in os.listdir()


# In[36]:


#! pip install pytesseract
import pytesseract
#pytesseract.image_to_string(, lang='eng')
print(pytesseract.image_to_string(Image.open('_.png'), lang='eng'))


# In[37]:


Image.open('_.png')


# In[18]:


import json
import requests
import pytesseract
url = json.load(open('./airbnb_map.json'))



for i in url:
    #from IPython.display import display, Image
    saveImage(i)
    # Display an image from a URL
    #display(Image(url=i))
    


# In[3]:


#! pip install pyppeteer
#https://github.com/pyppeteer/pyppeteer
import subprocess

location = 'jaipur--india'
args = [
    "node",
    "getAptInCity.js",
    location
]

# Execute the command
# try:
#result = subprocess.run(command, check=True)
# except subprocess.CalledProcessError as e:
#     print(f"The command failed with error: {e}")
completed_process = subprocess.run(args
                                   #, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
                                   )


# In[31]:



args = [
    "node",
    "RPC/fetchAirbnb.js",
    "get_img_url",
    "https://www.airbnb.com/rooms/46167540?adults=1&children=0&enable_m3_private_room=true&infants=0&pets=0&check_in=2023-09-17&check_out=2023-09-22&source_impression_id=p3_1694700080_UjFiBEV3n8iEBFMu&previous_page_section_name=1000&federated_search_id=c852781b-8f44-4187-a236-0aeb4c288fa8"
]

# Execute the command
# try:
#result = subprocess.run(command, check=True)
# except subprocess.CalledProcessError as e:
#     print(f"The command failed with error: {e}")
completed_process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
stdout = completed_process.stdout.decode()
stderr = completed_process.stderr.decode()


# In[35]:


args = [
    "node",
    "RPC/fetchAirbnb.js",
    "get_img_url",
    "https://www.airbnb.com/rooms/46167540?adults=1&children=0&enable_m3_private_room=true&infants=0&pets=0&check_in=2023-09-17&check_out=2023-09-22&source_impression_id=p3_1694700080_UjFiBEV3n8iEBFMu&previous_page_section_name=1000&federated_search_id=c852781b-8f44-4187-a236-0aeb4c288fa8"
]

output = subprocess.check_output(args)

result = output
result


# In[3]:


import subprocess


url = "node RPC/fetchAirbnb.js  get_img_url 'https://www.airbnb.com/rooms/46167540?adults=1&children=0&enable_m3_private_room=true&infants=0&pets=0&check_in=2023-09-17&check_out=2023-09-22&source_impression_id=p3_1694700080_UjFiBEV3n8iEBFMu&previous_page_section_name=1000&federated_search_id=c852781b-8f44-4187-a236-0aeb4c288fa8'"
subprocess.call(url.split(' '))


# In[ ]:


from PIL import Image

import pytesseract

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
print(pytesseract.image_to_string(Image.open('test.png')))

# In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# NOTE: In this case you should provide tesseract supported images or tesseract will return error
print(pytesseract.image_to_string('test.png'))

# List of available languages
print(pytesseract.get_languages(config=''))

# French text image to string
print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))
# Batch processing with a single file containing the list of multiple image file paths
print(pytesseract.image_to_string('images.txt'))

# Timeout/terminate the tesseract job after a period of time
try:
    print(pytesseract.image_to_string('test.jpg', timeout=2)) # Timeout after 2 seconds
    print(pytesseract.image_to_string('test.jpg', timeout=0.5)) # Timeout after half a second
except RuntimeError as timeout_error:
    # Tesseract processing is terminated
    pass

# Get bounding box estimates
print(pytesseract.image_to_boxes(Image.open('test.png')))

# Get verbose data including boxes, confidences, line and page numbers
print(pytesseract.image_to_data(Image.open('test.png')))

# Get information about orientation and script detection
print(pytesseract.image_to_osd(Image.open('test.png')))

# Get HOCR output
hocr = pytesseract.image_to_pdf_or_hocr('test.png', extension='hocr')

# Get ALTO XML output
xml = pytesseract.image_to_alto_xml('test.png')
#Support for OpenCV image/NumPy array objects

# import cv2

# img_cv = cv2.imread(r'/<path_to_image>/digits.png')

# # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# # we need to convert from BGR to RGB format/mode:
# img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
# print(pytesseract.image_to_string(img_rgb))
# # OR
# img_rgb = Image.frombytes('RGB', img_cv.shape[:2], img_cv, 'raw', 'BGR', 0, 0)
# print(pytesseract.image_to_string(img_rgb))


# In[262]:


trees_map()


# In[261]:


#take the text from my favorite books and render all the passages 
#create javascript bookmarklets and execute on them on pages
#make a chrome extension -> extract text from book and visualie and make notes
#make flash cards 
#make actions and functions that dont exist in programming languages 

# render a 3d model
# simplify the triangles from center to right side
# add a bezel on left hand corner - line by line execution -> CR line by line 



# for all news articles in Economist for last year
# find which ones have most relevance to astrophysics 


# make a game

# play a game 
# run forward 
# shoot lemons 
# jump every 5 seconds


# when i leave home
# ask rosy to wash dishes, clean bed, and empty litter box
# if run out of litter or laundry detergent or tortillas, order from instacart 






##https://www.zenrows.com/blog/pyppeteer#use

#how to make best magic english notebook ever
#write 5 noetbooks -> make code super clean and well documented -> use cursor -> relearn emacs 
#make it so people can easily contribute by reading notebook and adding their own function






def trees_map():
    trees_CSV = 'data/2015_Street_Tree_Census_-_Tree_Data.csv'
    from collections import defaultdict
    pointList = []

    with open(trees_CSV) as file:
        for sentence in file:
            parameters = sentence.split(',')
            pointList.append(parameters[-8:-6])
    return pointList[1:]


def trees_histogram():
    trees_CSV = 'data/2015_Street_Tree_Census_-_Tree_Data.csv'
    from collections import defaultdict
    counter = defaultdict(int)

    with open(trees_CSV) as file:
        for sentence in file:
            parameters = sentence.split(',')
            count = parameters[8]
            counter[count] += 1
    return counter


# In[249]:


json.load(open('./data/twitch.json', 'r'))


# # generate some code -> for values that are missing -> genreate UI inputs and fill them in - pizza - CC 
# 
# 
# make a circle in each corner
# make a line connecting each circle
# # english language programming environment 
# #generates python + javascript 
# 
# #program synthesis
# how to generate new programs that werent possible in previous languages 
# when netflix stock price goes up by 1% get me a pizza ->
# #run 20 lines of javascript that exectures like 300 lines of javascript 
# 
# 
# for all uber trips in nyc from 2012-2014
# visualize all the dropoffs  - dont have to know what the data structure is, or where its from, or how its plotted 
# 
# 
# #streams filters maps reduces
# 
# watch tweets mentioning pizza 
# download the meme 
# 
# 
# for all chemical reactions
# find ones which improve synthesis
# add a trait for purple glow in the dark
# add better biofuels for algae 
# and then send me an SGRNA kit 
# 
# 
# 
# for all twitch comments 
# find topics people talk about like win trading
# when win trading comments goes above 5 in an hour -> record the date in a json 
# 
# 
# 
# 
# for all stars in the sky, find dates when each can be best observed
# 
# for all planets in the solar system
# when they are brightest, send me an email 
# 
# 
# 
# 
# 
# find all papers on arxiv relating to astronomy 
# find a time series of citations for each one 
# find sentence pairs that are close in semantic similarity 
# get all diagrams and create one pdf out of 10,000 pdfs 
# 
# 
# 
# #remote year + community automation
# consensus based suitability analysis 
# :pollData for library, subway and supermarket
# find all apt in sf with noise < .5 complaints per sq mile 
# for each apt, find which ones are closest to library, subway and super market
# for each of those, find 5 PoI for tourists that are in 1 hour train distance 
# make a schedule from those PoI 

# In[232]:


shit


# In[244]:


shit[shit.spc_common == 'pin oak']
spc = shit.groupby('spc_common')
spc.sum()


# In[245]:


df = shit
bins = [0, 10, 20, 30, 40, 50]

df['ValueBin'] = pd.cut(df['spc_common'], bins, labels=['0-10', '11-20', '21-30', '31-40', '41-50'])

# Group by 'Category' and 'ValueBin' and count frequencies
hist_data = df.groupby(['Category', 'ValueBin']).size().reset_index(name='Frequency')

print(hist_data)


# In[74]:


#only thing that makes sense is to finish a dmeo 

#figure out how to run similar code on different data -> match columns by snyonym and ask user -> are these the same?
#complaint
CSV_PATH = './311_cases.csv'
import subprocess
from tqdm import tqdm
import pandas as pd 
import os
from h3 import h3
import warnings
warnings.filterwarnings('ignore')

def process_date(df_chunk, key):
    df_chunk[key] = df_chunk[key].str.slice(0, 16)
    #df_chunk[key] = pd.to_datetime(df_chunk[key], utc=True, format='%Y-%m-%d %H:%M')

def binRow(df311):
    APERTURE_SIZE = 9
    hex_col = 'hex'+str(APERTURE_SIZE)
    df311[hex_col] = df311.apply(lambda x: h3.geo_to_h3(x.Latitude,x.Longitude,APERTURE_SIZE),1)
    df311g = df311.groupby(hex_col).size()#.to_frame('cnt').reset_index()
    return (df311g)

def saveComplaint(complaint):
    print(complaint)
    data = pdf.loc[pdf['Category'] ==(complaint)]
    binned = binRow(data)
    complaint = complaint.replace(' ', '-').replace('/','-')
    hi = [(hex, count)  for hex,count in binned.to_dict()]
    with open('./data/'+complaint+'.json', 'w') as outfile: json.dump(hi, outfile)
    
stuff = ['Noise - Residential',                         
'HEAT/HOT WATER' ,                              
'Street Condition',                           
'Illegal Parking ',                              
'Blocked Driveway'  ,                            
'Street Light Condition' ,                       
'HEATING'          ,                             
'PLUMBING'     ,                                 
'Water System'  ,                                
'Noise - Street/Sidewalk' ,                      
'GENERAL CONSTRUCTION'  ,                                                                
'UNSANITARY CONDITION']                          
import json
import os
import requests
#get all the airbnb in sf that are not noisy
#getAirbnb() => geocoding mapbox api => lat/lng => h3a9 => lookup in binned table
#address => geocoe => lat/lng => h3a9 => lookup in binned table 


        
ll = geoCode('450 10th St, San Francisco, CA 94103')
h3Cell = h3.geo_to_h3(ll[0], ll[1], 9)
#figure out how to make fast
#one line to say -> find the least noisy airbnb in SF and NYC
#also fast to say -> find the noisiest airbnb in SF and NYC
#plot both on a map 
#in regular python data analysis -> takes 2-3 hours for an experienced dev
#in english -> takes 16 ms -> onkeypress get instant results
#how to cache 311 -> 3gb for sf and 16gb for nyc -> cant cache on browser
#h3_cells for noise column =
#100ms is probably fine for onKeyPress RPC

#poll = sf, houston, kansas
#get all airbnb in poll.1 if not noisy and near pizza
#AllAirbnbInCityFilteredNoise('sf')
def make_pandas_data_frame():
    print (f'Exact number of rows: {n_rows}')
    df_tmp = pd.read_csv(CSV_PATH, nrows=5)
    df_tmp.head()


    types = {'Category': 'category', 
                  'Longitude': 'float32',
                  'Latitude': 'float32',
            }
    cols = list(types.keys())
    chunksize = 5_000_000

    df_list = [] # list to hold the batch dataframe
    for df_chunk in tqdm(pd.read_csv(CSV_PATH, usecols=cols, dtype=types, chunksize=chunksize)):
        df_list.append(df_chunk) 
    pdf = pd.concat(df_list)
    filtered_df = pdf[(pdf['Latitude'] != 0) & (pdf['Longitude'] != 0)]
    pdf = filtered_df
    del df_list

    APERTURE_SIZE = 9
    hex_col = 'hex'+str(APERTURE_SIZE)
    pdf[hex_col] = 0
    return pdf

def getNoisyCells():
    fp = './data/sf-noise-regions.csv'
    if os.path.isfile(fp):
        print('cached')
        return json.loads(open(fp).read())
    pdf = make_pandas_data_frame()
    data = pdf.loc[pdf['Category'] ==('Noise Report')]
    binned = binRow(data)
    with open(fp, 'w') as f:
        f.write(binned.to_json())
    return binned
getNoisyCells()

def meanDict(_): return sum([_[v] for v in _.keys()]) / len(_)


def isNoisy(nc, apt):
    #latlong = geoCode(addr)
    h3Cell = h3.geo_to_h3(apt['latitude'], apt['longitude'], 9) #.1 square kilometers
    return nc[h3Cell] < meanDict(nc)


def getAirbnbForCity():
    df = pd.read_csv('./data/sf_airbnb.csv')
    airbnb = df[['latitude', 'longitude', 'id', 'listing_url']].to_dict('records')
    return airbnb

def getAllAirbnbInCityThatAreNotNoisy(LTorGT):
    from collections import defaultdict
    all_airbnb = getAirbnbForCity() #make this get for different cities 
    nc = defaultdict(int, getNoisyCells())
    mean = meanDict(nc)
    filtered = [apt for apt in all_airbnb if 
                isNoisy(nc, apt) == LTorGT
                #nc[h3.geo_to_h3(apt['latitude'], apt['longitude'],9)] < mean
               ]
    listing_url = [apt['listing_url'] for apt in filtered]
    return listing_url

one= getAllAirbnbInCityThatAreNotNoisy()
two = airbnb
#one
one


# In[223]:


'airbnb' in 'find all airbnb that are not noisy and are near a yoga studio'


# In[212]:


# with open('./asdf.txt', 'w') as f:
#     json.dump(dict(binned), f)
from collections import defaultdict
#json.loads(getNoisyCells())
#meanDict(getNoisyCells())


# In[219]:


data = pdf.loc[pdf['Category'] ==('Noise Report')]
binned = binRow(data)
# with open(fp, 'w') as f:
#     json.dump(binned.to_csv(), f)
#dict(binned)

binned


# In[122]:


import csv
# with open('./data/sf_airbnb.csv', newline='') as csvfile:
#     #csvreader = csv.(csvfile, delimiter=' ', quotechar='|')
#     for k in csvfile:
#         print(k)
        
        
import pandas as pd

# Load a CSV file into a DataFrame

# Show the DataFrame

df.columns.tolist() #street, city, state, szpcode


df.latitude
df.longitude
df.id



df[['latitude', 'longitude', 'id']]

# Convert the DataFrame to a list of dictionaries
#list_of_dicts = filtered_df.to_dict('records')
#convert df to {latitude: _, longitude: _, listing_id}
#find a list of airbnb that are not noisy -> get a list of links back + a data browser
#df.columns.tolist()

def approxMatch(one, two):
    #get all synonyms for one and two
    return set(one + two)


# In[67]:


#saveComplaint('Noise Report')

with open('complaint.csv', 'w') as file:
    file.write(nc.to_csv())


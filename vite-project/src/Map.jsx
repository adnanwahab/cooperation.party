import {useRef, useMemo, useState, useEffect} from 'react';
import {createRoot} from 'react-dom/client';
import ReactMap, {Source, Layer, Marker} from 'react-map-gl';
import DeckGL, {GeoJsonLayer, ArcLayer} from 'deck.gl';
import {range} from 'd3-array';
import {scaleQuantile} from 'd3-scale';
import { Listbox } from '@headlessui/react'


const typesOfPlaces = [
  { id: 1, name: 'coffee', unavailable: false },
  { id: 2, name: 'Bar', unavailable: false },
  { id: 3, name: 'Library', unavailable: false },
  { id: 4, name: 'Grocery', unavailable: true },
  { id: 5, name: 'yoga', unavailable: false },
]

function MyListbox() {
  const [selectedPerson, setSelectedPerson] = useState(typesOfPlaces[0])

  return (
    <Listbox value={selectedPerson} onChange={setSelectedPerson}>
      <Listbox.Button>{selectedPerson.name}</Listbox.Button>
      <Listbox.Options>
        {typesOfPlaces.map((person) => (
          <Listbox.Option
            key={person.id}
            value={person}
            disabled={person.unavailable}
          >
            {person.name}
          </Listbox.Option>
        ))}
      </Listbox.Options>
    </Listbox>
  )
}

async function getIsochrone(longitude, latitude, contours_minutes) {
  const accessToken = 'your-access-token-here';
  const isochroneUrl = `https://api.mapbox.com/isochrone/v1/mapbox/driving-traffic/${longitude}%2C${latitude}?contours_minutes=${contours_minutes}&polygons=true&denoise=0&generalize=0&access_token=${accessToken}`;

  try {
      const response = await fetch(isochroneUrl);
      
      if (response.ok) {
        const geojsonData = await response.json();
          console.log(geojsonData);
          // Do something with the geojson data
      } else {
          console.log('Failed to fetch isochrone data:', response.status, response.statusText);
      }
  } catch (error) {
      console.error('Error fetching isochrone data:', error);
  }
}
// Call the function
const colors = [
  [0, '#3288bd'],
  [1, '#66c2a5'],
  [2, '#abdda4'],
  [3, '#e6f598'],
  [4, '#ffffbf'],
  [5, '#fee08b'],
  [6, '#fdae61'],
  [7, '#f46d43'],
  [8, '#d53e4f']
]

const dataLayer = () => {
  let color = colors[Math.floor(Math.random() * 8)][1]
  console.log(color)
  return {
  id: 'data',
  type: 'fill',
  paint: {
    'fill-color': color,
    'fill-opacity': 0.8
  }
}}

async function fetchCoffeeShops() {
  const latitude = 37.8;
  const longitude = -122.4;
  const radius = 1000; // radius in meters

  // Build Overpass API URL
  const query = `[out:json][timeout:25];` +
                `(node["amenity"="cafe"](${latitude - 0.01},${longitude - 0.01},${latitude + 0.01},${longitude + 0.01});` +
                `way["amenity"="cafe"](${latitude - 0.01},${longitude - 0.01},${latitude + 0.01},${longitude + 0.01});` +
                `relation["amenity"="cafe"](${latitude - 0.01},${longitude - 0.01},${latitude + 0.01},${longitude + 0.01}););` +
                `out body;`;
  const overpassUrl = `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`;
  
  try {
    const response = await fetch(overpassUrl);
    
    if (response.ok) {
      const data = await response.json();
      const coffeeShops = data.elements;
      console.log("Coffee Shops:", coffeeShops);
      // Do something with the fetched coffee shops
    } else {
      console.log('Failed to fetch data:', response.status, response.statusText);
    }
  } catch (error) {
    console.error('Error fetching coffee shops:', error);
  }
}

function Map(props) {
  const mapRef = useRef();

  if (props.data === 'hello world' ||
  props.data === 'hello-world'
  ) return <></>

  // const geoJson = props.data[0]
  // const coffeeShops = props.data[0][0]
  const latitude = props.data[1][2]
  const longitude = props.data[1][3]

  const onClick = () => {
    console.log('hello')
  }

  const renderShop = (shop, index)=> {
    return <Marker
    key={`marker-${index}`}
    longitude={shop.lon}
    latitude={shop.lat}
    anchor="bottom"
    onClick={e => {
      // If we let the click event propagates to the map, it will immediately close the popup
      // with `closeOnClick: true`
      e.originalEvent.stopPropagation();
      // setPopupInfo(city);
    }}
  >
    {/* <Pin /> */}
  </Marker>
  }

  function merge (json) {
    return json.reduce((acc, item) => {
      acc.features.push(item.features[0])
      return acc
    })
  }

  let geoJson = props.data.map((listing, idx) => {
    return listing[1]
  })
  console.log('geoJson', geoJson);

  geoJson = merge(geoJson)

  let renderGeoJson = (<Source type="geojson" data={geoJson}>
      <Layer {...dataLayer()} />
    </Source>)

  let shopMarkers = props.data.map(listing => {
    return listing[0].map(renderShop)
  })

let places = ['commuteDistance', 'library', 'bar', 'coffee']
let placeCoefficents = {} 
places.forEach(place => {
  placeCoefficents[place] = 0
})
//sentences + componentData = list of component + states from props
//makeFn - for now just re-render and redo whole document - con might be slow w/o caching??
//CallFn - change a UI component 
const makeOnCoefficentChange = (name) => {
  return (e) => {
    placeCoefficents[name] = parseFloat(e.target.value) / 100
    setCoefficents(Object.assign({}, placeCoefficents))
  }
}
let coefficentsSliders = places.map((pair) => {
  return <><label>{pair}</label><input type='range' onChange={ makeOnCoefficentChange(pair)}/></>
})
let [getCoefficents, setCoefficents] = useState(placeCoefficents)

useEffect(() => {
  console.log('fetch to callFN here') //make do makeFN with better caching 
  //sentence defines not an endpoint but a function that can be called from endpoint
  //sentence -> returns a component
  //on interaction -> sends a networkRequest -> /rpc/function_name with json = parameters
  //returns function and then re-renders data
  console.log(getCoefficents, 'getCoefficents')
  let fn = async ()=>  {
    let _ = await fetch('http://localhost:8000/callFn/', {
      method: 'POST',
      body: JSON.stringify({
        getCoefficents
        // _k: Object.keys(getCoefficents).join(','),
        // _v: Object.values(getCoefficents).join(','),
            })
    })
    let data = await _.json()
    return data
  }
  fn().then((_) => {})
}, [getCoefficents])


  return (<>
  {coefficentsSliders}
    <ReactMap
       ref={mapRef}
        mapboxAccessToken="pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg"
      initialViewState={{
        longitude,
        latitude,
        zoom: 14
      }}
      style={{width: 1000, height: 2000}}
      onClick={onClick}
      // mapStyle="mapbox://styles/mapbox/streets-v3"
      mapStyle="https://api.maptiler.com/maps/streets/style.json?key=D8xiby3LSvsdgkGzkOmN"
    >
    {shopMarkers}
    {renderGeoJson}
    </ReactMap> 
    </>);
}
export default Map


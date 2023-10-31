import React, { useEffect, useState, useCallback} from 'react';
import * as d3 from 'd3';
import {H3HexagonLayer} from '@deck.gl/geo-layers';
import {Map} from 'react-map-gl';
import maplibregl from 'maplibre-gl';
import {HexagonLayer} from '@deck.gl/aggregation-layers';
import DeckGL from '@deck.gl/react';
import {GeoJsonLayer} from '@deck.gl/layers';
import {interpolateRainbow} from "https://cdn.skypack.dev/d3-scale-chromatic@3";
import {ScatterplotLayer} from '@deck.gl/layers';
//clone zillow + airbnb + houses in other countries
import { Transition } from '@headlessui/react'
import { WebMercatorViewport } from '@deck.gl/core';
import {DataFilterExtension} from '@deck.gl/extensions';
import {IconLayer} from '@deck.gl/layers';
import _ from 'underscore'
import * as turf from '@turf/turf';
import PopOver from './PopOver'

const ICON_MAPPING = {
  marker: {x: 0, y: 0, width: 128, height: 128, mask: true}
};
const INITIAL_VIEW_STATE = {
  longitude: 12,
  latitude: 29,
  zoom: 1.24,
  minZoom: 1,
  maxZoom: 17,
  pitch: 0,
  bearing: 0
}   
// INITIAL_VIEW_STATE.longitude = 139
// INITIAL_VIEW_STATE.latitude = 35
// INITIAL_VIEW_STATE.zoom = 10
function ProgressBar (props) {
  let style = {width: `${innerWidth}px`}
  return (<div className="fixed left-0 top-0 progress-bar h-4 w-full" style={style}></div>)
}

function AirbnbWorldMap(props) {
  const [cityData, setCityData] = useState([]);
  const [cityNames, setCityNames] = useState([])
  const [routes, setRoutes] = useState([])
  const [markers, setMarkers] = useState([])
  const [getPercent, setPercent] = useState(0)

  useEffect(() => {
    const fetchData = async () => {
      const request = await fetch(`https://shelbernstein.ngrok.io/cityList`)
      const cityList = await request.json()
      setCityNames(cityList)
    }
    fetchData()
  }, [])

  useEffect(() => {
    const fetchData = async () => {
      const CHUNK_SIZE = 10; // Number of requests to process at once, adjust as necessary
      
      const fetchCity = async city_name => {
          //await sleep(500);
          const req = await fetch(`https://shelbernstein.ngrok.io/data/airbnb/apt/${city_name}`);
          const json = await req.json();
          return json;
      };

      const newCityData = [...cityData]
  
      for (let i = 0; i < cityNames.length; i += CHUNK_SIZE) {
          const city_name = cityNames[i];
          const newData = await fetchCity(city_name)
          for (let key in newData) {
            cityData.push(newData[key]);
            //setCityData(cityData);
          }
      }

      //setCityData(newCityData);
  };  
    //fetchData();
}, [cityNames]);

  let layers = [
 
  ]

  layers.push( new ScatterplotLayer({
    id: 'airbnb+houses-within-bbox',
    //data: 'https://raw.githubusercontent.com/adnanwahab/cooperation.party/turkey2/data/city_location.json',
    pickable: false,
    opacity: 1.,
    stroked: false,
    filled: true,
    radiusScale: 1,
    radiusMinPixels: 1,
    radiusMaxPixels: 1,
    lineWidthMinPixels: 1,

    getPosition: d => {
      return d.reverse()
    },
    //getPosition: d => [d[0], d[1], 0],
    onClick: ({object}) => {
      let url = `https://www.airbnb.com/rooms/${object[0]}`
      window.open(url)
    },
    //getPosition: d => centroid,
    getRadius: (d, datum) => datum.index,
    getFillColor: d => {
      let rgb = d3.rgb(interpolateRainbow(d[0]))
      return [rgb.r, rgb.g, rgb.b]
    },
  }))

  layers.push( new ScatterplotLayer({
    id: 'scatterplot-layer',
    data: 'https://raw.githubusercontent.com/adnanwahab/cooperation.party/turkey2/data/city_location.json',
    pickable: false,
    opacity: 1.,
    stroked: false,
    filled: true,
    radiusScale: 1,
    radiusMinPixels: 1,
    radiusMaxPixels: 1,
    lineWidthMinPixels: 1,

    getPosition: d => {
      return d.reverse()
    },
    //getPosition: d => [d[0], d[1], 0],
    onClick: ({object}) => {
      let url = `https://www.airbnb.com/rooms/${object[0]}`
      window.open(url)
    },
    //getPosition: d => centroid,
    getRadius: (d, datum) => datum.index,
    getFillColor: d => {
      let rgb = d3.rgb(interpolateRainbow(d[1] / 365))
      return [rgb.r, rgb.g, rgb.b]
    },
  }))
  //i spent 2-3 years walking around 2-3 hours a day thinking about cool stuff -> could have had more optimal walking path made for me like strava  + train + world explorer
  let iconLayer = new IconLayer({
      id: 'icon-layer',
      //data: markers,
      data: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/bart-stations.json',
      pickable: true,
      // iconAtlas and iconMapping are required
      // getIcon: return a string
      iconAtlas: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png',
      iconMapping: ICON_MAPPING,
      getIcon: d => 'marker',
      onClick: (_)=> _.object.url,
      sizeScale: 150,
      getPosition: d => {
        return d.coordinates
        //lon first
        //return [+ d.lat, + d.lon]

        return [+ d.lon, + d.lat]
      },
      getSize: d => 50,
      getColor: d => { 
        console.log('wtf man')
        return [Math.random() * 255, 140, 0]
      }
    })
  //layers.push(iconLayer)

  const allCoordinates = []
  routes.forEach((route) => {
    route.geometry.coordinates.forEach((list, i ) => list.push(i / route.geometry.coordinates.length))
    allCoordinates.push(...route.geometry.coordinates)
  })
  //console.log('allCoordeinates', allCoordinates)
  let roads = new ScatterplotLayer({
    id: 'roads',
    data: allCoordinates,
    pickable: false,
    opacity: 1.,
    stroked: false,
    filled: true,
    radiusScale: 1,
    radiusMinPixels: 5,
    radiusMaxPixels: 5,
    lineWidthMinPixels: 1,
    getPosition: (one, two, three) => {
      return one
    },
    //getPosition: d => [d[0], d[1], 0],
    //getPosition: d => centroid,
    getRadius: d => 10,
    getFillColor: (_, datum) => {
      let rgb = d3.rgb(interpolateRainbow(_[2]))
      return [rgb.r, rgb.g, rgb.b]
      return [0, 255 * _[2], 255]
    },
    // getFilterValue: f => {
    //   //console.log(f)
    //   //f.properties.timeOfDay
    //   return Math.random () > .5
    // },  // in seconds
    //filterRange: [43200, 46800],  // 12:00 - 13:00
    //extensions: [new DataFilterExtension({filterSize: 1})]

  })
  //layers.push(roads)
  
  const [currentViewState, setViewState] = useState(computeBoundingBox(INITIAL_VIEW_STATE))
  const fetchRoutes = async () => {
    //everytime you move, draw contiguous paths from center to 4 corners
    let {top, left, right, bottom} = currentViewState;
    let url = `https://shelbernstein.ngrok.io/osm_bbox?min_lat=${top}&min_lng=${left}&max_lat=${bottom}&max_lng=${right}`;
    const response = await fetch(url);
    const json = await response.json();

    markers.push.apply(markers, json.places)

    setMarkers(markers)
    for (let i = 0; i < 1; i++) {
      let min_lat = left + .05 * Math.random(), 
      min_lng = top + .05 * Math.random(), 
      max_lat = right + .05 * Math.random(), 
      max_lng = bottom + .05 * Math.random();
      const req = await fetch(
        `https://api.mapbox.com/directions/v5/mapbox/driving/${min_lat},${min_lng};${max_lat},${max_lng}?alternatives=true&geometries=geojson&language=en&overview=full&steps=true&access_token=pk.eyJ1IjoiYXdhaGFiIiwiYSI6ImNrdjc3NW11aTJncmIzMXExcXRiNDNxZWYifQ.tqFU7uVd6mbhHtjYsjtvlg`
      );
      let json = await req.json();
      routes.push.apply(routes, json.routes)
      setRoutes(routes)
    }
    setPercent(0)
    setTimeout(function recur () {
      //setPercent(getPercent+.1)
      //if (getPercent < 1) setTimeout(recur, 100)
    }, 100)
    //setMarkers(json.places)
    //console.log(json.routes.length)
  }

  useEffect(() => {
    fetchRoutes();
  }, [currentViewState.left]);

  const [open, setOpen] = useState(false)

  return (<>
    <h3 className="">World Map! - Scroll to zoom in to see every home in the world at a higher resolution</h3>
    <div className="relative" style={{left: `${props.left}px`, height: '600px'}}>
    <PopOver open={open} setOpen={setOpen}/>
    <ColorLabels></ColorLabels>
    <ProgressBar percentage={100}></ProgressBar>
    <DeckGL
        width={1200}
        height={600}
      layers={layers}
      initialViewState={INITIAL_VIEW_STATE}
      controller={true}
      getTooltip={getTooltip}
      glOptions={{preserveDrawingBuffer: true}}
      onViewStateChange={_.debounce(({viewState}) => {
        //console.log('wtf dont do that')
        setViewState(computeBoundingBox(viewState))
      }, 1000)}
      // parameters={
      //   blendFunc: [GL.ONE, GL.ONE, GL.ONE, GL.ONE],
      //   depthTest: false
      // }
    >
      <Map 
       glOptions={{preserveDrawingBuffer: true}}
      reuseMaps mapLib={maplibregl} mapStyle={MAP_STYLE} preventStyleDiffing={true}>
      </Map>
    </DeckGL>

    </div>
    <div>
      <div>Latitude: {currentViewState.bottom.toPrecision(4)}  |   {currentViewState.top.toPrecision(4)}</div>
      <div>Longitude: {currentViewState.left.toPrecision(4)} |  {currentViewState.right.toPrecision(4)}</div>
      <div>Total Roads Drawn: {routes.length} / 1 billion</div>
      <div>Places Of Interest Drawn: {markers.length}</div>
      <div>Airbnbs rendered on Screen: {routes.length} /  7 million</div>
      <div>Benches rendered on Screen: {markers.length} /  7 million</div>
      <div>Houses rendered on Screen:  / 135 million from  
        <span className="text-color-blue-500"> Zillow</span> 
        <span className="text-color-red-500">Redfin</span> 
        <span className="text-color-green-500">Compass</span> 
        <span className="text-color-yellow-500">immobilienscout24.de</span>
      
      </div>
    </div>
    </>
  );
}
async function fetchInterestingData (min_lat, min_lng, max_lat, max_lng) {
  //SELECT * FROM points
  // WHERE amenity IS NOT NULL -- this condition filters for rows which represent some sort of amenity/POI
  // AND MbrWithin(geom, BuildMBR(xmin, ymin, xmax, ymax)); 
  const request = await fetch(`https://shelbernstein.ngrok.io/cityList`)
  const cityList = await request.json()
}
function clamp (_, min, max) {
  return Math.max(Math.min(_, max), min)
}


function computeBoundingBox(viewPort) {
  const viewport = new WebMercatorViewport({...viewPort});
  const topLeft = viewport.unproject([0, 0]);
  const bottomRight = viewport.unproject([viewport.width, viewport.height]);

  const boundingBox = {
    top: clamp(topLeft[1], -90, 90),
    left: clamp(topLeft[0], -180, 180),
    bottom: clamp(bottomRight[1], -90, 90),
    right: clamp(bottomRight[0], -180, 180),
    centroid: [viewport.latitude, viewport.longitude]
  };
  //console.log('bbox', boundingBox)
  //console.log('spatial lite and get rainbow routes')
  return boundingBox
}

export default function AirbnbPriceMap (){
const [isShowing, setIsShowing] = useState(true)
  return ( <>
        {/* <Transition
        show={isShowing}
        enter="transition-opacity duration-75"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="transition-opacity duration-150"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
      > */}
     <AirbnbWorldMap />
    {/* </Transition> */}
  </>)
}
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json';

export const colorRange = [
  [1, 152, 189],
  [73, 227, 206],
  [216, 254, 181],
  [254, 237, 177],
  [254, 173, 84],
  [209, 55, 78]
];

function getTooltip(params) {
    if (! params.picked) return 
    return JSON.stringify(params.object, null, 2)
    return params.object.url
}
//hex colors
function ColorLabels() {
    return <></>
}
async function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms)
  })
}

let layer = new HexagonLayer({
  id: 'hexagon-layer',
  //data: props.data,
  pickable: true,
  extruded: true,
  radius: 20000,
  radiusScale: 100,
  elevationScale: 4,
//    getPosition: d => d.slice(0, 2),
  getPosition: d => [d[1][1], d[1][0]].map(parseFloat),
});
//layers.push(layer)

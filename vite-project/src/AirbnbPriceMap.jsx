import React, { useEffect, useState, useCallback} from 'react';
import * as d3 from 'd3';
import {H3HexagonLayer} from '@deck.gl/geo-layers';
import {Map} from 'react-map-gl';
import maplibregl from 'maplibre-gl';
import {AmbientLight, PointLight, LightingEffect} from '@deck.gl/core';
import {HexagonLayer} from '@deck.gl/aggregation-layers';
import DeckGL from '@deck.gl/react';
import {GeoJsonLayer} from '@deck.gl/layers';
import {interpolatePurples} from "https://cdn.skypack.dev/d3-scale-chromatic@3";
import {ScatterplotLayer} from '@deck.gl/layers';
//clone zillow + airbnb + houses in other countries
import { Transition } from '@headlessui/react'
import { WebMercatorViewport } from '@deck.gl/core';
import {DataFilterExtension} from '@deck.gl/extensions';
import {IconLayer} from '@deck.gl/layers';
import _, { map } from 'underscore'

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


INITIAL_VIEW_STATE.longitude = 139
INITIAL_VIEW_STATE.latitude = 35
INITIAL_VIEW_STATE.zoom = 10
function ProgressBar (props) {
  let style = {width: `${props.percentage}%`}
  return (<div className="fixed top-0 progress-bar h-8 bg-blue-500" style={style}> </div>)
}

function AirbnbWorldMap(props) {
  const [cityData, setCityData] = useState([]);
  const [cityNames, setCityNames] = useState([])
  const [routes, setRoutes] = useState([])
  const [markers, setMarkers] = useState([])

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
    new ScatterplotLayer({
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
      getRadius: d => 10,
      getFillColor: d => [Math.random()* 255, 0, 255],
      // getFilterValue: f => {
      //   //console.log(f)
      //   //f.properties.timeOfDay
      //   return Math.random () > .5
      // },  // in seconds
      //filterRange: [43200, 46800],  // 12:00 - 13:00
      //extensions: [new DataFilterExtension({filterSize: 1})]

    })
  ]
  let places = []
  let iconLayer = new IconLayer({
      id: 'icon-layer',
      data: places,
      pickable: true,
      // iconAtlas and iconMapping are required
      // getIcon: return a string
      iconAtlas: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png',
      iconMapping: ICON_MAPPING,
      getIcon: d => 'marker',
      onClick: (_)=> _.object.url,
      sizeScale: 15,
      getPosition: d => [+ d.longitude, + d.latitude],
      getSize: d => 5,
      getColor: d => [Math.random() * 255, 140, 0]
    })


  routes.map((route, route_index) => {
    console.log(route)
    let newLayer = new GeoJsonLayer({
      id: 'geojson',
      //data: 
      data: route.routes[0].geometry,
      stroked: true,
      filled: true,
      lineWidthScale: 20,
      lineWidthMinPixels: 10,
      parameters: {
        depthTest: false
      },

      getLineColor: function (_) {
          //console.log(_)
          return [255, Math.random() * 255, 0, 255]
      },
      getLineWidth: function () {
          return 20
      },

      pickable: true,
     // onHover: setHoverInfo,

      // updateTriggers: {
      //   getLineColor: {year},
      //   getLineWidth: {year}
      // },

      transitions: {
        getLineColor: 1000,
        getLineWidth: 1000
      }
    })
    layers.push(newLayer)
  })
  console.log('layers', layers)
  const [currentViewState, setViewState] = useState(computeBoundingBox(INITIAL_VIEW_STATE))

  const fetchRoutes = async () => {

    const {top, left, right, bottom} = currentViewState;
    let url = `https://shelbernstein.ngrok.io/osm_bbox?min_lat=${top}&min_lng=${left}&max_lat=${bottom}&max_lng=${right}`;
    const response = await fetch(url);
    const json = await response.json();
    setRoutes(json.routes)
    console.log(json.routes.length)
  }

  useEffect(() => {
    fetchRoutes();
  }, [currentViewState.left]);



  return (<>
    <h3 className="">World Map! - Scroll to zoom in to see every home in the world at a higher resolution</h3>
    <h3 className="">{cityData.length}</h3>
    <div className="relative" style={{left: `${props.left}px`, height: '600px'}}>
    <ColorLabels></ColorLabels>
    {/* <ProgressBar percentage={props.data.length / 1e4}></ProgressBar> */}
    <ProgressBar percentage={cityData.length / 10}></ProgressBar>
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
      <div>Latitude: {currentViewState.top.toPrecision(4)} , {currentViewState.bottom.toPrecision(4)}</div>
      <div>Longitude: {currentViewState.left.toPrecision(4)} , {currentViewState.right.toPrecision(4)}</div>
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

function computeBoundingBox(viewPort) {
  const viewport = new WebMercatorViewport({...viewPort});
  const topLeft = viewport.unproject([0, 0]);
  const bottomRight = viewport.unproject([viewport.width, viewport.height]);

  const boundingBox = {
    top: topLeft[1],
    left: topLeft[0],
    bottom: bottomRight[1],
    right: bottomRight[0],
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

import DonutChart from "./DonutChart"
import Histogram from './Histogram'
import React, { useEffect, useState, useCallback, useRef} from 'react';
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
import {isWebGL2} from '@luma.gl/core';
import {ScreenGridLayer} from '@deck.gl/aggregation-layers';
import baseName from './httpFunctions'
import RainbowCat from './RainbowCat'




function Robot() {
  return <div>
    <></>
    <div>cat roomba</div>
    <>A roomba could follow a cat because robots should take care of cats because there are too many humans to take care of too few cats</>
    <>3 years ago i wanted my roomba to follow my cat</>
    <>100 million cats</>
    <>50 million cat owners</>
    <>50 million roombas sold</>
    
    <h3>Cool visualizations of RoS</h3>
    <button class="text-black">find cat</button>
    <canvas></canvas>
    <div>draw starcraft mode for roomba + robotic arm</div>
    <div>click on list of objects that ipad camera sees </div>
    <div>make an scv that can acquire objets and place them near cat</div>
    <div>track position of cat last seen and store them in database</div>
    <div>roomba + arm + ipad + realsense to all connect to pytorch x 3 (scene reconstruction, object detection)</div>
    <div>Detect when cat likes roomba </div>
    {/* button for every pick up able object in view */}
    <img src="http://shelbernstein.ngrok.io/current_frame" />
    <div>Robotic Art</div>
    <div>inspired by botparty.org</div>
    <div>inspired by nanosaur</div>
    {/* show continual progress for the rest of your life */}
    {/* fix it by - not doing whippets for remainder of life */}
    {/* all of telepathy will make it a good story - if you quit and stick to that. */}
    {/* get rid of the whippets - and the ywill let you code in peace  */}
  </div>
}




const colorRange = [
  [255, 255, 178, 25],
  [254, 217, 118, 85],
  [254, 178, 76, 127],
  [253, 141, 60, 170],
  [240, 59, 32, 212],
  [189, 0, 38, 255]
];

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

function AirbnbWorldMap(props) {
  
  const [routes, setRoutes] = useState([])
  const [markers, setMarkers] = useState([])
  const [getPercent, setPercent] = useState(0)
  const [openPopover, setOpenPopover] = useState(false)
  const [currentViewState, setViewState] = useState(computeBoundingBox(INITIAL_VIEW_STATE))

  let layers = []

  layers.push( new ScatterplotLayer({
    visible:currentViewState.zoom < 10 && currentViewState.zoom > 5.5,
    id: 'airbnb+houses-within-bbox-dot-matrix',
    data: 'https://raw.githubusercontent.com/adnanwahab/cooperation.party/turkey2/data/all-airbnb.json',
    pickable: true,
    opacity: 1.,
    radiusScale: 1,
    radiusMinPixels: 1,
    radiusMaxPixels: 1,
    lineWidthMinPixels: 1,
    getPosition: d => [d[1], d[0]],
    onHover:() => {},
    onMouseOut: () => {},
    onClick: ({object}) => setOpenPopover(object[2]),
    getRadius: 1,
    getFillColor: d => {
      let rgb = d3.rgb(interpolateRainbow(d[0]))
      return [rgb.r, rgb.g, rgb.b]
    },
  }))

  layers.push( new ScreenGridLayer({
    visible: currentViewState.zoom < 5.5,
    id: 'airbnb+houses-within-bbox-screengrid',
    data: 'https://raw.githubusercontent.com/adnanwahab/cooperation.party/turkey2/data/all-airbnb.json',
    opacity: 1.,
    getPosition: d => [d[1], d[0]],
    getWeight: () => 1,
    cellSizePixels: 30 - 10 * currentViewState.zoom,  //change on zoomLevel
    //colorRange,
    getFillColor: d => {
      let rgb = d3.rgb(interpolateRainbow(d[0]))
      return [rgb.r, rgb.g, rgb.b]
    },
  }))

  //if (false)
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

  let iconLayer = new IconLayer({
      id: 'icon-layer',
      data: markers,
      //data: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/bart-stations.json',
      pickable: true,
      // iconAtlas and iconMapping are required
      // getIcon: return a string
      iconAtlas: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png',
      iconMapping: ICON_MAPPING,
      getIcon: d => 'marker',
      onClick: (_)=> _.object.url,
      sizeScale: 15,
      getPosition: d => {
        return [+ d.lon, + d.lat]

      },
      getSize: d => 5,
      getColor: d => { 
        return [Math.random() * 255, 140, 0]
      }
    })
    //console.log(markers)
  layers.push(iconLayer)

  const allCoordinates = []
  if (currentViewState.zoom > 5)
  routes.forEach((route) => {
    console.log(route[0].routes[0].geometry.coordinates)
    //route[0].routes[0].geometry.coordinates.forEach((list, i ) => allCoordinates.push(list))
    allCoordinates.push(...route[0].routes[0].geometry.coordinates)
  })
  console.log('allcoordinates', allCoordinates, routes.length)
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
      let rgb = d3.rgb(interpolateRainbow(Math.random()))
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
  layers.push(roads)
  
  const fetchRoutes = async () => {
    //everytime you move, draw contiguous paths from center to 4 corners
    let {top, left, right, bottom} = currentViewState;
    //console.log(top,bottom, right, left)
   //console.log(bottom,left, top, right)

    let url = `${baseName}/osm_bbox/?min_lat=${bottom}&min_lng=${left}&max_lat=${top}&max_lng=${right}`;
    const response = await fetch(url);
    const json = await response.json();
    setRoutes(json.routes)
    setMarkers(json.places)
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

  return (<>
    <Robot />
    <h3 className="">World Map! - Scroll to zoom in to see every home in the world at a higher resolution + <button className="text-black">Click here to place a house and generate program instructions for robot to build it.</button></h3>
    <div className="relative" style={{left: `${props.left}px`, height: '600px'}}>
    <PopOver open={openPopover} setOpen={setOpenPopover}/>
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
      onWebGLInitialized={onInitialized}
      onViewStateChange={_.debounce(({viewState}) => {
        setViewState(computeBoundingBox(viewState))
      }, 1000)}
    >
      <Map 
       glOptions={{preserveDrawingBuffer: true}}
      reuseMaps mapLib={maplibregl} mapStyle={MAP_STYLE} preventStyleDiffing={true}>
      </Map>
    </DeckGL>

    </div>
    <div className="grid grid-cols-2">
      <div className="">
        <div >
          <div>Latitude: {currentViewState.bottom.toPrecision(4)}  |   {currentViewState.top.toPrecision(4)}</div>
          <div>Longitude: {currentViewState.left.toPrecision(4)} |  {currentViewState.right.toPrecision(4)}</div>
          <div>Total Roads Drawn: {routes.length} / 1 billion</div>
          <div>Places Of Interest Drawn: {markers.length}</div>
          <div>Airbnbs rendered on Screen: {routes.length} /  7 million</div>
          <div>Benches rendered on Screen: {markers.length} /  7 million</div>
          <div>Houses rendered on Screen:  / 135 million from  
            <span className="text-blue-500"> Zillow</span> 
            <span className="text-red-500">Redfin</span> 
            <span className="text-green-500">Compass</span> 
            <span className="text-yellow-500 hidden">immobilienscout24.de</span>
          </div>
          </div>
        </div>
        <span className="flex">
          <DonutChart random={Math.random()}/>
          <Histogram random={Math.random()} />
        </span>
    </div>
    </>
  );
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
    centroid: [viewport.latitude, viewport.longitude],
    zoom: viewPort.zoom
  };
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

function getTooltip(params) {
    if (! params.picked) return 
    return JSON.stringify(params.object, null, 2)
}

function ColorLabels() {
    return <></>
}
function clamp(_, min, max) {
  return Math.max(Math.min(_, max), min)
}
function ProgressBar (props) {
  let style = {width: `${innerWidth}px`}
  return (<div className="fixed left-0 top-0 progress-bar h-4 w-full" style={style}></div>)
}

function onInitialized (gl) {
  if (!isWebGL2(gl)) {
    console.warn('GPU aggregation is not supported'); // eslint-disable-line
    if (disableGPUAggregation) {
      disableGPUAggregation();
    }
  }
};
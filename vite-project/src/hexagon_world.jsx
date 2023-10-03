import React from 'react';
import {useState, useMemo, useCallback} from 'react';
import {Map} from 'react-map-gl';
import maplibregl from 'maplibre-gl';
import {createRoot} from 'react-dom/client';
import { ScatterplotLayer } from '@deck.gl/layers';
import {H3ClusterLayer} from '@deck.gl/geo-layers';

import DeckGL from '@deck.gl/react';
import {
  COORDINATE_SYSTEM,
  _GlobeView as GlobeView,
  LightingEffect,
  AmbientLight,
  _SunLight as SunLight
} from '@deck.gl/core';
import {GeoJsonLayer} from '@deck.gl/layers';
import {SimpleMeshLayer} from '@deck.gl/mesh-layers';

import {SphereGeometry} from '@luma.gl/core';
import {load} from '@loaders.gl/core';
import {CSVLoader} from '@loaders.gl/csv';

import {H3HexagonLayer} from '@deck.gl/geo-layers';
import {latLngToCell} from "h3-js";
//const h3 = require("h3-js");

import {HexagonLayer} from '@deck.gl/aggregation-layers';
import * as d3 from 'd3'

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json';


const resolution = 7;

// Create an empty array to store H3 cell data
const h3Data = [];
// Generate random points within each H3 cell
const numPointsPerCell = 1000;
function generateRandomData(numPoints) {
  const data = [];
  for (let i = 0; i < numPoints; i++) {
    data.push({
      position: [
        (Math.random() - 0.5) * 360, // Random longitude between -180 and 180
        (Math.random() - 0.5) * 180, // Random latitude between -90 and 90
      ],
      size: Math.random() * 100, // Random size for each point
      color: [Math.random() * 255, Math.random() * 255, Math.random() * 255], // Random color
    });
  }
  return data;
}


for (let i = 0; i < numPointsPerCell; i++) {
    // Generate a random H3 index

    // Convert H3 index to a latitude and longitude point
    const cell = latLngToCell(Math.random() * 90, Math.random() *  180, 7);
  //  console.log(cell)
    // Store the H3 index and point in the data array
    h3Data.push({ h3Index: cell});
}


const DATA_URL = 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/globe';

const INITIAL_VIEW_STATE = {
  longitude: 139,
  latitude: 35,
  zoom: 11,
};
//let benches = await d3.json('https://pypypy.ngrok.io/data/airbnb/h3_poi/bench.json')
//console.log(benches)
const benches = []
const TIME_WINDOW = 900; // 15 minutes
const EARTH_RADIUS_METERS = 6.3e6;
const SEC_PER_DAY = 60 * 60 * 24;

const ambientLight = new AmbientLight({
  color: [255, 255, 255],
  intensity: 0.9
});
const sunLight = new SunLight({
  color: [255, 255, 255],
  intensity: 1.1,
  timestamp: 0
});
// create lighting effect with light sources
const lightingEffect = new LightingEffect({ambientLight, sunLight});

/* eslint-disable react/no-deprecated */
export default function App(props) {
  let data = props.data
  console.log(props)
  const [currentTime, setCurrentTime] = useState(0);

  const timeRange = [currentTime, currentTime + TIME_WINDOW];

  const formatLabel = useCallback(t => getDate(data, t).toUTCString(), [data]);

  if (data) {
    sunLight.timestamp = Date.now()
  }

  const backgroundLayers = useMemo(
    () => [
      new SimpleMeshLayer({
        id: 'earth-sphere',
        data: [0],
        mesh: new SphereGeometry({radius: EARTH_RADIUS_METERS, nlat: 18, nlong: 36}),
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        getPosition: [0, 0, 0],
        getColor: [255, 255, 255]
      }),
      new GeoJsonLayer({
        id: 'earth-land',
        data: 'https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_land.geojson',
        // Styles
        stroked: false,
        filled: true,
        opacity: 0.1,
        getFillColor: [30, 80, 120]
      })
    ],
    []
  );


  const _data = [...Array(1e6).keys()].map(_ => [Math.random() * 90, Math.random() * 180])
  .map(_ => latLngToCell(_[0], _[1], 4))
  
    console.log(_data)

  const hexagon = new H3ClusterLayer({
    id: 'h3',
    //getFillColor: _ => Object.values(d3.rgb(d3.interpolatePurples(_[1].vending_machine / 100))).slice(0, 3),
    getFillColor: [255, 0, 255, 0],
    data: _data,
    elevationRange: [0, 0],
    elevationScale: 1,
    getHexagon: d => d[0],
    pickable: true,
    radius: 10000,
    onClick: (_) => console.log(_),
    //upperPercentile,
    //material,
    //opacity:.9,
    extruded: false,
  //   transitions: {
  //     elevationScale: 3000
  //   }
  });

  let hexagonList = Object.entries(data).map((pair) => {
    console.log(pair)
    const hexagon2 = new H3HexagonLayer({
      id: pair[0],
      //getFillColor: _ => Object.values(d3.rgb(d3.interpolatePurples(_[1].vending_machine / 100))).slice(0, 3),
      getFillColor: [255, 0, 255, 0],
      data: Object.entries(pair[1]),
      elevationRange: [0, 20],
      elevationScale: 20,
      getHexagon: d => d[0],
      pickable: true,
      radius: 10000,
      onClick: (_) => console.log(_),
      //upperPercentile,
      //material,
      //opacity:.9,
      extruded: false,
    //   transitions: {
    //     elevationScale: 3000
    //   }
    });
    return hexagon2
  })

  const layer2 = new ScatterplotLayer({
    id: 'scatterplot-layer',
    data: generateRandomData(1000),
    getPosition: (d) => d.position,
    getRadius: (d) => d.size,
    getColor: (d) => d.color,
    radiusScale: 10,
    radiusMinPixels: 20,
    radiusMaxPixels: 200,
  });

  // const dataLayers =
  //   data &&
  //   data.map(
  //     ({date, flights}) =>
  //       new AnimatedArcLayer({
  //         id: `flights-${date}`,
  //         data: flights,
  //         getSourcePosition: d => [d.lon1, d.lat1, d.alt1],
  //         getTargetPosition: d => [d.lon2, d.lat2, d.alt2],
  //         getSourceTimestamp: d => d.time1,
  //         getTargetTimestamp: d => d.time2,
  //         getHeight: 0.5,
  //         getWidth: 1,
  //         timeRange,
  //         getSourceColor: [255, 0, 128],
  //         getTargetColor: [0, 128, 255]
  //       })
  //   );

  const attemptLayer = new HexagonLayer({
    id: 'hexagon-layer',
    data: [...Array(1e6).keys()].map(_=> {return {COORDINATES: [Math.random() * 180, Math.random() * 90]}}),
    pickable: true,
    extruded: true,
    radius: 10000,
    elevationScale: 4,
    opacity:0,
    getPosition: d => d.COORDINATES
  });

  return (
    <> 
    <div class="relative w-full h-full" style={{height: '750px'}}>
      <DeckGL
        // views={new GlobeView()}
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        // effects={[lightingEffect]}
        layers={[
        backgroundLayers, 
        hexagon, 
        layer2, 
        attemptLayer, 
        ...hexagonList
        ]}
      >
           <Map reuseMaps mapLib={maplibregl} mapStyle={MAP_STYLE} preventStyleDiffing={true}></Map>
      </DeckGL>
      </div>
    </>
);

}
import React from 'react';
import {createRoot} from 'react-dom/client';
import {Map} from 'react-map-gl';
import maplibregl from 'maplibre-gl';
import {AmbientLight, PointLight, LightingEffect} from '@deck.gl/core';
import {HexagonLayer} from '@deck.gl/aggregation-layers';
import DeckGL from '@deck.gl/react';

import * as d3 from 'd3';
import {H3HexagonLayer} from '@deck.gl/geo-layers';

// Source data CSV
const DATA_URL =
  'https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/3d-heatmap/heatmap-data.csv'; // eslint-disable-line

const ambientLight = new AmbientLight({
  color: [255, 255, 255],
  intensity: 1.0
});

const pointLight1 = new PointLight({
  color: [255, 255, 255],
  intensity: 0.8,
  position: [-0.144528, 49.739968, 80000]
});

const pointLight2 = new PointLight({
  color: [255, 255, 255],
  intensity: 0.8,
  position: [-3.807751, 54.104682, 8000]
});

const lightingEffect = new LightingEffect({ambientLight, pointLight1, pointLight2});

const material = {
  ambient: 0.64,
  diffuse: 0.6,
  shininess: 32,
  specularColor: [51, 51, 51]
};

const INITIAL_VIEW_STATE = {
    longitude: 139.6503,
    latitude: 35.6762,
    zoom: 11,
    maxZoom: 20,
    pitch: 30,
    bearing: 0
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

function getTooltip({object}) {
//   if (!object) {
//     return null;
//   }
//   const lat = object.position[1];
//   const lng = object.position[0];
//   const count = object.points.length;

//   return `\
//     latitude: ${Number.isFinite(lat) ? lat.toFixed(6) : ''}
//     longitude: ${Number.isFinite(lng) ? lng.toFixed(6) : ''}
//     ${count} Accidents`;
}
const getCsv = await d3.csv(DATA_URL, (data) => {
    return [Number(data.lng), Number(data.lat)]
})

console.log(getCsv)
//(error, response) => { return response.map(d => ) }
/* eslint-disable react/no-deprecated */

let benches = await d3.json('https://pypypy.ngrok.io/data/airbnb/h3_poi/bench.json')
console.log(benches)


export default function App({

  mapStyle = MAP_STYLE,
  radius = 1000,
  upperPercentile = 100,
  coverage = 1
}) {
    const data = getCsv
  const layers = [
    new H3HexagonLayer({
      id: 'heatmap',
      colorRange,
      coverage,
      getFillColor: d => [255, (1 - d.count / 500) * 255, 0],
      data: Object.entries(benches),
      elevationRange: [0, 200],
      elevationScale: 1,
      extruded: true,
      getHexagon: d => d[0],
      pickable: true,
      radius,
      upperPercentile,
      material,
    //   transitions: {
    //     elevationScale: 3000
    //   }
    })
  ];

  return (
<div>
    <DeckGL
    width={500}
    height={500}

      layers={layers}
      effects={[lightingEffect]}
      initialViewState={INITIAL_VIEW_STATE}
      controller={true}
      getTooltip={getTooltip}
      
    >
      <Map reuseMaps mapLib={maplibregl} mapStyle={mapStyle} preventStyleDiffing={true} />
    </DeckGL>
    </div>
  );
}

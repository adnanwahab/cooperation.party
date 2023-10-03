import React from 'react';
import {useState, useMemo, useCallback} from 'react';

import {createRoot} from 'react-dom/client';
import { ScatterplotLayer } from '@deck.gl/layers';

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
  longitude: 0,
  latitude: 20,
  zoom: 0
};

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
export default function App({data}) {
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

  const hexagon = new H3HexagonLayer({
    id: 'h3',
    //getFillColor: _ => Object.values(d3.rgb(d3.interpolatePurples(_[1].vending_machine / 100))).slice(0, 3),
    getFillColor: [255, 255, 255, 0],
    data: h3Data,
    elevationRange: [0, 0],
    elevationScale: 1,
    getHexagon: d => console.log('_' + d.h3Index) || d.h3Index,
    pickable: true,
    radius: 100000,
    //onClick: (_) => console.log(_),
    //upperPercentile,
    //material,
    //opacity:.9,
    extruded: false,
  //   transitions: {
  //     elevationScale: 3000
  //   }
  });

  

  const layer2 = new ScatterplotLayer({
    id: 'scatterplot-layer',
    data: generateRandomData(1000),
    getPosition: (d) => d.position,
    getRadius: (d) => d.size,
    getColor: (d) => d.color,
    radiusScale: 1,
    radiusMinPixels: 2,
    radiusMaxPixels: 20,
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

  return (
    <> 
      <DeckGL
        views={new GlobeView()}
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        effects={[lightingEffect]}
        layers={[backgroundLayers, hexagon, layer2]}
      />
    </>
  );
}

function getDate(data, t) {
  const index = Math.min(data.length - 1, Math.floor(t / SEC_PER_DAY));
  const date = data[index].date;
  const timestamp = new Date(`${date}T00:00:00Z`).getTime() + (t % SEC_PER_DAY) * 1000;
  return new Date(timestamp);
}

export function renderToDOM(container) {
  const root = createRoot(container);
  root.render(<App />);

  async function loadData(dates) {
    const data = [];
    for (const date of dates) {
      const url = `${DATA_URL}/${date}.csv`;
      const flights = await load(url, CSVLoader, {csv: {skipEmptyLines: true}});

      // Join flight data from multiple dates into one continuous animation
      const offset = SEC_PER_DAY * data.length;
      for (const f of flights) {
        f.time1 += offset;
        f.time2 += offset;
      }
      data.push({flights, date});
      root.render(<App data={data} />);
    }
  }

  loadData([
    '2020-01-14',
    '2020-02-11',
    '2020-03-10',
    '2020-04-14',
    '2020-05-12',
    '2020-06-09',
    '2020-07-14',
    '2020-08-11',
    '2020-09-08',
    '2020-10-13',
    '2020-11-10',
    '2020-12-08'
  ]);
}
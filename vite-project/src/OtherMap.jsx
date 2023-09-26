import {useRef, useMemo, useState, useEffect} from 'react';
import {createRoot} from 'react-dom/client';
import ReactMap, {Source, Layer, Marker} from 'react-map-gl';
import DeckGL, {GeoJsonLayer, ArcLayer} from 'deck.gl';
import HexagonMap from './HexagonMap' //commute and so on

function OtherMap(props) {
  const mapRef = useRef();
  return (<div class="relative h-96">
      <HexagonMap />
    </div>);
}
export default OtherMap

//
import {useRef, useMemo, useState, useEffect} from 'react';
import React from 'react';
import {Runtime, Inspector} from "@observablehq/runtime";
import define2 from "https://api.observablehq.com/d/84ce55045edfd14f.js?v=3";

import * as d3 from 'd3'
// import define from "https://api.observablehq.com/@jashkenas/breakout.js?v=3";
// let module = new Runtime().module(define, name => {
//   if (name === "score") return new Inspector(document.querySelector("#observablehq-score-0d14fa84"));

//   return ["highscore","gameloop","draw"].includes(name);
// })
//window.module = module

export default function BarChart(props) {
  if (! props.data) return <>no data</>
  else console.log('BAR CHART COOL BEANS :-p', props)
  const cb = props.cb || function () {}
  let data = props.data
    // data = data.map(_ => {
    //   console.log('_', _)
    //   return {
    //     letter: _[0],
    //     frequency: _[1]
    //   }
    // })
  const chartRef = useRef();
  useEffect(() => {
    const runtime = new Runtime();
   let module = runtime.module(define2, name => {
      if (name === "chart") return new Inspector(chartRef.current);
      
    })
    
    module.redefine('alphabet', data)
    //Observer.
    module.value('selected').then((_) => {
      console.log('selected', _)
    })
    
  //   module.redefine('callback', function (_) {
  //     console.log(_)
  //     if (props.callback) props.callback(_)
  //   })
  setInterval(function () {
    if (! props.setFormData) return
    window.d3 = d3
    d3.selectAll('.bar').on('mouseover.foo', function (e) {
      console.log('selected', e.target.__data__)
      cb(e.target.__data__.letter)
      props.setFormData('city', e.target.__data__.letter)
      props.apply_()
    }, 20000)
  })
    return () => runtime.dispose();
  }, [props.schedule, props.city]);

  return (
    <>
      <div ref={chartRef} />
    </>
  );
}
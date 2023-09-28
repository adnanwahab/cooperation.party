import {useRef, useMemo, useState, useEffect} from 'react';
import React from 'react';
import {Runtime, Inspector} from "@observablehq/runtime";
import define2 from "https://api.observablehq.com/d/84ce55045edfd14f.js?v=3";

export default function BarChart(props) {
  const chartRef = useRef();
  useEffect(() => {
    const runtime = new Runtime();
    runtime.module(define2, name => {
      if (name === "chart") return new Inspector(chartRef.current);
    }).redefine('alphabet', props.data)
    return () => runtime.dispose();
  }, []);

  return (
    <>
      <div ref={chartRef} />
    </>
  );
}
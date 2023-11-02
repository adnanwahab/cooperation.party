import React, { useEffect, useState, useCallback, useRef} from 'react';
import * as d3 from 'd3';

export default function histogram (props) {
    const width = 300;
    const height = 150;
    const svgRef = useRef(null); // Reference to the SVG element
    let totalCount = 1;

    useEffect(() => {
      if (!svgRef.current) return;
  
      const svg = d3.select(svgRef.current);
      const data = d3.map(Array(1000), (d, i) => ({
        pos: i,
        value: d3.randomNormal(1, 10)()
      }));
      const margin = { top: 0, right: 0, bottom: 30, left: 0 };

      const svg_width = width + margin.left + margin.right;
      const svg_height = height + margin.top + margin.bottom;
      const xAxisText = "Price of Housing (%)";
    
      svg
        .attr("width", svg_width)
        .attr("height", svg_height)
        .attr("viewBox", [
          0,
          0,
          width + margin.left + margin.right,
          height + margin.top + margin.bottom
        ])
        .attr("style", "max-width: 100%; height: auto;");
    
      const chart = svg
        .append("g")
        .attr('class', 'histogram')
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
    
      const prob = false;
      let bins = d3
        .bin()
        .thresholds(40)
        .value((d) => d.value)(data);
    
      let x = d3
        .scaleLinear()
        .domain([bins[0].x0, bins[bins.length - 1].x1])
        .range([0, width]);
    
      chart
        .append("g")
        .attr("transform", `translate(0, ${height})`)
        .call(
          d3
            .axisBottom(x)
            .ticks(width / 80)
            .tickSizeOuter(0)
        )
        .call((g) =>
          g
            .append("text")
            .attr("x", width)
            .attr("y", 25)
            .attr("fill", "currentColor")
            .attr("text-anchor", "end")
            .text(`${xAxisText.length == 0 ? "Rate" : xAxisText} →`)
        );
    
      chart
        .append("g")
        .attr("transform", `translate(0, ${height})`)
        .call(
          d3
            .axisBottom(x)
            .ticks(width / 80)
            .tickSizeOuter(0)
        );
    
      if (prob) {
        totalCount = data.length;
      }
    
      let y = d3
        .scaleLinear()
        .domain([0, d3.max(bins, (d) => d.length / totalCount)])
        .range([height, 0]);
    
      const yAxisAnnotation = prob ? "Prob." : "Freq.";
      const yAxisSupplementText = "(no. of counties)";
      chart
        .append("g")
        .call(d3.axisLeft(y).ticks(height / 40))
        .call((g) => g.select(".domain").remove())
        .call((g) =>
          g
            .append("text")
            .attr("x", -20)
            .attr("y", -30)
            .attr("fill", "currentColor")
            .attr("text-anchor", "start")
            .text(`↑ ${yAxisAnnotation} ${yAxisSupplementText}`)
        );
    
      chart
        .append("g")
        .attr("fill", "steelblue")
        .selectAll()
        .data(bins)
        .join("rect")
        .attr("x", (d) => x(d.x0) + 1)
        .attr("width", (d) => x(d.x1) - x(d.x0) - 1)
        .attr("y", (d) => y(d.length / totalCount))
        .attr("height", (d) => y(0) - y(d.length / totalCount));
    
  
    }, []);

    useEffect(() => {
        const data = d3.map(Array(1000), (d, i) => ({
            pos: i,
            value: d3.randomNormal(1, 10)()
          }));
        let bins = d3
        .bin()
        .thresholds(40)
        .value((d) => d.value)(data);
        const svg = d3.select(svgRef.current);

        const chart = svg.select('.histogram')
        let y = d3
        .scaleLinear()
        .domain([0, d3.max(bins, (d) => d.length / totalCount)])
        .range([height, 0]);

            
      let x = d3
      .scaleLinear()
      .domain([bins[0].x0, bins[bins.length - 1].x1])
      .range([0, width]);
 
        chart
        .selectAll('rect')
        .data(bins)
        .attr("x", (d) => x(d.x0) + 1)
        .attr("width", (d) => x(d.x1) - x(d.x0) - 1)
        .attr("y", (d) => y(d.length / totalCount))
        .attr("height", (d) => y(0) - y(d.length / totalCount));

    }, [props.random])

      return (
        <svg 
          ref={svgRef}
          width={width} 
          height={height} 
          //viewBox={[-width / 2, -height / 2, width, height]}
          style={{ maxWidth: "100%", height: "auto" }}
        />
      );
}
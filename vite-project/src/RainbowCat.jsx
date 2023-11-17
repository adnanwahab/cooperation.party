import React, {useRef, useEffect} from "react";
import {Runtime, Inspector} from "@observablehq/runtime";
import notebook from "@kerryrodden/rainbow-cat";

function RainbowCat() {
  const rainbowCatRef = useRef();

  useEffect(() => {
    const runtime = new Runtime();
    runtime.module(notebook, name => {
      if (name === "rainbowCat") return new Inspector(rainbowCatRef.current);
    });
    return () => runtime.dispose();
  }, []);

  return (
    <>
      <div ref={rainbowCatRef} />
      <p>Credit: <a href="https://observablehq.com/@kerryrodden/rainbow-cat">Rainbow Cat by Kerry Rodden</a></p>
    </>
  );
}

export default RainbowCat;
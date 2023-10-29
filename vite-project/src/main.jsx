import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import Map from './Map.jsx'

// const _save = HTMLCanvasElement.prototype.getContext
// HTMLCanvasElement.prototype.getContext = function (_, opts) {
//     console.log('this is a pseudo context', _, opts)
//   const gl = canvas.getContext(_, {
//       preserveDrawingBuffer: true
//   });
//       return _save(_, opts)
//   }

ReactDOM.createRoot(document.getElementById('root')).render(
  <App />
)
/* global window */

// export function renderToDom(container) {
//   createRoot(container).render(<App />);
// }

//renderToDom(document.querySelector('#map'))
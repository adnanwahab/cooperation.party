import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import Map from './Map.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <App />
)
/* global window */

// export function renderToDom(container) {
//   createRoot(container).render(<App />);
// }

//renderToDom(document.querySelector('#map'))
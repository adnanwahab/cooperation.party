function Slider (props) {
  let _ = {}
  _[props.label] = .5
  return <div className="block">
  <label>{props.label}</label>
  <input type="range" onChange={(e) => {
      _[props.label] = e.target.value / 100;
      setFormData('sliders', Object.assign(getFormData('sliders'), _))
      props.apply_()
      }
  } />
  </div>
}

function ProgressBar(props) {
    //put a clock if this computation is taking more than .3 seconds
    return (
    <div class="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
    <div class="bg-blue-600 h-2.5 rounded-full" style={{"width": props.progress + "%"}}></div>
  </div>
  )
}


function SentenceDiagramPresenter() {
    //this component renders a component from a hashmap 
    //does the data-fetching 
    //updates the component once the live query updates
    //streams the response
    //uses concurrent to suspend and re-animate 
    //every component in the document has one of these
    //because lets say each component fetches 200mb of data

    
    
}
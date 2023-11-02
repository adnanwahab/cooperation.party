let makeBaseName = {
    'cooperation.party':'https://cooperation-party.fly.dev',
    'localhost':'http://localhost:8000',
    'cooperation-party.fly.dev': 'https://cooperation-party.fly.dev/'
}

// if (window.location.hostname == 'cooperation.party' || ) {
//   url = 'http://cooperation-party.fly.dev'

// } else if (window.location.hostname === 'localhost') {
//     url = 'https://shelbernstein.ngrok.io'
// }


let baseName = makeBaseName[window.location.hostname]
export default baseName
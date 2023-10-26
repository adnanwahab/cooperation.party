//use Fly.IO + websockets to stream data to client

const fs = require('fs')
const puppeteer = require( 'puppeteer' )
function delay(time) {
  return new Promise(function (resolve) {
    setTimeout(resolve, time)
  })
}

const express = require('express')
const app = express()
const port = 3001

// app.post('/rpc', async (req, res) => {
//     let data = await getHTML(req.body.channelName)
//     res.json(JSON.stringify(data))
// })

// app.listen(port, () => {
//   console.log(`Example app listening on port ${port}`)
// })




async function getHTML(){
  let channelName = process.argv[2].trim()
  console.log('hi', channelName)

  const browser = await puppeteer.launch({
    pipe:true
  })

  const page = await browser.newPage()
  await page.setViewport({
    width: 1920,
    height: 1080,
  })

  await page.goto(`https://www.twitch.tv/${channelName}`, {
    waitUntil: "networkidle2",
    timeout: 0,
  })

  await delay(1000)
  await page.waitForFunction( function () {
    const el = document.querySelector( 'div[data-a-target="chat-welcome-message"].chat-line__status' )
  return !!el
  }, { poll: 250 } )

  const sel = '.chat-line__message, .chat-line__status, div[data-a-target="chat-line-message"]'
  await page.waitForSelector(sel) 
//  const text = await page.$$eval('*', els => els.map(e => e.textContent)) 
  const text = await page.$$eval(sel, els => els.map(e => e.textContent)) 
  await browser.close()
  console.log('texts', text)
  // let json = JSON.parse(fs.readFileSync('twitch.json'))
  // console.log(json.length)
  // text.forEach(t => json.push(t))
  // console.log(json.length)

  fs.writeFileSync(`twitch-${channelName}.json`, JSON.stringify(text))
  return text
}




exports.handler = async (event) => {
 
  const text = getHTML()
  return {
    statusCode: 200,
    body: JSON.stringify({text,
      'asdfasdfasdf': Math.random()
    }),
    // // more keys you can return:
    // headers: { "headerName": "headerValue", ... },
    // isBase64Encoded: true,
  }
}

getHTML()

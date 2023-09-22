//f(location) -> list of appartments in that city

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path')
async function getApt(url, location) {
    //location = location.replace(/\w|\,/g, '-')
    console.log(location, 'GET apt')
    const browser = await puppeteer.launch({ headless: true });  // Change to false if you want to view the browser
    const page = await browser.newPage();
    const fp = path.resolve(`data/airbnb/apt/${location}.json`);

    await page.goto(url);
    const qs = '.cy5jw6o.dir.dir-ltr a';
  
    let tweets = [];
    function urlToFileName(url) {
        return url
        return 'https://www.airbnb.com/rooms/' + url.match(/rooms\/(\d+)/)[1]
      }


    try {
        await page.waitForSelector(qs);
  
        tweets = await page.$$eval(qs, (tweetNodes) => {
            return tweetNodes.map(tweet => (tweet.href) );
        })
  
    } catch (error) {
        console.error('Error:', error);
        return [];
    }
  
    await browser.close();

    console.log(`writing ${location}.json`)
    console.log(tweets)
    tweets = Array.from(new Set(tweets.map(urlToFileName)))
    await fs.writeFile(fp, JSON.stringify(tweets, null, 2));
  
    console.log(location, 'GET apt')

    return tweets;
}

// Example usage
let location = process.argv[2]
location = location.replace(', ','--')
const url = `https://www.airbnb.com/s/${location}/homes`

getApt(url, location)
    // .then(tweets => console.log(tweets))
    // .catch(err => console.error('An error occurred:', err));

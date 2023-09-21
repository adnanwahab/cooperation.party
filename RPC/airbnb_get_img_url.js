//f(apt) -> list of googlemap
const puppeteer = require('puppeteer');
const fs = require('fs');

let cache = {};
async function getImgUrl(aptListing, idx) {
    let origAptListing = aptListing;
    function urlToFileName(url) {
        return url.match(/rooms\/(\d+)/)[1]
      }
      aptListing = urlToFileName(aptListing);

    const url = `data/airbnb/gm/${aptListing}.json`
    //console.log(aptListing, idx);
    //if (aptListing in cache) {return}
    cache[aptListing] = true;
    console.log(fs.existsSync(url), url);
    if (fs.existsSync(url)) return

    const browser = await puppeteer.launch({ headless: true }); // Replace true with false if you want to see the browser
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    await page.goto(origAptListing);

    await page.waitForSelector('section');

    await page.evaluate(() => {
        window.scrollBy(0, 100000);
    });

    await delay(3000);

    await page.evaluate(() => {
        // Additional JavaScript code you may need to execute on the page.
    });

    await delay(1000);
    
    const imgUrls = await page.$$eval('.gm-style img', imgs => imgs.map(img => img.src).filter(src => src.indexOf("maps.googleapis.com") !== -1));

    console.log('saving to ' + url);
    fs.writeFileSync(url, JSON.stringify(imgUrls));

    await browser.close();

    return imgUrls;
}

// Helper function to delay the script execution for `ms` milliseconds
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

(async () => {
    const urls = JSON.parse(fs.readFileSync(process.argv[2]))
    console.log(urls, process.argv[2])
    for (let idx = 0; idx < urls.length; idx++) {
        const url = urls[idx];
        const task = getImgUrl(url['link'], idx); // Replace 'link' with the appropriate key for your URLs
        await task; // If you wish to run these tasks concurrently, you can collect them in an array and use `Promise.all()`
    }
})();
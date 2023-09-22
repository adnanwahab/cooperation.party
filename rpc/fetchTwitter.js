const fs = require('fs');
const puppeteer = require( 'puppeteer' );
function delay(time) {
  return new Promise(function (resolve) {
    setTimeout(resolve, time)
  })
}
const express = require('express')
const app = express()
const port = 3001
module.exports = async () => 
(async () => {
  // Launch a headless browser
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  // Navigate to Twitter's search page for "pizza"
  await page.goto('https://twitter.com/search?q=pizza%20until%3A2023-09-06%20since%3A2023-09-05&src=typed_query');
  console.log(await page.content())
  // Wait for the tweets to load
  await page.waitForSelector('article');
  // Extract tweet information
  const tweets = await page.evaluate(() => {
    const tweetNodes = document.querySelectorAll('article');
    console.log(tweetNodes)
    const data = [];
    tweetNodes.forEach((tweet) => {
      try {
        const tweetText = tweet.querySelector('div[lang]').innerText;
        const tweetAuthor = tweet.querySelector('div span span').innerText;
        data.push({ tweetText, tweetAuthor });
      } catch (e) {
        console.error(e);
      }
    });
    return data;
  });
  // Output the scraped tweets
  await browser.close();
  return tweets;
})();

module.exports()
// use an encoding
//convert sentence to a matrix 538
//match with previously made code -> hand written or custom -> periodically look at custon generated code and clean-up improve and send back to algorithm

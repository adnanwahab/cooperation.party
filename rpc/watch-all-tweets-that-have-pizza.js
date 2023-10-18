const puppeteer = require('puppeteer');

(async () => {
  // Launch a headless browser
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Navigate to Twitter's search page for "pizza"
  await page.goto('https://twitter.com/search?q=pizza%20until%3A2023-09-06%20since%3A2023-09-05&src=typed_query');

  // Wait for the tweets to load
  await page.waitForSelector('article');

  // Extract tweet information
  const tweets = await page.evaluate(() => {
    const tweetNodes = document.querySelectorAll('article');
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
  console.log(tweets);

  await browser.close();
  return tweets
})();

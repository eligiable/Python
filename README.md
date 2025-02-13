# Python
Python Scripts to Automate

### AWS Web Scrapping

> Python script for web scraping specific data from amazon.com.
> 
> Disclaimer: Always review Amazon's Terms of Service before scraping. Unauthorized scraping may violate their policies.
>
> **How it Works:**
>
> 1. **Headers**: The User-Agent header mimics a browser request and avoids being blocked.
> 2. **URL Construction**: The script constructs a URL for the search query.
> 3. **Request**: It sends a GET request to the Amazon search results page.
> 4. **Parsing**: The HTML content is parsed using BeautifulSoup.
> 5. **Data Extraction**: The script extracts product titles and prices from the search results.
> 6. **Output**: The extracted data is printed to the console.
>
> **Limitations**:
>
> 1. **Dynamic Content**: Amazon uses JavaScript to load content dynamically. This script may not work for pages with heavy JavaScript usage.
> 2. **Anti-Scraping Measures**: Amazon may block your IP if it detects scraping activity.
> 3. **CAPTCHA**: You may encounter CAPTCHA challenges, which require manual intervention.
>
> **Advanced Options**:
>
> 1. **Use Proxies**: Rotate IP addresses to avoid bans.
> 2. **Selenium**: Use Selenium to handle JavaScript-rendered content.
> 3. **API**: Consider using Amazon's official API (e.g., Amazon Product Advertising API) for reliable data access.

### Social Media Posting Auto BOT

> **How to Run:**
>
> * Save the code to a file, e.g., social_media_bot.py.
> * Create a **config.json** file with your API keys and endpoints.
```
{
    "moderator_endpoint": "your_endpoint",
    "moderator_key": "your_key"
}
```
> **Run the Script:**
```
python social_media_bot.py
```
> **Run Unit Test:**
```
python -m unittest social_media_bot.py
```

#Install the required libraries:
#pip install requests beautifulsoup4 lxml pandas openpyxl

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define headers to mimic a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# Function to scrape Amazon search results
def scrape_amazon(search_query):
    # Construct the URL for the search query
    url = f"https://www.amazon.com/s?k={search_query.replace(' ', '+')}"
    
    # Send a GET request to the URL
    response = requests.get(url, headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "lxml")
    
    # Find all product containers
    products = soup.find_all("div", {"data-component-type": "s-search-result"})
    
    # Lists to store scraped data
    titles = []
    prices = []
    
    # Loop through each product and extract details
    for product in products:
        # Extract product title
        title = product.find("span", class_="a-size-medium")
        title = title.text.strip() if title else "N/A"
        
        # Extract product price
        price = product.find("span", class_="a-price-whole")
        price = price.text.strip() if price else "N/A"
        
        # Append data to lists
        titles.append(title)
        prices.append(price)
    
    # Create a DataFrame to store the data
    data = {
        "Title": titles,
        "Price ($)": prices,
    }
    df = pd.DataFrame(data)
    
    # Save the DataFrame to an Excel file
    output_file = "amazon_products.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

# Main function
if __name__ == "__main__":
    search_query = input("Enter the product to search on Amazon: ")
    scrape_amazon(search_query)

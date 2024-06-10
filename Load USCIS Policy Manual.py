# Databricks notebook source
# DBTITLE 1,imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import requests
import re
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pyspark.sql.types import StringType
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin

# COMMAND ----------

# DBTITLE 1,functions

#Add retries with backoff to avoid 429 while fetching the doc
retries = Retry(
    total=3,
    backoff_factor=3,
    status_forcelist=[429],
)

def extract_page_urls_from_url(url, parent_class):
    page_urls = []
    
    # Fetch the HTML content of the page
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all links within elements with specified parent class
        parent_items = soup.find_all('div', class_=parent_class)
        for parent_item in parent_items:
            for link in parent_item.find_all('a', href=True):
                relative_url = link['href']
                absolute_url = urljoin(url, relative_url)
                page_urls.append(absolute_url)
    
    return page_urls

def get_policy_urls(): 
    base_url = "https://www.uscis.gov/policy-manual/"
    url = "https://www.uscis.gov/policy-manual/table-of-contents"  # Specify the full URL of the page
    parent_class = "toc-tree"
    page_urls = extract_page_urls_from_url(url, parent_class)

    # Make sure to join base_url with each page URL if they are relative
    fully_formed_urls = [urljoin(base_url, page_url) for page_url in page_urls]

    return fully_formed_urls

def fetch_content_from_urls(urls):
    page_contents = []
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract page title
            page_title = soup.find(class_='page-title').get_text() if soup.find(class_='page-title') else None
            
            # Find the <section> element with id="book-content"
            content_section = soup.find('section', id='book-content')
            if content_section:
                # Extract the text content of the section
                content = content_section.get_text(separator='\n')
                
                # Remove extraneous newline characters
                content = re.sub(r'\n{2,}', '\n', content)
                
                page_contents.append({
                    'url': url,
                    'page_title': page_title,
                    'content': content.strip()  # Strip leading/trailing whitespace
                })
            else:
                page_contents.append({
                    'url': url,
                    'page_title': page_title,
                    'content': None
                })
    
    return page_contents

# COMMAND ----------

spark = SparkSession.builder \
    .appName("PageContents") \
    .getOrCreate()

urls = get_policy_urls()

first_50_urls = urls[:50]  # Get the first 100 URLs
page_contents = fetch_content_from_urls(first_50_urls)
page_contents_df = pd.DataFrame(page_contents)

# Convert the pandas DataFrame to a Spark DataFrame
df = spark.createDataFrame(page_contents_df)

df.write.format("delta").mode("overwrite").saveAsTable("workspace.default.ic_documentation_2")


# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM workspace.default.ic_documentation_2
# MAGIC WHERE content IS NULL

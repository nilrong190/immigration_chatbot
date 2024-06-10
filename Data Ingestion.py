# Databricks notebook source
# DBTITLE 1,installs
# Install required packages
%pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 langchain==0.1.5 databricks-vectorsearch==0.22 torch

# Restart Python library
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Chunk the text from the html into segments
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from transformers import AutoTokenizer, OpenAIGPTTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import mlflow
import time
from typing import Iterator
from mlflow import MlflowClient
import mlflow.deployments
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

# Define the name of the Vector Search endpoint
VECTOR_SEARCH_ENDPOINT_NAME = "ic_immigration_vs_endpoint"

# Define the maximum chunk size for text splitting
max_chunk_size = 500

# Load the tokenizer
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

# Create a text splitter using the tokenizer
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)

# Define a function to split text into chunks
def split_text(paragraphs, min_chunk_size=20, max_chunk_size=500):
    new_paragraphs = []
    previous_paragraph = ""
    # Merge paragraphs together to avoid too small docs.
    for paragraph in paragraphs:
        content = paragraph + "\n"
        if len(tokenizer.encode(previous_paragraph + paragraph)) <= max_chunk_size/2:
            previous_paragraph += paragraph + "\n"
        else:
            new_paragraphs.extend(text_splitter.split_text(previous_paragraph.strip()))
            previous_paragraph = paragraph + "\n"
    if previous_paragraph:
        new_paragraphs.extend(text_splitter.split_text(previous_paragraph.strip()))
    # Discard too small chunks
    return [c for c in new_paragraphs if len(tokenizer.encode(c)) > min_chunk_size]

# Define a pandas UDF to parse and split the documents
@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_text)

# Read the input table and split the text into chunks
(spark.table("workspace.default.ic_documentation_2")
      .withColumn('text', F.explode(parse_and_split(col('content'))))
      .drop(col("content"))
      .write.format("delta").mode('overwrite').saveAsTable("workspace.default.ic_documentation_chunked"))

# Enable Change Data Feed for the table
spark.sql("ALTER TABLE workspace.default.ic_documentation_chunked SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# display(spark.table("workspace.default.ic_documentation_chunked"))

# COMMAND ----------

# DBTITLE 1,Endpoint helper functions
def endpoint_exists(vsc, vs_endpoint_name):
    """
    Check if a Vector Search endpoint exists.

    Parameters:
    vsc (VectorSearchClient): Vector Search client object.
    vs_endpoint_name (str): Name of the Vector Search endpoint.

    Returns:
    bool: True if the endpoint exists, False otherwise.
    """
    try:
        return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
    except Exception as e:
        # Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
        if "REQUEST_LIMIT_EXCEEDED" in str(e):
            print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. The demo will consider it exists")
            return True
        else:
            raise e


def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
    """
    Wait for a Vector Search endpoint to be ready.

    Parameters:
    vsc (VectorSearchClient): Vector Search client object.
    vs_endpoint_name (str): Name of the Vector Search endpoint.

    Returns:
    dict: Endpoint details when the endpoint is ready.
    """
    for i in range(180):
        try:
            endpoint = vsc.get_endpoint(vs_endpoint_name)
        except Exception as e:
            # Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
            if "REQUEST_LIMIT_EXCEEDED" in str(e):
                print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. Please manually check your endpoint status")
                return
            else:
                raise e
        status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
        if "ONLINE" in status:
            return endpoint
        elif "PROVISIONING" in status or i < 6:
            if i % 20 == 0:
                print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
            time.sleep(10)
        else:
            raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
    raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

# DBTITLE 1,index helper functions
def index_exists(vsc, endpoint_name, index_full_name):
    """
    Check if an index exists in a Vector Search endpoint.

    Parameters:
    vsc (VectorSearchClient): Vector Search client object.
    endpoint_name (str): Name of the Vector Search endpoint.
    index_full_name (str): Full name of the index.

    Returns:
    bool: True if the index exists, False otherwise.
    """
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False
    
def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
    """
    Wait for an index in a Vector Search endpoint to be ready.

    Parameters:
    vsc (VectorSearchClient): Vector Search client object.
    vs_endpoint_name (str): Name of the Vector Search endpoint.
    index_name (str): Name of the index.

    Raises:
    Exception: If there is an error with the index or if the index is not ready within the timeout.

    """
    for i in range(180):
        idx = vsc.get_index(vs_endpoint_name, index_name).describe()
        index_status = idx.get('status', idx.get('index_status', {}))
        status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
        url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
        if "ONLINE" in status:
            return
        if "UNKNOWN" in status:
            print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
            return
        elif "PROVISIONING" in status:
            if i % 40 == 0:
                print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
            time.sleep(10)
        else:
            raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
    raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

deploy_client = mlflow.deployments.get_deploy_client("databricks")

# Create a VectorSearchClient object
vsc = VectorSearchClient()

# Check if the Vector Search endpoint exists
if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    # If the endpoint does not exist, create it
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

# Wait for the Vector Search endpoint to be ready
wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)

# Print a message indicating that the endpoint is ready
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE ic_documentation_chunked2 AS
# MAGIC     SELECT
# MAGIC         *, 
# MAGIC         explode(text) AS text2
# MAGIC     FROM 
# MAGIC         ic_documentation_chunked

# COMMAND ----------

# The table we'd like to index
source_table_fullname = f"workspace.default.ic_documentation_chunked2"
# Where we want to store our index
vs_index_fullname = f"workspace.default.ic_documentation_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
    print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=vs_index_fullname,
        source_table_name=source_table_fullname,
        pipeline_type="TRIGGERED",
        primary_key="url",
        embedding_source_column='text2',  # The column containing our text
        embedding_model_endpoint_name='databricks-bge-large-en'  # The embedding endpoint used to create the embeddings
    )
    # Let's wait for the index to be ready and all our embeddings to be created and indexed
    wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
    # Trigger a sync to update our vs content with the new data saved in the table
    wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
    vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# Get the deploy client for MLflow deployments on Databricks
deploy_client = mlflow.deployments.get_deploy_client("databricks")

# Define the question for similarity search
question = "How does a resident non-citizen apply for citizenship?"

# Perform similarity search using the Vector Search endpoint
results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
    query_text=question,
    columns=["url", "page_title", "text2"],
    num_results=2
)

# Extract the relevant documents from the search results
docs = results.get('result', {}).get('data_array', [])

# Display the documents
docs

# Databricks notebook source
# DBTITLE 1,installs
# MAGIC %pip install mlflow==2.10.1 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.18.0 mlflow[databricks] cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC %pip install lxml==4.9.3 transformers==4.30.2
# MAGIC dbutils.library.restartPython()  

# COMMAND ----------

# DBTITLE 1,imports
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import mlflow
from typing import Iterator
from mlflow import MlflowClient
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.chat_models import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough
from mlflow.models import infer_signature
from operator import itemgetter
import langchain
import cloudpickle
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize

# COMMAND ----------

# DBTITLE 1,global settings
VECTOR_SEARCH_ENDPOINT_NAME="ic_immigration_vs_endpoint"
SECRET_TOKEN='dose8f76950a53c1443b649e8fde002cdb91'

# COMMAND ----------

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Suppress only the single InsecureRequestWarning from urllib3 needed
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Example request with SSL verification disabled
response = requests.get('https://dbc-fad544b8-a2f8.cloud.databricks.com', verify=False)

# COMMAND ----------

WorkspaceClient().grants.update(c.SecurableType.TABLE, "workspace.default.ic_documentation_vs_index", 
                                changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="8072f060-8886-4e0a-8f84-b1a852efdf28")])

# COMMAND ----------

# DBTITLE 1,Helper functions - test setup
def test_demo_permissions(host, secret_key, vs_endpoint_name, index_name, embedding_endpoint_name = None, managed_embeddings = True):
  error = False
  CSS_REPORT = """
  <style>
  .dbdemos_install{
                      font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,Apple Color Emoji,Segoe UI Emoji,Segoe UI Symbol,Noto Color Emoji,FontAwesome;
  color: #3b3b3b;
  box-shadow: 0 .15rem 1.15rem 0 rgba(58,59,69,.15)!important;
  padding: 10px 20px 20px 20px;
  margin: 10px;
  font-size: 14px !important;
  }
  .dbdemos_block{
      display: block !important;
      width: 900px;
  }
  .code {
      padding: 5px;
      border: 1px solid #e4e4e4;
      font-family: monospace;
      background-color: #f5f5f5;
      margin: 5px 0px 0px 0px;
      display: inline;
  }
  </style>"""

  def display_error(title, error, color=""):
    displayHTML(f"""{CSS_REPORT}
      <div class="dbdemos_install">
                          <h1 style="color: #eb0707">Configuration error: {title}</h1> 
                            {error}
                        </div>""")
  
  def get_email():
    try:
      return spark.sql('select current_user() as user').collect()[0]['user']
    except:
      return 'Uknown'

  def get_token_error(msg, e):
    return f"""
    {msg}<br/><br/>
    Your model will be served using Databrick Serverless endpoint and needs a Pat Token to authenticate.<br/>
    <strong> This must be saved as a secret to be accessible when the model is deployed.</strong><br/><br/>
    Here is how you can add the Pat Token as a secret available within your notebook and for the model:
    <ul>
    <li>
      first, setup the Databricks CLI on your laptop or using this cluster terminal:
      <div class="code dbdemos_block">pip install databricks-cli</div>
    </li>
    <li> 
      Configure the CLI. You'll need your workspace URL and a PAT token from your profile page
      <div class="code dbdemos_block">databricks configure</div>
    </li>  
    <li>
      Create the dbdemos scope:
      <div class="code dbdemos_block">databricks secrets create-scope dbdemos</div>
    <li>
      Save your service principal secret. It will be used by the Model Endpoint to autenticate. <br/>
      If this is a demo/test, you can use one of your PAT token.
      <div class="code dbdemos_block">databricks secrets put-secret dbdemos rag_sp_token</div>
    </li>
    <li>
      Optional - if someone else created the scope, make sure they give you read access to the secret:
      <div class="code dbdemos_block">databricks secrets put-acl dbdemos '{get_email()}' READ</div>

    </li>  
    </ul>  
    <br/>
    Detailed error trying to access the secret:
      <div class="code dbdemos_block">{e}</div>"""

  try:
    secret = dbutils.secrets.get(secret_scope, secret_key)
    secret_principal = "__UNKNOWN__"
    try:
      from databricks.sdk import WorkspaceClient
      w = WorkspaceClient(token=dbutils.secrets.get(secret_scope, secret_key), host=host)
      print(f"w: {w}")
      print(f"w.current_user: {w.current_user.me().name}")
      secret_principal = w.current_user.me().emails[0].value
      print(f"secret_principal: {secret_principal}")
    except Exception as e_sp:
      error = True
      display_error(f"Couldn't get the SP identity using the Pat Token saved in your secret", 
                    get_token_error(f"<strong>This likely means that the Pat Token saved in your secret {secret_scope}/{secret_key} is incorrect or expired. Consider replacing it.</strong>", e_sp))
      return
  except Exception as e:
    error = True
    display_error(f"We couldn't access the Pat Token saved in the secret {secret_scope}/{secret_key}", 
                  get_token_error("<strong>This likely means your secret isn't set or not accessible for your user</strong>.", e))
    return
  
  try:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=secret, disable_notice=True)
    print(f"vsc: {vsc}")
    vs_index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name)
    print(f"vs_index: {vs_index}")
    if embedding_endpoint_name:
      if managed_embeddings:
        from langchain_community.embeddings import DatabricksEmbeddings
        results = vs_index.similarity_search(query_text='What is Apache Spark?', columns=["paragraph"], num_results=1)
      else:
        from langchain_community.embeddings import DatabricksEmbeddings
        embedding_model = DatabricksEmbeddings(endpoint=embedding_endpoint_name)
        embeddings = embedding_model.embed_query('What is Apache Spark?')
        results = vs_index.similarity_search(query_vector=embeddings, columns=["paragraph"], num_results=1)

  except Exception as e:
    error = True
    vs_error = f"""
    Why are we getting this error?<br/>
    The model is using the Pat Token saved with the secret {secret_scope}/{secret_key} to access your vector search index '{index_name}' (host:{host}).<br/><br/>
    To do so, the principal owning the Pat Token must have USAGE permission on your schema and READ permission on the index.<br/>
    The principal is the one who generated the token you saved as secret: `{secret_principal}`. <br/>
    <i>Note: Production-grade deployement should to use a Service Principal ID instead.</i><br/>
    <br/>
    Here is how you can fix it:<br/><br/>
    <strong>Make sure your Service Principal has USE privileve on the schema</strong>:
    <div class="code dbdemos_block">
    spark.sql('GRANT USAGE ON CATALOG `{catalog}` TO `{secret_principal}`');<br/>
    spark.sql('GRANT USAGE ON DATABASE `{catalog}`.`{db}` TO `{secret_principal}`');<br/>
    </div>
    <br/>
    <strong>Grant SELECT access to your SP to your index:</strong>
    <div class="code dbdemos_block">
    from databricks.sdk import WorkspaceClient<br/>
    import databricks.sdk.service.catalog as c<br/>
    WorkspaceClient().grants.update(c.SecurableType.TABLE, "{index_name}",<br/>
                                            changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="{secret_principal}")])
    </div>
    <br/>
    <strong>If this is still not working, make sure the value saved in your {secret_scope}/{secret_key} secret is your SP pat token </strong>.<br/>
    <i>Note: if you're using a shared demo workspace, please do not change the secret value if was set to a valid SP value by your admins.</i>

    <br/>
    <br/>
    Detailed error trying to access the endpoint:
    <div class="code dbdemos_block">{str(e)}</div>
    </div>
    """
    if "403" in str(e):
      display_error(f"Permission error on Vector Search index {index_name} using the endpoint {vs_endpoint_name} and secret {secret_scope}/{secret_key}", vs_error)
    else:
      display_error(f"Unkown error accessing the Vector Search index {index_name} using the endpoint {vs_endpoint_name} and secret {secret_scope}/{secret_key}", vs_error)
  def get_wid():
    try:
      return dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('orgId')
    except:
      return None
  if get_wid() in ["5206439413157315", "984752964297111", "1444828305810485", "2556758628403379"]:
    print(f"----------------------------\nYou are in a Shared FE workspace. Please don't override the secret value (it's set to the SP `{secret_principal}`).\n---------------------------")

  if not error:
    print('Secret and permissions seems to be properly setup, you can continue the demo!')

# COMMAND ----------

# DBTITLE 1,Helper functions - model
# Helper function
def get_latest_model_version(model_name):
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# DBTITLE 1,Test the setup
catalog="workspace"
db="default"
index_name=f"{catalog}.{db}.ic_documentation_vs_index"

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
host

#test_demo_permissions(host, secret_scope="CARES_POC", secret_key=SECRET_TOKEN, vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="databricks-dbrx-instruct")

# COMMAND ----------

# DBTITLE 1,set up authentication for the model
# url used to send the request to your model from the serverless endpoint
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = SECRET_TOKEN

# COMMAND ----------

# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
embedding_model2 = DatabricksEmbeddings(endpoint="databricks-dbrx-instruct")
# print(f"Test embeddings: {embedding_model.embed_query('What is Apache Spark?')[:20]}...")

def get_retriever(persist_dir: str = None):
    print(host)
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, service_principal_client_id='8072f060-8886-4e0a-8f84-b1a852efdf28', service_principal_client_secret='dose8f76950a53c1443b649e8fde002cdb91')
  
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text2", embedding=embedding_model2, columns=["url", "page_title"]
    )
    return vectorstore.as_retriever()

# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("How does a resident alien apply for a green card?")
print(f"Relevant documents: {similar_documents[3]}")

# COMMAND ----------

# Test Databricks Foundation LLM model
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 800)
print(f"Test chat model: {chat_model.invoke('How does a resident alien apply for a green card?')}")

# COMMAND ----------

chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 2048)
retriever = get_retriever()

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically, using the key concepts of the most recent chat history below. There should be high similarity to ensure that we are using the correct source. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_template_knn = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve the most relevant documents so that we can better answer the question. The external data source utilizes a K-Nearest Neighbors (KNN) approach to search for relevant documents in a vector space. Therefore, the query should identify the key concepts from the most recent chat history and ask for the 4 most relevant documents based on these concepts. The number of documents, 4, should be specified in the query to ensure comprehensive retrieval. This query should be framed in natural language and focus on high semantic relevance to the key concepts of the chat history. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template_knn
)

is_question_about_uscis_str = """
You are classifying documents to know if this question is related to United States Citizenship and Immigrations Services regarding statutes, policies, interpretations, implementation, social impacts or something from a very different field. Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: What is the purpose of the USCIS policy manual?, classify this question: Do you have more details?
Expected Response: Yes

Question: Knowing this followup history: What is the purpose of the USCIS policy manual?, classify this question: Write me a song.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_uscis_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_uscis_str
)

is_about_uscis_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_uscis_prompt
    | chat_model
    | StrOutputParser()
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

question_with_history_and_context_str = """
You are a trustful assistant for people who need help understanding United State Citizenship and Immigations Service policies. You are answering legal, policy, interpretation, operational guideline, and opinions related to USCIS policy. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible unless you are asked to provide more detailed or complete information.  You may also be asked to summarize in which case you should be as complete as possible, but concise. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "assistant". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [f"{d.metadata['url']} page:{d.metadata['page_title']}" for d in docs]

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about U.S. Citizenship and Immigation policies.', "sources": []})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": is_about_uscis_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

dialog = {
    "messages": [
        {"role": "user", "content": "What is the purpose of USCIS policy?"}, 
        {"role": "assistant", "content": "USCIS policies aim to manage the immigration system in a way that balances the needs and interests of the United States, the immigrant community, and the general public."}, 
        {"role": "user", "content": "Who are the USCIS policies intended for?"},
        {"role": "assistant", "content": "USCIS policies are intended for a wide range of stakeholders, including immigrants, USCIS employees, employers, legal representatives, and government agencies, among others. These policies aim to ensure a fair, efficient, and secure immigration system while addressing the needs and responsibilities of these diverse groups."},
        {"role": "user", "content": "What are the key requirements to apply for a green card?"},
        {"role": "assistant", "content": "To apply for a green card, applicants must fall under an eligible category, have a qualifying sponsor, and be admissible to the United States. They need to submit a completed Form I-485 with supporting documents and attend a biometrics appointment and interview. Payment of the required application fees is also necessary unless a fee waiver is granted."},
        {"role": "user", "content": "How does the USCIS policies define a resident alien?"},
        {"role": "assistant", "content": "USCIS policies define a resident alien as a non-citizen who is legally authorized to live and work in the United States on a permanent basis. This status is typically granted to individuals who hold a green card (lawful permanent resident), allowing them to reside permanently in the U.S., subject to certain conditions and responsibilities, such as paying taxes and maintaining lawful status."},
        {"role": "user", "content": "How is a child treated under the USCIS policy?"},
        {"role": "assistant", "content": "Under USCIS policy, a child is generally defined as an unmarried person under the age of 21. The policies provide specific provisions for children in various contexts, including: Derivatives: Children may qualify as derivatives on their parent's immigration application, meaning they can obtain immigration benefits based on their parent's status. Immediate Relatives: Children of U.S. citizens are considered immediate relatives and can apply for a green card without numerical limitations or waiting periods. Protection under the Child Status Protection Act (CSPA): This act helps protect children from aging out (turning 21 before their application is processed) by allowing them to retain their eligibility for immigration benefits under certain conditions. Special Immigrant Juvenile (SIJ) Status: This status provides a pathway to a green card for children who have been abused, neglected, or abandoned by one or both parents and meet other specific criteria. Overall, USCIS policies aim to safeguard the rights and interests of children in the immigration process, providing various protections and pathways to residency and citizenship."},
    ]
}

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.ic_immigration_chatbot_model"

with mlflow.start_run(run_name="ic_immigration_chatbot_rag") as run:

    output = full_chain.invoke(dialog)
    signature = infer_signature(dialog, output)    
    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__
        ],
        input_example=dialog,
        signature=signature,
        example_no_conversion=True,
    )

# COMMAND ----------

serving_endpoint_name = f"ic_immigation_endpoint_{catalog}_{db}"[:63]
latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_TOKEN": "{{SECRET_TOKEN}}",  # <scope>/<secret> that contains an access token
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions).

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# MAGIC
# MAGIC # Semantic Cache Solution Accelerator
# MAGIC
# MAGIC Generative AI models are increasingly revolutionizing industries, with techniques like Retrieval Augmented Generation (RAG) and Compound AI systems leading the charge. These models empower organizations by enhancing capabilities such as information retrieval, decision-making, and content generation. However, the implementation of these systems is often accompanied by significant costs, especially in terms of computational resources. Despite these challenges, the rapid advancement of AI platforms and the development of more efficient algorithms are enabling businesses to optimize costs and scale AI-driven solutions more effectively.
# MAGIC
# MAGIC Semantic cache is a technique that is adopted by many enterprises to reduce the computational load of AI-driven systems. As generative AI models handle increasingly complex queries, there is often semantic overlap between different queries, such as users asking variations of the same question. Without semantic caching, these systems would need to repeatedly perform resource-intensive computations, leading to inefficiencies. By storing the previously processed queries and responses, semantic caching allows AI models to retrieve relevant information without recalculating, thereby reducing latency, lowering server load, and conserving computational resources. This becomes especially important as AI applications scale, ensuring cost-effectiveness and maintaining high performance, particularly in natural language processing, where nuanced query variations are frequent.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How Semantic Caching works
# MAGIC
# MAGIC Semantic caching leverages a vector database to store and retrieve answers based on the meaning or semantics of a question rather than just its keywords. In this system, each question is embedded as a vector, and the cached answers are stored. When a new query is submitted, the system searches the database for similar vectors, returning a cached response when a match is found. When a suitable match is not found, the system proceeds to execute the standard pipeline to generate a response, and in turn persists the new question and answer pair in the database. 
# MAGIC
# MAGIC This technique is particularly effective for handling high-volume, repetitive queries such as those often found in customer FAQs, where users frequently ask the same or similar questions. Some of the key business benefits of semantic cache are:
# MAGIC
# MAGIC - Reduce Costs: With fewer computationally expensive model calls, businesses will see significant cost savings. The system bypasses the need to generate new answers for questions that have already been asked, leading to reduced usage of cloud resources and lower operational costs.
# MAGIC - Faster Response Time: Customer satisfaction is closely tied to how quickly they receive answers. With semantic caching, chatbots can instantly retrieve answers from the cache, dramatically reducing the time it takes to respond to queries.
# MAGIC - Scalability: As businesses scale, so do the number of customer inquiries. Caching frequently asked questions ensures the chatbot can handle increased volumes without a corresponding increase in costs or latency.
# MAGIC
# MAGIC Some use cases we see in the market that are especially suitable for semantic caching include:
# MAGIC
# MAGIC - FAQs: Questions that customers frequently ask—such as product details, order statuses, or return policies—are prime candidates for caching. Businesses can quickly address these repetitive queries without taxing the system.
# MAGIC - Support Tickets: For companies that manage large-scale customer support, semantic caching can be implemented to address commonly recurring issues.
# MAGIC - Internal Knowledge Bases: Employees often ask the same internal queries, and caching these answers can improve productivity by providing instant access to stored knowledge.
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Semantic Cache on Databricks Mosaic AI
# MAGIC
# MAGIC Databricks provides an optimal platform for building AI agents with semantic caching capabilities. With Databricks Mosaic AI, users have access to all necessary components such as a vector database, agent development framework, agent serving, and an agent evaluation framework on a unified, highly governed platform. This ensures that key assets, including data, vector indexes, models, agents, and endpoints, are centrally managed under robust governance.
# MAGIC
# MAGIC Mosaic AI also offers an open architecture, allowing users to experiment with various models for embeddings and generation. Leveraging the Mosaic AI Agent Framework and Evaluation tools, users can rapidly iterate on applications until they meet production-level standards. Once deployed, KPIs like hit ratios and latency can be monitored using MLflow traces, which are automatically logged in Inference Tables for easy tracking.
# MAGIC
# MAGIC If you're looking to implement semantic caching for your AI system on Databricks, we're excited to introduce the Semantic Cache Solution Accelerator. This accelerator is designed to help you get started quickly and efficiently, providing a streamlined path to implementation.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Case Study
# MAGIC
# MAGIC Imagine you're operating an AI chatbot on your products' public documentation page. This chatbot answers visitors' questions about your products using a retriever-augmented generation (RAG) architecture. After reviewing the submitted user questions and responses, you notice a large number of redundant queries—phrased differently but carrying the same meaning. You're getting feedback from the users that the chatbot's response time is too long and also facing pressure from management to reduce the operational costs of the chatbot. 
# MAGIC
# MAGIC In the following notebooks, we'll explore how semantic caching can significantly lower both total cost and latency, with only a minor trade-off in response quality.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC
# MAGIC The dataset we use for this solution accelrator is synthetic and stored inside `./data/`. We first generated a list of 10 questions related to Databricks Machine Learning product features using [dbrx-instruct](https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html#dbrx-instruct). We then bootstrapped these questions to generate 100 questions. We reformulate each of the 10 questions slightly differently without changing the meaning. We used [Meta Llama 3.1 70B Instruct](https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html#meta-llama-31-70b-instruct) for this. 
# MAGIC
# MAGIC The goal of this exploratory data analysis is to find the similarity threshold that strikes the balance between precision and recall when we check the cache to search for previously asked question. The meaning of this will become clearer in the next notebooks.

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.

# COMMAND ----------



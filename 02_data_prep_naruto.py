# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/personalized_image_gen).

# COMMAND ----------

# MAGIC %run ./99_utils_text_to_image

# COMMAND ----------

# MAGIC %md
# MAGIC #Prepare your images for fine-tuning
# MAGIC  Tailoring the output of a generative model is crucial for building a successful application. This applies to use cases powered by image generation models as well. Imagine a scenaio where a furniture designer seeks to generate images for ideation purposes and they want their old products to be reflected on the generated images. Not only that but they also want to see some variations, for example, in material or color. In such instances, it is imperative that the model knows their old products and can apply some new styles on them. Customization is necessary in a case like this. We can do this by fine-tuning a pre-trained model on our own images.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manage your images in Unity Catalog Volumes

# COMMAND ----------

# MAGIC %md
# MAGIC This solution accelerator uses 25 training images stored in the subfolders of ```/images/chair/``` to fine-tune a model. If you have imported this accelerator from GitHub, the images should already be in place.  If you simply downloaded the notebooks, you will need to create the folder structure in your workspace and import the images from https://github.com/databricks-industry-solutions/personalized_image_gen for the following cells to work without modification.

# COMMAND ----------

theme = "naruto"
catalog = "hiroshi" # Name of the catalog we use to manage our assets (e.g. images, weights, datasets) 
volumes_dir = f"/Volumes/{catalog}/{theme}" # Path to the directories in UC Volumes

# COMMAND ----------

# Make sure that the catalog and the schema exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}") 
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{theme}") 

# COMMAND ----------

from datasets import load_dataset

dataset = load_dataset("lambdalabs/naruto-blip-captions")

# COMMAND ----------

show_image_grid(dataset["train"][:25]["image"], 5, 5) # Custom function defined in util notebook

# COMMAND ----------

dataset["train"][:25]["text"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manage Dataset in UC Volumes
# MAGIC We create a Hugging Face Dataset object and store it in Unity Catalog Volume.

# COMMAND ----------


spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{theme}.dataset")
dataset.save_to_disk(f"/Volumes/{catalog}/{theme}/dataset")

# COMMAND ----------

# MAGIC %md Let's free up some memory again.

# COMMAND ----------

import gc
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | bitsandbytes | Accessible large language models via k-bit quantization for PyTorch. | MIT | https://pypi.org/project/bitsandbytes/
# MAGIC | diffusers | A library for pretrained diffusion models for generating images, audio, etc. | Apache 2.0 | https://pypi.org/project/diffusers/
# MAGIC | stable-diffusion-xl-base-1.0 | A model that can be used to generate and modify images based on text prompts. | CreativeML Open RAIL++-M License | https://github.com/Stability-AI/generative-models

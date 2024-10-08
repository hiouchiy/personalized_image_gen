# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/personalized_image_gen).

# COMMAND ----------

# MAGIC %md
# MAGIC Azure Standard_NC48ads_A100_v4 (A100 x 2)を推奨。

# COMMAND ----------

# DBTITLE 1,Install requirements and load helper functions
# MAGIC %run ./99_utils_text_to_image

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the normal version of Stable Diffusion XL model before Fine-tuning

# COMMAND ----------

from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0", 
  torch_dtype=torch.float16).to("cuda")

image = pipeline(
  prompt="Bill Gates with a hoodie under the blue sky", 
  negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, bad face, bad finger", 
  num_inference_steps=25, 
  guidance_scale=7.5).images[0]
show_image(image)

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine-tune Stable Diffusion XL with LoRA
# MAGIC For fine-tuning, we use [Text-to-Image](https://huggingface.co/docs/diffusers/en/training/text2image), which is a technique to update the weights of a pre-trained text-to-image model. We use the [Diffusers](https://huggingface.co/docs/diffusers/en/index)' implementation of DreamBooth in this solution accelerator.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up TensorBoard
# MAGIC [TensorBoard](https://www.tensorflow.org/tensorboard) is an open source monitoring solution for model training. It reads an event log and exposes the training metrics in near real-time on its dashboard, which helps gauge the status of fine-tuning without having to wait until it's done.
# MAGIC
# MAGIC Note that when you write the event log to DBFS, it won't show until the file is closed for writing, which is when the training is complete. This is not good for real time monitoring. So we suggest to write the event log out to the driver node and run your TensorBoard from there (see the cell below on how to do this). Files stored on the driver node may get removed when the cluster terminates or restarts. But when you are running the training on Databricks notebook, MLflow will automatically log your Tensorboard artifacts, and you will be able to recover them later. You can find the example of this below.

# COMMAND ----------

import os
from tensorboard import notebook

logdir = "/databricks/driver/logdir/sdxl/" # Write event log to the driver node
# logdir = "/dbfs/tmp/logdir/sdxl/" # Write event log to DBFS
notebook.start("--logdir {} --reload_multifile True".format(logdir))

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's specifiy some variables.

# COMMAND ----------

theme = "naruto"
catalog = "sdxl_image_gen"
volumes_dir = "/Volumes/sdxl_image_gen"
os.environ["DATASET_NAME"] = f"{volumes_dir}/{theme}/dataset"
os.environ["OUTPUT_DIR"] = f"{volumes_dir}/{theme}/loraadaptor"
os.environ["LOGDIR"] = logdir

# Make sure that the volume exists
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{theme}.loraadaptor")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Parameters
# MAGIC To ensure we can use DreamBooth with LoRA on a heavy pipeline like Stable Diffusion XL, we use the following hyperparameters:
# MAGIC
# MAGIC * Gradient checkpointing (`--gradient_accumulation_steps`)
# MAGIC * 8-bit Adam (`--use_8bit_adam`)
# MAGIC * Mixed-precision training (`--mixed-precision="fp16"`)
# MAGIC * Some other parameters are defined in `yamls/zero2.yaml`
# MAGIC <br>
# MAGIC
# MAGIC Other parameters:
# MAGIC * Use `--output_dir` to specify your LoRA model repository name.
# MAGIC * Use `--caption_column` to specify name of the caption column in your dataset.
# MAGIC * Make sure to pass the right number of GPUs to the parameter `num_processes` in `yamls/zero2.yaml`: e.g. `num_processes` should be 8 for `g5.48xlarge`.
# MAGIC <br>
# MAGIC
# MAGIC
# MAGIC The following cell will run for about 15 minutes on a single node cluster with 8xA10GPU instances on the default training images. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### For Multi-GPU on Single node
# MAGIC
# MAGIC Using HF accelerate is better.

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file yamls/zero2.yaml personalized_image_generation/train_text_to_image_lora_sdxl.py \
# MAGIC   --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
# MAGIC   --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
# MAGIC   --dataset_name=$DATASET_NAME \
# MAGIC   --dataloader_num_workers=8 \
# MAGIC   --resolution=512 \
# MAGIC   --center_crop \
# MAGIC   --random_flip \
# MAGIC   --train_batch_size=2 \
# MAGIC   --gradient_accumulation_steps=4 \
# MAGIC   --max_grad_norm=1 \
# MAGIC   --max_train_steps=7500 \
# MAGIC   --learning_rate=1e-04 \
# MAGIC   --lr_scheduler="cosine" \
# MAGIC   --lr_warmup_steps=0 \
# MAGIC   --mixed_precision="fp16" \
# MAGIC   --checkpointing_steps=3750 \
# MAGIC   --validation_prompt="A naruto with blue eyes." \
# MAGIC   --seed=1337 \
# MAGIC   --output_dir=$OUTPUT_DIR \
# MAGIC   --report_to="tensorboard" \
# MAGIC   --logging_dir=$LOGDIR

# COMMAND ----------

# MAGIC %md
# MAGIC ### For Multi-GPU on Multi node
# MAGIC
# MAGIC Using TorchDistributor is better.

# COMMAND ----------

import os
from pyspark.ml.torch.distributor import TorchDistributor
distributor = TorchDistributor(
    num_processes=2,
    local_mode=True,
    use_gpu=True)

distributor.run(
  'personalized_image_generation/train_text_to_image_lora_sdxl.py', 
  '--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0',
  '--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix',
  f'--dataset_name={os.environ["DATASET_NAME"]}',
  '--dataloader_num_workers=8',
  '--resolution=512',
  '--center_crop',
  '--random_flip',
  '--max_grad_norm=1',
  f'--output_dir={os.environ["OUTPUT_DIR"]}',
  '--train_batch_size=2',
  '--gradient_accumulation_steps=4',
  '--learning_rate=1e-04',
  '--lr_scheduler=constant',
  '--lr_warmup_steps=0',
  '--max_train_steps=7500',
  # '--num_train_epochs=1',
  '--checkpointing_steps=3750',
  '--seed=1337',
  '--validation_prompt="A naruto with blue eyes"',
  '--report_to=tensorboard',
  f'--logging_dir={os.environ["LOGDIR"]}',
  '--mixed_precision=fp16'
)

# COMMAND ----------

# MAGIC %sh ls -ltrh $OUTPUT_DIR

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test inference
# MAGIC Lets take the fine-tuned model and generate some images!

# COMMAND ----------

from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0", 
  torch_dtype=torch.float16).to("cuda")

pipeline.load_lora_weights(f"{volumes_dir}/{theme}/loraadaptor", weight_name="pytorch_lora_weights.safetensors")

image = pipeline(
  prompt="Bill Gates with a hoodie under the blue sky", 
  negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, bad face, bad finger", 
  num_inference_steps=25, 
  guidance_scale=7.5).images[0]
show_image(image)

# COMMAND ----------

import os
import glob

persons = ["Steve Jobs", "Donald Trump", "Angelina Jolie", "Beyoncé", "Michael Jackson"]
num_imgs_to_preview = len(persons)
imgs = []
for person in persons:
    imgs.append(
        pipeline(
            prompt=f"A photo of {person} with a hoodie under the blue sky",
            negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy", 
            num_inference_steps=25,
        ).images[0]
    )
show_image_grid(imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLflow

# COMMAND ----------

import mlflow
import torch

class sdxl_fine_tuned(mlflow.pyfunc.PythonModel):
    def __init__(self, vae_name, model_name):
        self.vae_name = vae_name
        self.model_name = model_name

    def load_context(self, context):
        """
        This method initializes the vae and the model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = diffusers.AutoencoderKL.from_pretrained(
            self.vae_name, torch_dtype=torch.float16
        )
        self.pipe = diffusers.DiffusionPipeline.from_pretrained(
            self.model_name,
            vae=self.vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.load_lora_weights(context.artifacts["repository"])
        self.pipe = self.pipe.to(self.device)

    def predict(self, context, model_input):
        """
        This method generates output for the given input.
        """
        prompt = model_input["prompt"][0]
        num_inference_steps = model_input.get("num_inference_steps", [25])[0]
        # Generate the image
        image = self.pipe(
            prompt=prompt, num_inference_steps=num_inference_steps
        ).images[0]
        # Convert the image to numpy array for returning as prediction
        image_np = np.array(image)
        return image_np


# COMMAND ----------

vae_name = "madebyollin/sdxl-vae-fp16-fix"
model_name = "stabilityai/stable-diffusion-xl-base-1.0"
output = f"{volumes_dir}/{theme}/loraadaptor/pytorch_lora_weights.safetensors"

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec, TensorSpec
import transformers, bitsandbytes, accelerate, deepspeed, diffusers

experiment_name = f"/Workspace/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/experiment"
mlflow.set_experiment(experiment_name)

mlflow.set_registry_uri("databricks-uc")

# Define input and output schema
input_schema = Schema(
    [ColSpec(DataType.string, "prompt"), ColSpec(DataType.long, "num_inference_steps")]
)
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, 768, 3))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame(
    {"prompt": [f"A photo of a {theme} in a living room"], "num_inference_steps": [25]}
)

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=sdxl_fine_tuned(vae_name, model_name),
        artifacts={"repository": output},
        pip_requirements=[
            "transformers==" + transformers.__version__,
            "bitsandbytes==" + bitsandbytes.__version__,
            "accelerate==" + accelerate.__version__,
            "deepspeed==" + deepspeed.__version__,
            "diffusers==" + diffusers.__version__,
        ],
        input_example=input_example,
        signature=signature,
    )
    mlflow.set_tag("dataset", f"{volumes_dir}/{theme}/dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog

# COMMAND ----------

# Make sure that the schema for the model exist
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.model")

# Register the model 
registered_name = f"{catalog}.model.sdxl-fine-tuned-{theme}"
result = mlflow.register_model(
    "runs:/" + run.info.run_id + "/model",
    registered_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the registered model back to make inference
# MAGIC If you come accross an out of memory issue, restart the Python kernel to release the GPU memory occupied in Training. For this, uncomment and run the following cell, and re-define the variables such as ```theme```, ```catalog```, and ```volume_dir```.

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

def get_latest_model_version(mlflow_client, registered_name):
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{registered_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

import mlflow
from mlflow import MlflowClient
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_name = f"{catalog}.model.sdxl-fine-tuned-{theme}"
model_version = get_latest_model_version(mlflow_client, registered_name)
logged_model = f"models:/{registered_name}/{model_version}"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Armed with this model, the design team can now explore new variations of their products and even produce all-together new items reflective of the designs of previously produced items in their portfolio.

# COMMAND ----------

# Use any of the following token to generate personalized images: 'bcnchr', 'emslng', 'hsmnchr', 'rckchr', 'wdnchr'
input_example = pd.DataFrame(
    {
        "prompt": ["A photo of a long brown sofa in the style of the bcnchr chair"],
        "num_inference_steps": [25],
    }
)
image = loaded_model.predict(input_example)
show_image(image)

# COMMAND ----------

# Assign an alias to the model
mlflow_client.set_registered_model_alias(registered_name, "champion", model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | bitsandbytes | Accessible large language models via k-bit quantization for PyTorch. | MIT | https://pypi.org/project/bitsandbytes/
# MAGIC | diffusers | A library for pretrained diffusion models for generating images, audio, etc. | Apache 2.0 | https://pypi.org/project/diffusers/
# MAGIC | stable-diffusion-xl-base-1.0 | A model that can be used to generate and modify images based on text prompts. | CreativeML Open RAIL++-M License | https://github.com/Stability-AI/generative-models

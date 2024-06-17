import os
import argparse
import boto3
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
import pandas as pd
import torch
from datasets import Dataset
import logging
from huggingface_hub.hf_api import HfFolder

logger = logging.getLogger(__name__)


def load_data_from_s3(bucket_name, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return pd.read_csv(obj['Body'])


def preprocess_function(examples, tokenizer, max_length):
    return tokenizer(examples['text'], truncation=True, max_length=max_length, padding='max_length')


def main(args):
    logger.info(f"Fetching dataset {args.bucket_name} from bucket {args.data_key}")
    df = load_data_from_s3(args.bucket_name, args.data_key)

    train_dataset = Dataset.from_pandas(df)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules='all-linear',
        bias="none",
        task_type="CASUAL_LM"
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    logger.info(f"Fetching tokenizer for model {args.model_name} from HuggingFace")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    logger.info(f"Fetching pre-trained model {args.model_name} from HuggingFace")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=quant_config, device_map="auto", token=args.hf_token)
    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Configuring model for LoRA training")
    peft_model = get_peft_model(model, lora_config)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        peft_model.is_parallelizable = True
        peft_model.model_parallel = True

    logger.info("Tokenizing dataset")
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer, args.max_length), batched=True)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=500,
        learning_rate=args.learning_rate,
        weight_decay=0,
        push_to_hub=False,
        save_total_limit=1,
        save_strategy="steps",
        save_steps=6000
    )
    # Initialize Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )
    # Train the model
    logger.info("Starting model training")
    trainer.train()

    logger.info(f"Saving merged model to directory f{args.output_dir}")
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--model_name', type=str, default='mistral-7B-instruct-v0.2')
    parser.add_argument('--bucket_name', type=str)
    parser.add_argument('--data_key', type=str)
    parser.add_argument('--output_dir', type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--hf_token', type=str)
    
    args = parser.parse_args()
    HfFolder.save_token(args.hf_token)
    main(args)

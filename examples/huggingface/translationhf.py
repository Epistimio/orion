# %% [markdown]
# # Fine-tune a pretrained model from Hugging Face
# 
# - fine-tuning Helsinki-NLP
# 
# source tutorial: https://huggingface.co/docs/transformers/training
import importlib.util
import logging
import os
os.environ["COMET_API_KEY"] = "iwR8pGUAqOGs1UWAE0HIL7dbv" 
os.environ["COMET_WORKSPACE"] = "patate" 
os.environ["COMET_PROJECT_NAME"] = "test3" 
os.environ["COMET_MODE"] = "ONLINE"
os.environ["COMET_LOG_ASSETS"] = "True"
os.environ["COMET_AUTO_LOG_METRICS"] = "True"
import comet_ml
from transformers.integrations import CometCallback 
import numpy as np
import torch
import argparse
from copy import deepcopy



# %%
from datasets import load_dataset, load_metric, load_dataset_builder
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainerCallback, TrainingArguments)

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


import hydra
from omegaconf import DictConfig 



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-st', '--size_train_dataset', help='Number of samples to use from training data set. If not specified, use complete dataset', type=int, required=False)
    parser.add_argument('-se', '--size_eval_dataset', help='Number of samples to use from evaluation data set. If not specified, use complete dataset', type=int, required=False)
    parser.add_argument('-f', '--freeze_base_model', help='Freeze parameters of base model during training', action='store_true', required=False)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', type=float, required=False)
    parser.add_argument('-e', '--num_train_epochs', help="Number of training epochs", type=int, required=False)
    parser.add_argument('-b', '--per_device_train_batch_size', help="Per device batch size", type=int, required=False)
    parser.add_argument('-opt', '--optim', help="Optimizer (one of: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor", type=str, required=False)
    parser.add_argument('-wd', '--weight_decay', help="Weight decay for AdamW optimizer", type=float, required=False)
    parser.add_argument('-b1', '--adam_beta1', help="beta1 hyperparameter for AdamW optimizer", type=float, required=False)
    parser.add_argument('-b2', '--adam_beta2', help="beta2 hyperparameter for AdamW optimizer", type=float, required=False)
    parser.add_argument('-eps', '--adam_epsilon', help="epsilon hyperparameter for AdamW optimizer", type=float, required=False)
    parser.add_argument('-log', '--logfile', help='Log file name and path', type=str, required=False)
    args = parser.parse_args()
    return vars(args)

def set_training_args(training_args, args):
    for argname, argvalue in args.items():
        if argvalue is not None:
            setattr(training_args, argname, argvalue)
    return training_args

class GPUMemoryCallback(TrainerCallback):
  def on_epoch_end(self, args, state, control, **kwargs):
    print("GPU mem: Tot - ", torch.cuda.get_device_properties(0).total_memory,
          "res - ", torch.cuda.memory_reserved(0),
          "used - ", torch.cuda.memory_allocated(0))

def get_free_gpu():
    for i in range(torch.cuda.device_count()):
        gpu_procs_str = torch.cuda.list_gpu_processes(i)
        if "no processes are running" in gpu_procs_str:
            return i
    return None


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> float:
    print("args",cfg)
    #%%
    # Get command line arguments and apply hyperparaters to training arguments
    args = cfg.args
    #%%
    # Logger setup
    if args['logfile'] is not None:
        logfile = args['logfile']
    else:
        logfile = "translation_hf.log"
    logging.basicConfig(filename=logfile, level=logging.INFO)
    logger = logging.getLogger()

    # %%
    # Get a GPU if available
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    logger.info("Compute device: %s", device)

    batch_size=16
    #%%
    # Get hyperparameters
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(os.getcwd())+"/test_trainer", 
        save_total_limit=2, 
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch", 
        predict_with_generate=True)
    
    training_args = set_training_args(training_args, args)
    print("Training arguments:", training_args)

    
    # %%
    # Load a dataset
    dataset_name = "wmt16"    # others: yelp_review_full, rotten_tomatoes
    logger.info("Dataset: %s", dataset_name)

    raw_dataset = load_dataset(dataset_name, "ro-en")

    model_checkpoint = "Helsinki-NLP/opus-mt-en-ro"

    # Create tokenizer and tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if "mbart" in model_checkpoint:
        tokenizer.src_lang = "en-XX"
        tokenizer.tgt_lang = "ro-RO"
    
    
    if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "translate English to Romanian: "
    else:
        prefix = ""
    
    max_input_length = 128
    max_target_length = 128
    source_lang = "en"
    target_lang = "ro"

    def preprocess_function(examples):
        inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_dataset.map(preprocess_function, batched=True)

    # Use only a subset of the available data, if desired
    size_train_dataset = len(tokenized_datasets["train"])
    size_eval_dataset = len(tokenized_datasets["validation"])
    if args["size_train_dataset"] is not None:
        size_train_dataset = args["size_train_dataset"]
    if args["size_eval_dataset"] is not None:
        size_eval_dataset = args["size_eval_dataset"]
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(size_train_dataset))
    eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(size_eval_dataset))

    # Create model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    model_name = model_checkpoint.split("/")[-1]

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Train the model
    metric = load_metric("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    class CustomCallback(TrainerCallback):
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
    
        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                control_copy = deepcopy(control)
                self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
                return control_copy

    #cometCallb = CometCallback()
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[CometCallback()],
        )
    trainer.add_callback(CustomCallback(trainer))
    

    trainer.train()
    # Evaluate model
    eval_metrics = trainer.evaluate()
    #experiment.log_metrics(eval_metrics) 
    # Print out memory stats
    print("Total GPU memory:", torch.cuda.get_device_properties(0).total_memory)
    print("GPU memory reserved:", torch.cuda.memory_reserved(0))
    print("GPU memory allocated:", torch.cuda.memory_allocated(0))

    return -eval_metrics["eval_bleu"]

#=======================================================================
# Main
#=======================================================================
if __name__ == '__main__':
    main()
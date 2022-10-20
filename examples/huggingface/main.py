# %% [markdown]
# # Fine-tune a pretrained model from Hugging Face
#
# - fine-tuning BERT
#
# source tutorial: https://huggingface.co/docs/transformers/training

import argparse
import logging

import hydra
import numpy as np
import torch
from datasets import load_dataset, load_dataset_builder, load_metric
from omegaconf import DictConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from orion.client import report_objective


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-st",
        "--size_train_dataset",
        help="Number of samples to use from training data set. If not specified, use complete dataset",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-se",
        "--size_eval_dataset",
        help="Number of samples to use from evaluation data set. If not specified, use complete dataset",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-f",
        "--freeze_base_model",
        help="Freeze parameters of base model during training",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-lr", "--learning_rate", help="Learning rate", type=float, required=False
    )
    parser.add_argument(
        "-e",
        "--num_train_epochs",
        help="Number of training epochs",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-b",
        "--per_device_train_batch_size",
        help="Per device batch size",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-opt",
        "--optim",
        help="Optimizer (one of: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        help="Weight decay for AdamW optimizer",
        type=float,
        required=False,
    )
    parser.add_argument(
        "-b1",
        "--adam_beta1",
        help="beta1 hyperparameter for AdamW optimizer",
        type=float,
        required=False,
    )
    parser.add_argument(
        "-b2",
        "--adam_beta2",
        help="beta2 hyperparameter for AdamW optimizer",
        type=float,
        required=False,
    )
    parser.add_argument(
        "-eps",
        "--adam_epsilon",
        help="epsilon hyperparameter for AdamW optimizer",
        type=float,
        required=False,
    )
    parser.add_argument(
        "-log", "--logfile", help="Log file name and path", type=str, required=False
    )
    args = parser.parse_args()
    return vars(args)


def set_training_args(training_args, args):
    for argname, argvalue in args.items():
        if argvalue is not None:
            setattr(training_args, argname, argvalue)
    return training_args


class GPUMemoryCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(
            "GPU mem: Tot - ",
            torch.cuda.get_device_properties(0).total_memory,
            "res - ",
            torch.cuda.memory_reserved(0),
            "used - ",
            torch.cuda.memory_allocated(0),
        )


def get_free_gpu():
    for i in range(torch.cuda.device_count()):
        gpu_procs_str = torch.cuda.list_gpu_processes(i)
        if "no processes are running" in gpu_procs_str:
            return i
    return None


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Get Hydra arguments and apply hyperparaters to training arguments
    args = cfg.args

    # Logger setup
    if args["logfile"] is not None:
        logfile = args["logfile"]
    else:
        logfile = "basic_bert.log"
    logging.basicConfig(filename=logfile, level=logging.INFO)
    logger = logging.getLogger()

    # Get a GPU if available
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    logger.info("Compute device: %s", device)

    # Get hyperparameters
    training_args = TrainingArguments(
        output_dir="test_trainer", save_total_limit=2, evaluation_strategy="epoch"
    )
    training_args = set_training_args(training_args, args)
    print("Training arguments:", training_args)

    # Load a dataset
    dataset_name = "rotten_tomatoes"  # others: yelp_review_full, rotten_tomatoes
    logger.info("Dataset: %s", dataset_name)
    ds_builder = load_dataset_builder(dataset_name)
    num_classes = ds_builder.info.features["label"].num_classes
    dataset = load_dataset(dataset_name)

    # Set the name of the transformer model to use
    model_name = "bert-base-cased"

    # Create tokenizer and tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Use only a subset of the available data, if desired
    size_train_dataset = len(tokenized_datasets["train"])
    size_eval_dataset = len(tokenized_datasets["test"])
    if args["size_train_dataset"] is not None:
        size_train_dataset = args["size_train_dataset"]
    if args["size_eval_dataset"] is not None:
        size_eval_dataset = args["size_eval_dataset"]
    train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(size_train_dataset))
    )
    eval_dataset = (
        tokenized_datasets["test"].shuffle(seed=42).select(range(size_eval_dataset))
    )

    # Create classifier model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes
    )
    if args["freeze_base_model"]:
        logger.info("Freezing base model")
        for param in model.bert.parameters():
            param.requires_grad = False
    model.to(device)

    # Train the model
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(GPUMemoryCallback())

    trainer.train()

    # Evaluate model and report objective to Orion
    eval_metrics = trainer.evaluate()
    print("Final evaluation metrics:", eval_metrics)

    report_objective(1.0 - eval_metrics["eval_accuracy"], "error")

    # Print out memory stats
    print("Total GPU memory:", torch.cuda.get_device_properties(0).total_memory)
    print("GPU memory reserved:", torch.cuda.memory_reserved(0))
    print("GPU memory allocated:", torch.cuda.memory_allocated(0))


# =======================================================================
# Main
# =======================================================================
if __name__ == "__main__":
    main()

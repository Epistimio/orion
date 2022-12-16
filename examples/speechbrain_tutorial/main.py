import logging
import sys

import speechbrain as sb
import torch
from download_data import download
from speechbrain.utils.distributed import run_on_main
from train import ASR, dataio_prepare

from orion.client import report_objective

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print("Starting download")
    hparams = download(hparams_file, run_opts, overrides)
    print("finish download")
    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # In this case, pre-training is essential because mini-librispeech is not
    # big enough to train an end-to-end model from scratch. With bigger dataset
    # you can train from scratch and avoid this step.
    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=torch.device("cpu"))

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    #print("Starting fit")
    #asr_brain.fit(
    #    asr_brain.hparams.epoch_counter,
    #    datasets["train"],
    #    datasets["valid"],
    #    train_loader_kwargs=hparams["train_dataloader_opts"],
    #    valid_loader_kwargs=hparams["valid_dataloader_opts"],
    #)
    print("Starting evaluate")
    # Load best checkpoint for evaluation
    valid_stats = asr_brain.evaluate(
        test_set=datasets["valid"],
        min_key="WER",
        test_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    report_objective(valid_stats)

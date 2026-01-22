#!/usr/bin/env python3

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback, TrainingArguments

from trainer.base import Trainer


class CSVLogCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_log_filepath = None
        self.eval_log_filepath = None

    def on_log(self, args, state, control, model, **kwargs) -> None:
        if args.local_rank not in {-1, 0}:
            return

        if self.train_log_filepath is None:
            self.train_log_filepath = os.path.join(args.output_dir, "train_history.csv")

            with open(self.train_log_filepath, "a") as f:
                f.write("step,loss,lr\n")

        if self.eval_log_filepath is None:
            self.eval_log_filepath = os.path.join(args.output_dir, "eval_history.csv")

            with open(self.eval_log_filepath, "a") as f:
                f.write("step,loss,accuracy\n")

        is_eval = any("eval" in k for k in state.log_history[-1].keys())

        if is_eval:
            with open(self.eval_log_filepath, "a") as f:
                f.write(
                    "{},{},{}\n".format(
                        state.global_step,
                        state.log_history[-1]["eval_loss"],
                        state.log_history[-1]["eval_accuracy"]
                        if "eval_accuracy" in state.log_history[-1]
                        else np.nan,
                    )
                )

        else:
            with open(self.train_log_filepath, "a") as f:
                f.write(
                    "{},{},{}\n".format(
                        state.global_step,
                        state.log_history[-1]["loss"]
                        if "loss" in state.log_history[-1]
                        else state.log_history[-1]["train_loss"],
                        state.log_history[-1]["learning_rate"]
                        if "learning_rate" in state.log_history[-1]
                        else None,
                    )
                )


def _cat_data_collator(features: List) -> Dict[str, torch.tensor]:
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]

    return {
        k: torch.cat([f[k] for f in features])
        for k in features[0].keys()
        if not k.startswith("__")
    }


def decoding_accuracy_metrics(eval_preds, num_decoding_classes: int = None):
    """
    Compute accuracy metrics for decoding task.

    For binary classification (num_decoding_classes=2):
        - Model outputs [batch, 1] raw logits
        - Apply sigmoid + threshold to get binary predictions

    For multi-class classification (num_decoding_classes>2):
        - Model outputs [batch, num_classes] logits
        - Use argmax to get class predictions

    Args:
        eval_preds: Tuple of (predictions, labels)
        num_decoding_classes: Number of classes (default: infer from predictions)
    """
    preds, labels = eval_preds

    # If num_decoding_classes is not provided, infer from prediction shape
    if num_decoding_classes is None:
        num_decoding_classes = preds.shape[-1] if len(preds.shape) > 1 else 1

    # DEBUG OUTPUT
    import sys

    print(f"\n{'=' * 80}", file=sys.stderr)
    print("[METRICS COMPUTATION]", file=sys.stderr)
    print(f"num_decoding_classes: {num_decoding_classes}", file=sys.stderr)
    print(f"preds shape: {preds.shape}", file=sys.stderr)
    print(
        f"  min={preds.min():.6f}, max={preds.max():.6f}, mean={preds.mean():.6f}",
        file=sys.stderr,
    )
    print(f"labels shape: {labels.shape}", file=sys.stderr)
    print(f"  unique values: {np.unique(labels)}", file=sys.stderr)
    print(
        f"  distribution: {[(v, np.sum(labels == v)) for v in np.unique(labels)]}",
        file=sys.stderr,
    )
    print(f"preds sample (first 10): {preds.flatten()[:10]}", file=sys.stderr)
    print(f"labels sample (first 10): {labels[:10]}", file=sys.stderr)

    # Binary classification: apply sigmoid + threshold
    if num_decoding_classes == 2:
        # preds should be [batch, 1] from BCEWithLogitsLoss
        if len(preds.shape) > 1 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        # Apply sigmoid and threshold at 0.5
        sigmoid_vals = 1 / (1 + np.exp(-preds))
        preds_binary = (sigmoid_vals > 0.5).astype(int)

        print("BINARY CLASSIFICATION:", file=sys.stderr)
        print(f"  After squeeze shape: {preds.shape}", file=sys.stderr)
        print(
            f"  Sigmoid min={sigmoid_vals.min():.6f}, max={sigmoid_vals.max():.6f}, mean={sigmoid_vals.mean():.6f}",
            file=sys.stderr,
        )
        print(f"  Sigmoid sample (first 10): {sigmoid_vals[:10]}", file=sys.stderr)
        print(
            f"  Binary preds distribution: class_0={np.sum(preds_binary == 0)}, class_1={np.sum(preds_binary == 1)}",
            file=sys.stderr,
        )
        print(f"  Binary preds sample (first 10): {preds_binary[:10]}", file=sys.stderr)

        preds = preds_binary
    else:
        # Multi-class: use argmax
        preds = preds.argmax(axis=-1)
        print("MULTI-CLASS: Using argmax", file=sys.stderr)
        print(
            f"  Preds distribution: {[(v, np.sum(preds == v)) for v in np.unique(preds)]}",
            file=sys.stderr,
        )

    accuracy = accuracy_score(labels, preds)
    print(f"FINAL ACCURACY: {accuracy:.4f}", file=sys.stderr)
    print(f"{'=' * 80}\n", file=sys.stderr)

    return {"accuracy": round(accuracy, 3)}


def make_trainer(
    model_init,
    training_style,
    train_dataset,
    validation_dataset,
    do_train: bool = True,
    do_eval: bool = True,
    run_name: str = None,
    output_dir: str = None,
    overwrite_output_dir: bool = True,
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
        None,
        None,
    ),
    optim: str = "adamw_torch",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    per_device_train_batch_size: int = 64,
    per_device_eval_batch_size: int = 64,
    dataloader_num_workers: int = 0,
    max_steps: int = 400000,
    num_train_epochs: int = 1,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.01,
    evaluation_strategy: str = "steps",
    prediction_loss_only: bool = False,
    logging_strategy: str = "steps",
    save_strategy: str = "steps",
    save_total_limit: int = 5,
    save_steps: int = 10000,
    logging_steps: int = 10000,
    eval_steps: int = None,
    logging_first_step: bool = True,
    greater_is_better: bool = True,
    seed: int = 1,
    fp16: bool = True,
    deepspeed: str = None,
    num_decoding_classes: int = None,
    compute_metrics=None,
    **kwargs,
) -> Trainer:
    """
    Make a Trainer object for training a model.
    Returns an instance of transformers.Trainer.

    See the HuggingFace transformers documentation for more details
    on input arguments:
    https://huggingface.co/transformers/main_classes/trainer.html

    Custom arguments:
    ---
    model_init: callable
        A callable that does not require any arguments and
        returns model that is to be trained (see scripts.train.model_init)
    training_style: str
        The training style (ie., framework) to use.
        One of: 'BERT', 'CSM', 'NetBERT', 'autoencoder',
        'decoding'.
    train_dataset: src.batcher.dataset
        The training dataset, as generated by src.batcher.dataset
    validation_dataset: src.batcher.dataset
        The validation dataset, as generated by src.batcher.dataset

    Returns
    ----
    trainer: transformers.Trainer
    """
    trainer_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        do_train=do_train,
        do_eval=do_eval,
        overwrite_output_dir=overwrite_output_dir,
        prediction_loss_only=prediction_loss_only,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        optim=optim,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        greater_is_better=greater_is_better,
        save_steps=save_steps,
        logging_strategy=logging_strategy,
        logging_first_step=logging_first_step,
        logging_steps=logging_steps,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps if eval_steps is not None else logging_steps,
        seed=seed,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        deepspeed=deepspeed,
        **kwargs,
    )

    data_collator = _cat_data_collator
    is_deepspeed = deepspeed is not None
    # TODO: custom compute_metrics so far not working in multi-gpu setting
    if training_style == "decoding" and compute_metrics is None:
        # Create a closure that captures num_decoding_classes
        def compute_metrics_with_classes(eval_preds):
            return decoding_accuracy_metrics(
                eval_preds, num_decoding_classes=num_decoding_classes
            )

        compute_metrics = compute_metrics_with_classes

    trainer = Trainer(
        args=trainer_args,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        is_deepspeed=is_deepspeed,
    )

    trainer.add_callback(CSVLogCallback)

    return trainer

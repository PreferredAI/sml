import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import argparse
import datetime
import random
import yaml

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_utils import load_vocab, load_w2v, read_datasets
from eval_utils import prepare_eval_data, neg_sample, evaluate, filter_score
from models import SMLOppo, SMLFlex


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config.yaml")
parser.add_argument("--model", choices=["OPPO", "FLEX"], required=True)
parser.add_argument("--dataset", choices=["VSO", "Yelp"], required=True)
parser.add_argument("--testset", type=str, required=True)
parser.add_argument("--num_eval_samples", type=int, default=1000)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--shuffle_buffer", type=int, default=100000)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--latent_dim", type=int, default=300)
parser.add_argument("--act_fn", type=str, default="tanh")
parser.add_argument("--lambda_reg", type=float, default=0.0001)
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument("--verbose", action="store_true", default=True)


def set_seeds(seed=None):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def get_model(args, config):
    w2i, _ = load_vocab(args.dataset)
    word_emb, _ = load_w2v(args.dataset, w2i, seed=args.seed)

    if args.model == "OPPO":
        model = SMLOppo(
            word_emb,
            txt_len=config["TXT_LEN"],
            img_dim=config["IMG_DIM"],
            latent_dim=args.latent_dim,
            act_fn=args.act_fn,
            lambda_reg=args.lambda_reg,
            tau=args.tau,
        )
    elif args.model == "FLEX":
        model = SMLFlex(
            word_emb,
            txt_len=config["TXT_LEN"],
            img_dim=config["IMG_DIM"],
            latent_dim=args.latent_dim,
            act_fn=args.act_fn,
            lambda_reg=args.lambda_reg,
            tau=args.tau,
        )
    else:
        raise ValueError("Wrong model name!")

    return model


def get_writers(args):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = (
        f"logs/gradient_tape/{args.dataset}/{args.testset}/{args.model}/{current_time}/train"
    )
    test_log_dir = (
        f"logs/gradient_tape/{args.dataset}/{args.testset}/{args.model}/{current_time}/test"
    )
    train_writer = tf.summary.create_file_writer(train_log_dir)
    test_writer = tf.summary.create_file_writer(test_log_dir)
    return train_writer, test_writer


def main():
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))[args.dataset]

    set_seeds(args.seed)

    train_ds, test_ds = read_datasets(
        dataset=args.dataset,
        testset=args.testset,
        txt_len=config["TXT_LEN"],
        img_dim=config["IMG_DIM"],
        batch_size=args.batch_size,
        cache=True,
        shuffle_buffer=args.shuffle_buffer,
    )

    model = get_model(args, config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)

    # TensorBoard logging
    train_writer, test_writer = get_writers(args)

    # Prepare eval data
    img_feat, test_inds, test_labels = prepare_eval_data(test_ds, args.dataset, args.verbose)
    img_feat_tensor = tf.convert_to_tensor(img_feat, dtype=tf.float32)

    test_labels, test_candidates = neg_sample(
        labels=test_labels,
        num_samples=args.num_eval_samples,
        seed=args.seed,
        verbose=args.verbose,
    )

    best_loss = np.inf
    best_epoch = 0
    eval_results = {}
    for epoch in range(1, args.num_epochs + 1):
        # Training
        pbar = tqdm(
            train_ds, desc=f"Epoch {epoch}/{args.num_epochs}", disable=not args.verbose
        )
        for i, batch_x in enumerate(pbar):
            train_loss(compute_apply_gradients(model, batch_x, optimizer))
            if i % 20 == 0:
                pbar.set_postfix({"loss": f"{train_loss.result():.4f}"})

        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch - 1)

        # Evaluation
        scores = []
        pbar = tqdm(test_ds, "Testing", disable=not args.verbose)
        for i, batch_x in enumerate(pbar):
            scores.append(
                model.score(batch_x["text"], img_feat_tensor, batch_x["sentiment"]).numpy()
            )
            test_loss(model.compute_loss(batch_x))
            if i % 20 == 0:
                pbar.set_postfix({"loss": f"{test_loss.result():.4f}"})

        scores = np.concatenate(scores, axis=0)[test_inds]
        scores = filter_score(scores, test_candidates)

        eval_results[epoch] = evaluate(test_labels, scores)
        med_rank, percentile_rank, ndcg, recalls = eval_results[epoch]

        with test_writer.as_default():
            _loss = test_loss.result().numpy()
            tf.summary.scalar("loss", _loss, step=epoch - 1)
            tf.summary.scalar("metrics/MedR", med_rank, step=epoch - 1)
            tf.summary.scalar("metrics/PR", percentile_rank, step=epoch - 1)
            tf.summary.scalar("metrics/NDCG", ndcg, step=epoch - 1)
            for (k, recall) in recalls:
                tf.summary.scalar(f"metrics/R@{k}", recall, step=epoch - 1)

            if _loss <= best_loss:
                best_loss = _loss
                best_epoch = epoch
                model.save_weights(
                    f"./checkpoints/{args.dataset}/{args.testset}/{args.model}/ckpt"
                )

        # Reset metrics
        train_loss.reset_states()
        test_loss.reset_states()

    med_rank, percentile_rank, ndcg, recalls = eval_results[best_epoch]
    print(f"Best loss @ epoch {best_epoch}: {best_loss:.4f}")
    print(f"MedR = {med_rank:.1f}")
    print(f"PR = {percentile_rank * 100:.2f}")
    print(f"NDCG = {ndcg * 100:.2f}")
    for (k, recall) in recalls:
        print(f"Recall@{k} = {recall * 100:.2f}")


if __name__ == "__main__":
    main()

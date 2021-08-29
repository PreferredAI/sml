import glob
import os
import pickle

import gensim
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_vocab(dataset):
    vocab_path = f"./data/{dataset}/vocab.pkl"
    word2idx = pickle.load(open(vocab_path, "rb"))
    idx2word = {idx: word for (word, idx) in word2idx.items()}
    return word2idx, idx2word


def load_w2v(dataset, word2idx, seed=None):
    word_emb_path = f"./data/{dataset}/word_emb.npy"
    unk_words_path = f"./data/{dataset}/unk_words.pkl"

    try:
        word_emb = np.load(word_emb_path)
        unk_words = pickle.load(open(unk_words_path, "rb"))
        return word_emb, unk_words
    except:
        print("Word embeddings are not available!")

    print("Loading word2vec model")
    wv = gensim.downloader.load("word2vec-google-news-300")

    rng = np.random.RandomState(seed)
    word_emb = rng.randn(len(word2idx), 300) * 0.01
    unk_words = set()
    for w, i in word2idx.items():
        if w in wv.vocab:
            word_emb[i] = wv[w]
        else:
            unk_words.add(w)

    print("Unknown words:", len(unk_words))

    np.save(word_emb_path, word_emb)
    pickle.dump(unk_words, open(unk_words_path, "wb"))

    return word_emb, unk_words


def read_datasets(
    dataset, testset, txt_len, img_dim, batch_size=1, cache=False, shuffle_buffer=0
):
    def _parse_fn(example_proto):
        feature_description = {
            "sentiment": tf.io.FixedLenFeature([], tf.int64, default_value=0),
            "photo_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "img_feat": tf.io.FixedLenFeature(
                [img_dim],
                tf.float32,
                default_value=np.zeros(img_dim, dtype=np.float32),
            ),
            "text": tf.io.FixedLenFeature(
                [txt_len],
                tf.int64,
                default_value=np.zeros(txt_len, dtype=np.int64),
            ),
        }
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset_paths = sorted(filter(os.path.isdir, glob.glob(f"./data/{dataset}/*")))

    train_ds = tf.data.TFRecordDataset(
        [f"{path}/data.tfrecord" for path in dataset_paths if not path.endswith(testset)]
    ).map(_parse_fn, num_parallel_calls=AUTOTUNE)

    test_ds = tf.data.TFRecordDataset([f"./data/{dataset}/{testset}/data.tfrecord"]).map(
        _parse_fn, num_parallel_calls=AUTOTUNE
    )

    if cache:
        train_ds = train_ds.cache()
        test_ds = test_ds.cache()

    if shuffle_buffer:
        train_ds = train_ds.shuffle(buffer_size=shuffle_buffer)

    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, test_ds

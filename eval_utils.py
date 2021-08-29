from collections import defaultdict, OrderedDict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from tqdm import tqdm


def get_pos_rank(label, rank):
    pos_rank_list = []
    for i, rank_i in enumerate(rank):
        pos_label = label.indices[label.indptr[i] : label.indptr[i + 1]]
        pos_rank_list.append([np.argwhere(rank_i == l)[0][0] for l in pos_label])

    return pos_rank_list


def recall_score(label, rank, k=1):
    truncated_rank = rank[:, :k]

    row = np.repeat(np.arange(label.shape[0]), k)
    col = truncated_rank.ravel()
    pred = csr_matrix((np.ones(len(row)), (row, col)), label.shape, dtype=np.int)

    tp = label.multiply(pred).sum(axis=1)
    tp_fn = label.sum(axis=1)
    recall_arr = tp / tp_fn
    assert tp.shape == tp_fn.shape == recall_arr.shape

    return np.average(recall_arr)


def map_score(label, score):
    avg_prec_scores = []
    for i, score_i in enumerate(score):
        relevant = label.indices[label.indptr[i] : label.indptr[i + 1]]

        rank = rankdata(-score_i, "max")[relevant]
        L = rankdata(-score_i[relevant], "max")
        aux = (L / rank).mean()

        avg_prec_scores.append(aux)

    return np.average(avg_prec_scores)


def dcg_score(label, rank):
    """Only 1-D array"""
    relevance = np.take(label, rank)
    gain = 2 ** relevance - 1
    discounts = np.log2(np.arange(len(label)) + 2)

    return np.sum(gain / discounts)


def ndcg_score(label, rank):
    ideal_rank = np.argsort(-label, axis=1)
    dcg_arr = np.fromiter(
        (dcg_score(l, r) for l, r in zip(label, rank)), dtype=np.float, count=len(label)
    )
    idcg_arr = np.fromiter(
        (dcg_score(l, r) for l, r in zip(label, ideal_rank)),
        dtype=np.float,
        count=len(label),
    )

    return np.average(dcg_arr / idcg_arr)


def quick_ndcg_score(pos_rank_list):
    ndcg_scores = []
    for pos_rank in pos_rank_list:
        dcg = sum((1 / np.log2(r + 2) for r in pos_rank))
        idcg = sum((1 / np.log2(r + 2) for r in np.arange(len(pos_rank))))
        ndcg_scores.append(dcg / idcg)

    return np.average(ndcg_scores)


def evaluate(label, score):
    assert label.getformat() == "csr"
    assert label.shape == score.shape

    rank = np.argsort(-score, axis=1)
    pos_rank_list = get_pos_rank(label, rank)

    med_rank = np.median(
        [np.median([(r + 1) for r in pos_rank]) for pos_rank in pos_rank_list]
    )
    print(f"MedR = {med_rank:.1f}")

    percentile_rank = np.average(
        [
            np.average([(1.0 - (r + 1) / rank.shape[1]) for r in pos_rank])
            for pos_rank in pos_rank_list
        ]
    )
    print(f"PR = {percentile_rank * 100:.2f}")

    ndcg = quick_ndcg_score(pos_rank_list)
    print(f"NDCG = {ndcg * 100:.2f}")

    recalls = [(k, recall_score(label, rank, k)) for k in (5, 10)]
    for (k, recall) in recalls:
        print(f"Recall@{k} = {recall * 100:.2f}")

    return med_rank, percentile_rank, ndcg, recalls


def filter_score(score, candidates=None):
    if candidates == None:
        return score

    filtered_score = np.empty((score.shape[0], len(candidates[0])), dtype=np.float64)
    for i, c in enumerate(candidates):
        filtered_score[i] = score[i, c]

    return filtered_score


def prepare_eval_data(test_ds, dataset, verbose=False):
    X_test_inds = []
    text_id2idx = defaultdict()
    photo_id2feat = OrderedDict()
    text2photos = defaultdict(list)
    text_idx = -1
    for batch_data in tqdm(test_ds, "Loading test_ds", disable=not verbose):
        batch_text = batch_data["text"].numpy()
        batch_photo_ids = batch_data["photo_id"].numpy()
        batch_img_feat = batch_data["img_feat"].numpy()
        batch_sentiments = batch_data["sentiment"].numpy()
        for text, photo_id, feature, sentiment in zip(
            batch_text, batch_photo_ids, batch_img_feat, batch_sentiments
        ):
            if dataset == "VSO":
                text_idx += 1
            else:
                text_idx = text_id2idx.setdefault(str(text), len(text_id2idx))
            X_test_inds.append(text_idx)
            photo_id2feat[photo_id] = feature
            text2photos[text_idx].append(photo_id)

    img_feat = np.array(list(photo_id2feat.values()), dtype=np.float32)
    photo_id2idx = defaultdict()
    for photo_id in photo_id2feat:
        photo_id2idx.setdefault(photo_id, len(photo_id2idx))

    test_inds = np.empty(len(set(X_test_inds)), dtype=np.int)
    label_row, label_col = [], []
    selected_inds = set()
    for i, text_idx in enumerate(X_test_inds):
        if text_idx in selected_inds:
            continue
        selected_inds.add(text_idx)
        test_inds[text_idx] = i
        photo_inds = [photo_id2idx[photo_id] for photo_id in text2photos[text_idx]]
        label_col.extend(photo_inds)
        label_row.extend([text_idx] * len(photo_inds))

    test_labels = csr_matrix(
        (np.ones(len(label_row)), (label_row, label_col)),
        shape=(len(selected_inds), len(photo_id2idx)),
        dtype=np.int,
    )

    print("Test queries:", len(selected_inds))
    print("Test photos:", len(photo_id2idx))

    return img_feat, test_inds, test_labels


def neg_sample(labels, num_samples=1000, seed=None, verbose=False):
    if num_samples == -1:
        return labels, None, None

    assert labels.getformat() == "csr"

    rng = np.random.RandomState(seed)

    test_candidates = []
    row, col = [], []
    for i in tqdm(range(labels.shape[0]), "Negative sampling", disable=not verbose):
        label_i = labels.indices[labels.indptr[i] : labels.indptr[i + 1]]
        label_set = set(label_i)
        candidates = rng.choice(np.arange(labels.shape[1]), size=num_samples, replace=False)
        cand_set = set(candidates)
        for l in label_set:
            while l not in cand_set:
                rejected_idx = rng.randint(len(candidates))
                if candidates[rejected_idx] not in label_set:
                    candidates[rejected_idx] = l
                    cand_set.add(l)

        idx_map = {old: new for new, old in enumerate(candidates)}
        row.extend([i] * len(label_i))
        col.extend([idx_map[l] for l in label_i])
        test_candidates.append(candidates)

    test_labels = csr_matrix(
        (np.ones(len(row)), (row, col)), shape=(labels.shape[0], num_samples), dtype=np.int
    )

    return test_labels, test_candidates

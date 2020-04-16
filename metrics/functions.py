from scipy.stats.stats import pearsonr, rankdata, spearmanr
import numpy as np

def mean_absolute_error(y, y_hat):
    return np.mean(np.absolute(y_hat - y))


def mean_squared_error(y, y_hat):
    return np.square(np.subtract(y, y_hat)).mean()

def delta_average(y_true, y_rank):
    """Calculate the DeltaAvg score
    This is a much faster version than the Perl one provided in the
    WMT QE task 1.
    References: could not find any.
    Author: Fabio Kepler (contributed to MARMOT)
    Args:
        y_true: array of reference score (not rank) of each segment.
        y_rank: array of rank of each segment.
    Returns: the absolute delta average score.
    """
    sorted_ranked_indexes = np.argsort(y_rank)
    y_length = len(sorted_ranked_indexes)

    delta_avg = 0
    max_quantiles = y_length // 2
    set_value = (
        np.sum(y_true[sorted_ranked_indexes[np.arange(y_length)]]) / y_length
    )
    quantile_values = {
        head: np.sum(y_true[sorted_ranked_indexes[np.arange(head)]]) / head
        for head in range(2, y_length)
    }
    # Cache values, since there are many that are repeatedly computed
    # between various quantiles.
    for quantiles in range(2, max_quantiles + 1):  # Current number of quantiles
        quantile_length = y_length // quantiles
        quantile_sum = 0
        for head in np.arange(
            quantile_length, quantiles * quantile_length, quantile_length
        ):
            quantile_sum += quantile_values[head]
        delta_avg += quantile_sum / (quantiles - 1) - set_value

    if max_quantiles > 1:
        delta_avg /= max_quantiles - 1
    else:
        delta_avg = 0
    return abs(delta_avg)

def eval_sentence_level(sent_gold, sent_preds):
    scoring, ranking = score_sentence_level(sent_gold, sent_preds)
    scoring = np.array(
        scoring, 
        dtype=[("Pearson r", float), ("MAE", float), ("RMSE", float),],
    )

    ranking = np.array(
        ranking,
        dtype=[("Spearman r", float), ("DeltaAvg", float)],
    )
    return scoring, ranking

def score_sentence_level(gold, pred):
    pearson = pearsonr(gold, pred)
    mae = mean_absolute_error(gold, pred)
    rmse = np.sqrt(mean_squared_error(gold, pred))

    spearman = spearmanr(
        rankdata(gold, method="ordinal"), rankdata(pred, method="ordinal")        
    )
    delta_avg = delta_average(gold, rankdata(pred, method="ordinal"))
    return (pearson[0], mae, rmse), (spearman[0], delta_avg)

def print_sentences_scoring_table(scores):
    
    print("Sentence-level scoring:")
    print(
        "{:9}    {:9}    {:9}".format(
            "Pearson r",
            "MAE",
            "RMSE",
        )
    )
    print(scores)
    #print("{:<9.5f}    {:<9.5f}    {:<9.5f}".format(scores))

def print_sentences_ranking_table(scores):

    print("Sentence-level ranking:")
    print(
        "{:10}    {:9}".format(
            "Spearman r",
            "DeltaAvg",
        )
    )  # noqa
    print(scores)

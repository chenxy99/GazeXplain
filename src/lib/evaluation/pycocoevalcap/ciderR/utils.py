from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import numpy as np

from nltk.tokenize import RegexpTokenizer
from scipy.stats.mstats import gmean

def compute_penalty_by_length(candidate_len, reference_len, alpha=1.0):
    """
    :param candidate_len: candidate length
    :param reference_len: reference length
    :param alpha: parameter to adjust the penalty
    :return: penalty score by difference in length
    """
    delta = abs(reference_len-candidate_len)

    return np.e ** (-(delta ** 2) / (float(alpha) * float(reference_len) ** 2))


def compute_penalty_by_repetition(candidate_sent, reference_sent, penalty_func=lambda x: np.exp(-x)):
    """
    :param candidate_sent: candidate sentence
    :param reference_sent: reference sentence
    :param penalty_func: penalty function
    :return: penalty score by repetition
    """
    # tokenize only words
    tokenizer = RegexpTokenizer(r'\w+')
    tokens_candidate = tokenizer.tokenize(candidate_sent)
    tokens_reference = tokenizer.tokenize(reference_sent)

    word_freq_candidate = Counter(tokens_candidate)
    word_freq_reference = Counter(tokens_reference)

    scores = []

    for word, freq in word_freq_candidate.items():
        # words in the reference and in the hypothesis
        if word_freq_reference.get(word, None) is not None:
            diff = abs(word_freq_reference[word] - freq)
            scores.append(penalty_func(diff))
        else:
            # words in the hypothesis but not in the reference
            scores.append(penalty_func(freq - 1))
    
    return gmean(np.array(scores))


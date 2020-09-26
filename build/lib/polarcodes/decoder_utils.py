import numpy as np
from polarcodes.utils import *

def hard_decision(y):
    """
        Hard decision of a log-likelihood.
    """

    if y >= 0:
        return 0
    else:
        return 1

def upper_llr(l1, l2):
    """
    Update top branch LLR in the log-domain.
    This function supports shortening by checking for infinite LLR cases.

    Parameters
    ----------
    l1: float
        input LLR corresponding to the top branch
    l2: float
        input LLR corresponding to the bottom branch

    Returns
    ----------
    float
        the top branch LLR at the next stage of the decoding tree

    """

    # check for infinite LLR, used in shortening
    if l1 == np.inf and l2 != np.inf:
        return l2
    elif l1 != np.inf and l2 == np.inf:
        return l1
    elif l1 == np.inf and l2 == np.inf:
        return np.inf
    else:  # principal decoding equation
        return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)

def lower_llr(l1, l2, b):
    """
    Update bottom branch LLR in the log-domain.
    This function supports shortening by checking for infinite LLR cases.

    Parameters
    ----------
    l1: float
        input LLR corresponding to the top branch
    l2: float
        input LLR corresponding to the bottom branch
    b: int
        the decoded bit of the top branch

    Returns
    ----------
    float, np.nan
        the bottom branch LLR at the next stage of the decoding tree
    """

    if b == 0:  # check for infinite LLRs, used in shortening
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        else:  # principal decoding equation
            return l1 + l2
    elif b == 1:  # principal decoding equation
        return l1 - l2
    return np.nan

def active_llr_level(i, n):
    """
        Find the first 1 in the binary expansion of i.
    """

    mask = 2**(n-1)
    count = 1
    for k in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)

def active_bit_level(i, n):
    """
        Find the first 0 in the binary expansion of i.
    """

    mask = 2**(n-1)
    count = 1
    for k in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)

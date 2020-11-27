import numpy as np


def sample_vectors(p, num):

    return np.random.binomial(1, p, size=(num, len(p))).astype(np.float32)


def num_matches(x, y):

    return np.sum(x == y[None, :], axis=1)


def get_new_p(x, m):

    num = np.sum(m[:, None] * x, axis=0)
    den = np.sum(m[:, None], axis=0)

    return num / den


def main():
    # one-phase (this one) takes 60 - 70 steps to converge
    # two-phase (E2.py) only takes 4 steps

    p = np.array([1/2] * 10, dtype=np.float32)
    y = np.array([1] * 5 + [0] * 5, dtype=np.float32)
    num_samples = 50
    num_steps = 100

    print("t=0: p={:s}".format(str(p)))
    for i in range(num_steps):
        x = sample_vectors(p, num_samples)
        m = num_matches(x, y)
        p = get_new_p(x, m)
        print("t={:d}: p={:s}".format(i + 1, str(p)))


main()

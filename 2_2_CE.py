import numpy as np


def sample_vectors(p, num):

    return np.random.binomial(1, p, size=(num, len(p))).astype(np.float32)


def num_matches(x, y):

    return np.sum(x == y[None, :], axis=1)


def sort_matches(p):

    return np.sort(p)


def quantile_index(num_samples, rho):

    return int(np.ceil((1 - rho) * num_samples))


def get_new_p(x, m, gamma_hat):

    above_gamma = (m >= gamma_hat).astype(np.float32)

    num = np.sum(above_gamma[:, None] * x, axis=0)
    den = np.sum(above_gamma[:, None], axis=0)

    return num / den


def main():

    p = np.array([1/2] * 10, dtype=np.float32)
    y = np.array([1] * 5 + [0] * 5, dtype=np.float32)
    rho = 0.1
    num_samples = 50
    index = quantile_index(num_samples, rho)
    gamma = len(y)
    num_steps = 4

    print("t=0: p={:s}".format(str(p)))
    for i in range(num_steps):
        x = sample_vectors(p, num_samples)
        m = num_matches(x, y)
        s = sort_matches(m)
        gamma_hat = s[index]
        p = get_new_p(x, m, gamma_hat)
        print("t={:d}: gamma_hat={:.3f}, p={:s}".format(i + 1, gamma_hat, str(p)))


main()

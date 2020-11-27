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


def update_p(p_old, p_new, alpha):

    return alpha * p_new + (1 - alpha) * p_old


def main():
    # alpha=0.5 always converges within 50 steps
    # alpha=0.9, or alpha=1.0 sometimes gets stuck at gamma_hat \in [47, 49]

    p = np.array([1/2] * 50, dtype=np.float32)
    y = np.array([1] * 25 + [0] * 25, dtype=np.float32)
    rho = 0.1
    num_samples = 50
    index = quantile_index(num_samples, rho)
    gamma = len(y)
    num_steps = 50
    alpha = 0.5

    print("t=0: p={:s}".format(str(p)))
    for i in range(num_steps):
        x = sample_vectors(p, num_samples)
        m = num_matches(x, y)
        s = sort_matches(m)
        gamma_hat = s[index]
        p_new = get_new_p(x, m, gamma_hat)
        p = update_p(p, p_new, alpha)
        print("t={:d}: gamma_hat={:.3f}, p={:s}".format(i + 1, gamma_hat, str(p)))


main()

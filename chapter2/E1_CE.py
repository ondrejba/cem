import numpy as np
from scipy import stats


def sample_edges(u, num):

    return np.random.exponential(u, size=(num, len(u)))


def pdf(x, u):

    loc = np.zeros_like(u)
    scale = u

    return np.prod(stats.expon.pdf(x, loc, scale), axis=1)


def get_weights(x, u, v):

    return pdf(x, u) / pdf(x, v)


def shortest_path(x):

    return np.min(
        [x[:, 0] + x[:, 3],
         x[:, 1] + x[:, 4],
         x[:, 0] + x[:, 2] + x[:, 4],
         x[:, 1] + x[:, 2] + x[:, 3]], axis=0)


def sort_paths(p):

    return np.sort(p)


def quantile_index(num_samples, rho):

    return int(np.ceil((1 - rho) * num_samples))


def get_gamma_hat(ps, index, gamma):

    tmp = ps[index]

    if tmp < gamma:
        return tmp
    else:
        return gamma


def get_new_v(x, p, u, v, gamma_hat):

    above_gamma = (p >= gamma_hat).astype(np.float32)
    weights = get_weights(x, u, v)

    num = np.sum(above_gamma[:, None] * weights[:, None] * x, axis=0)
    den = np.sum(above_gamma[:, None] * weights[:, None], axis=0)

    return num / den


def main():

    u = np.array([0.25, 0.4, 0.1, 0.3, 0.2], dtype=np.float32)
    gamma = 2
    num_samples = 1000
    num_steps = 5
    rho = 0.1

    v = u
    index = quantile_index(num_samples, rho)

    print("t=0: v={:s}".format(str(v)))
    for i in range(num_steps):
        x = sample_edges(v, num_samples)
        p = shortest_path(x)
        gamma_hat = get_gamma_hat(sort_paths(p), index, gamma)
        v = get_new_v(x, p, u, v, gamma_hat)
        print("t={:d}: gamma_hat={:.3f}, v={:s}".format(i + 1, gamma_hat, str(v)))


main()

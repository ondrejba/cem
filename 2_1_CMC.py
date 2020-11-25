import argparse
import numpy as np


def sample_edges(u, num):

    return np.random.exponential(u, size=(num, len(u)))


def shortest_path(x):

    return np.min(
        [x[:, 0] + x[:, 3],
         x[:, 1] + x[:, 4],
         x[:, 0] + x[:, 2] + x[:, 4],
         x[:, 1] + x[:, 2] + x[:, 3]], axis=0)


def fraction_above_gamma(p, gamma):

    return np.sum(p >= gamma) / len(p)


def block_compute(u, gamma, num_samples):

    x = sample_edges(u, num_samples)
    p = shortest_path(x)
    f = fraction_above_gamma(p, gamma)
    return f


def main(num_samples):

    u = np.array([0.25, 0.4, 0.1, 0.3, 0.2], dtype=np.float32)
    gamma = 2
    max_block_size = int(1e+6)

    num_steps = int(np.ceil(num_samples / max_block_size))

    if num_steps == 1:
        f = block_compute(u, gamma, num_samples)
    else:
        f = 0

        for i in range(num_steps):

            if i < num_steps - 1:
                tmp_num = max_block_size
            else:
                tmp_num = num_samples - max_block_size * (num_steps - 1)

            tmp_f = block_compute(u, gamma, tmp_num)
            f += tmp_f * (tmp_num / num_samples)

    print(f)


parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parsed = parser.parse_args()
main(parsed.num_samples)

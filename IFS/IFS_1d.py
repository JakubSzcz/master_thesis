import random
import util.math as mymath

# parameters
# N_SAMPLES = 50000  # samples in base signal
# BLOCK_SIZE = 10  # size of range block (domain block = *2)
# N_DOMAINS = 1000  # number of Domains blocks
# N_RANGE = int(N_SAMPLES / BLOCK_SIZE)  # number of range block
DEC_ITERATIONS = 1  # iterations number while decoding
ENC_ITERATIONS = 1  # iterations number while encoding
D_THRESHOLD = 0.0001


def generate_range_domain_blocks(range_block_size: int, n_domains: int, signal, domains_overlaps: bool = True) -> (
        list, list, list):
    """
    Generates range blocks (covering whole signal without overlapping) and domain blocks (not whole signal covered without or with overlapping)
    :param range_block_size: size of the single range block (domain block is 2 times larger)
    :param n_domains: number of domain blocks to be generated
    :param signal: signal from which blocks are generated
    :param domains_overlaps: tells if domain blocks are overlapping, default is True
    :return: list of range, domain blocks and list of starting sample number of each domain block
    """

    # parameters, initialization and validation
    n_samples = len(signal)
    if n_samples % range_block_size != 0:
        raise ValueError("Signal size must be a multiple of range_block_size.")

    assert range_block_size > 0, "Range block size must be a positive integer."
    assert n_domains > 0, "Number of domain blocks must be a positive integer."
    assert n_domains < n_samples - (range_block_size * 2), "To many domains number provided."
    assert len(signal) > 0, "Empty signal provided."

    # range blocks generation
    range_blocks = [signal[i:i + range_block_size] for i in range(0, n_samples, range_block_size)]

    domain_blocks = []
    starting_sample = []
    # domain blocks generation
    if not domains_overlaps:
        for i in range(0, n_samples, (range_block_size * 2)):
            domain_blocks.append(signal[i:i + (range_block_size * 2)])
            starting_sample.append(i)
    else:
        for _ in range(n_domains):
            # take only such index that haven't been already used
            while True:
                ind = random.randint(0, n_samples - (range_block_size * 2))
                if ind not in starting_sample:
                    domain_blocks.append(signal[ind:ind + range_block_size * 2])
                    starting_sample.append(ind)
                    break

    return range_blocks, domain_blocks, starting_sample


def encode(ranges: list, domains: list, domains_starting_sample: list) -> list:
    """
    Encodes parameters of affine transformation (starting sample of chosen domain block, alpha and beta) for each
    range block from the selected domain block based on the minimal RMS after transformation
    :param ranges: list of range blocks to encode
    :param domains: list of domain blocks to encode from
    :param domains_starting_sample: list of starting sample number of each domain block
    :return: list of tuples of encoded parameters for each range block: (domain_starting_sample, alpha, beta)
    """

    # parameters and validations
    n_range = len(ranges)
    n_domains = len(domains)
    n_start_samples = len(domains_starting_sample)
    d_unique = set()
    codded = []

    assert n_range > 0, "No range blocks provided."
    assert n_domains > 0, "No domain blocks provided."
    assert n_start_samples > 0, "No domains starting samples list provided."
    assert n_domains == n_start_samples, "Domain blocks and starting samples does not match."

    # find the best alpha + beta for each range block
    for r_i, r in enumerate(ranges):
        # progress logging
        if r_i % 5 == 0:
            print(f"{round(r_i * 100 / n_range, 2)}%.")

        # parameters to encode
        d_rms_min = 1000000
        d_starting_sample = 0
        fit_alpha = 1
        fit_beta = 0

        # find the best base domain from domain pool to transform into range block with min d_rms
        for d_i, d in enumerate(domains):
            d_down = mymath.downsample(d)
            alpha, beta = mymath.calculate_alpha_beta(d_down, r)
            transformed = d_down
            for _ in range(ENC_ITERATIONS):
                transformed = mymath.transform(alpha, beta, transformed)
            d_rms_cal = mymath.d_rms(d_down, transformed)

            if d_rms_cal < d_rms_min:
                d_starting_sample = domains_starting_sample[d_i]
                d_unique.add(d_i)
                fit_alpha = alpha
                fit_beta = beta
                d_rms_min = d_rms_cal

            # already found d_rms satisfies threshold, stop searching
            if d_rms_min < D_THRESHOLD:
                break

        # encoded parameters for each range block
        codded.append((d_starting_sample, fit_alpha, fit_beta))

    print(f"Number of unique d used in the encoding process: {len(d_unique)}/{n_domains}")
    return codded

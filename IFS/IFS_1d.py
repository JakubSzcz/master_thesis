import random
import util.math as mymath
import numpy as np
import time


class IFS:
    # parameters default values
    RANGE_BLOCK_SIZE = 10  # size of range block (domain block = *2)
    DEC_ITERATIONS = 10  # iterations number while decoding
    D_THRESHOLD = 0.0001  # threshold over which we consider transformation to be sufficient
    TIME_ROUNDING = 5

    def setup_values(self, range_block_size: int, dec_iterations: int, d_threshold: float):
        self.RANGE_BLOCK_SIZE = range_block_size
        self.DEC_ITERATIONS = dec_iterations
        self.D_THRESHOLD = d_threshold

    def generate_range_domain_blocks(self, n_domains: int, signal, domains_overlaps: bool = True) -> (
            list, list, list):
        """
        Generates range blocks (covering whole signal without overlapping) and
        domain blocks (not whole signal covered without or with overlapping)
        :param n_domains: number of domain blocks to be generated
        :param signal: signal from which blocks are generated
        :param domains_overlaps: tells if domain blocks are overlapping, default is True
        :return: list of range, domain blocks and list of starting sample number of each domain block
        """

        start_time = time.time()
        print("starting generating range and domain blocks...")
        # parameters, initialization and validation
        n_samples = len(signal)
        if n_samples % self.RANGE_BLOCK_SIZE != 0:
            raise ValueError("Signal size must be a multiple of range_block_size.")

        assert self.RANGE_BLOCK_SIZE > 0, "Range block size must be a positive integer."
        assert n_domains > 0, "Number of domain blocks must be a positive integer."
        assert n_domains < n_samples - (self.RANGE_BLOCK_SIZE * 2), "To many domains number provided."
        assert len(signal) > 0, "Empty signal provided."

        # range blocks generation
        range_blocks = [signal[i:i + self.RANGE_BLOCK_SIZE] for i in range(0, n_samples, self.RANGE_BLOCK_SIZE)]

        domain_blocks = []
        starting_sample = []
        # domain blocks generation
        if not domains_overlaps:
            for i in range(0, n_samples, (self.RANGE_BLOCK_SIZE * 2)):
                domain_blocks.append(signal[i:i + (self.RANGE_BLOCK_SIZE * 2)])
                starting_sample.append(i)
        else:
            for _ in range(n_domains):
                # take only such index that haven't been already used
                while True:
                    ind = random.randint(0, n_samples - (self.RANGE_BLOCK_SIZE * 2))
                    if ind not in starting_sample:
                        domain_blocks.append(signal[ind:ind + self.RANGE_BLOCK_SIZE * 2])
                        starting_sample.append(ind)
                        break
        print(f"generating range and domain blocks finished with "
              f"{round(time.time() - start_time, self.TIME_ROUNDING)} seconds..")
        return range_blocks, domain_blocks, starting_sample

    def encode(self, ranges: list, domains: list, domains_starting_sample: list) -> list:
        """
        Encodes parameters of affine transformation (starting sample of chosen domain block, alpha and beta) for each
        range block from the selected domain block based on the minimal RMS after transformation
        :param ranges: list of range blocks to encode
        :param domains: list of domain blocks to encode from
        :param domains_starting_sample: list of starting sample number of each domain block
        :return: list of tuples of encoded parameters for each range block: (domain_starting_sample, alpha, beta)
        """

        print("starting encoding...")
        start_time = time.time()
        # parameters and validations
        n_range = len(ranges)
        n_domains = len(domains)
        n_start_samples = len(domains_starting_sample)
        progress_incrementor = int(0.05 * n_range)  # for logging purpose
        d_unique = set()
        codded = []

        assert n_range > 0, "No range blocks provided."
        assert n_domains > 0, "No domain blocks provided."
        assert n_start_samples > 0, "No domains starting samples list provided."
        assert n_domains == n_start_samples, "Domain blocks and starting samples does not match."

        # find the best alpha + beta for each range block
        for r_i, r in enumerate(ranges):
            # progress logging
            if r_i % progress_incrementor == 0:
                print(f"\rProgress: {round(r_i * 100 / n_range, 2)}%.", end="", flush=True)

            # parameters to encode
            d_rms_min = 1000000
            d_starting_sample = 0
            fit_alpha = 1
            fit_beta = 0

            # find the best base domain from domain pool to transform into range block with min d_rms
            for d_i, d in enumerate(domains):
                d_down = mymath.downsample(d)
                alpha, beta = mymath.calculate_alpha_beta(d_down, r)
                transformed = mymath.transform(alpha, beta, d_down)
                d_rms_cal = mymath.d_rms(d_down, transformed)

                if d_rms_cal < d_rms_min:
                    d_starting_sample = domains_starting_sample[d_i]
                    d_unique.add(d_i)
                    fit_alpha = alpha
                    fit_beta = beta
                    d_rms_min = d_rms_cal

                # already found d_rms satisfies threshold, stop searching
                if d_rms_min < self.D_THRESHOLD:
                    break

            # encoded parameters for each range block
            codded.append((d_starting_sample, fit_alpha, fit_beta))

        print(f"\rProgress: 100%.", flush=True)
        print(f"encoding finished with {round(time.time() - start_time, self.TIME_ROUNDING)} seconds.")
        print(f"Number of unique d used in the encoding process: {len(d_unique)}/{n_domains}")
        return codded

    def decode(self, encoded_parameters: list) -> list:

        """
        Decodes from random noise using IFS based on the parameters from the encoding process
         until close to original signal attractor is generated
        :param encoded_parameters: list of tuples of encoded parameters for each range block:
            (domain_starting_sample, alpha, beta)
        :return: attractor as a reconstructed signal close to the original signal
        """

        start_time = time.time()
        print("starting decoding...")
        # parameters
        n_range = len(encoded_parameters)
        n_samples = n_range * self.RANGE_BLOCK_SIZE
        random_vector = np.random.uniform(0, 1, n_samples)

        # prepare base random vector for reconstruction
        decoded = [random_vector[i:i + self.RANGE_BLOCK_SIZE] for i in range(0, n_samples, self.RANGE_BLOCK_SIZE)]
        # iteratively perform transformation for each range blocks
        # TODO iterations should be done as long as: to many iterations performed or
        #  d_rms between 2 consecutive transformation < D_THRESHOLD
        for _ in range(self.DEC_ITERATIONS):
            temp = np.array(decoded).flatten().tolist()
            for ind, w in enumerate(encoded_parameters):
                decoded[ind] = mymath.transform(w[1], w[2],
                                                mymath.downsample(temp[w[0]:w[0] + (self.RANGE_BLOCK_SIZE * 2)]))

        print(f"decoding finished with {round(time.time() - start_time, self.TIME_ROUNDING)} seconds.")
        return np.array(decoded).flatten().tolist()

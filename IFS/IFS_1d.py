
# parameters
N_SAMPLES = 50000 # samples in base signal
BLOCK_SIZE = 10 # size of range block (domain block = *2)
N_DOMAINS = 1000 # number of Domains blocks
N_RANGE = int(N_SAMPLES / BLOCK_SIZE) # number of range block
DEC_ITERATIONS = 1 # iterations number while decoding
ENC_ITERATIONS = 1 # iterations number while encoding
D_THRESHOLD = 0.01
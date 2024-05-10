from generate_data import *
from compress_data import *
import string

charset = "abc"
#charset = string.ascii_lowercase
string_lengths = [
    10, 100, 1000, 10000, 100_000, #1_000_000,
]
# print("compression ratio | str len | gzipped len")
# for string_length in string_lengths:
    
#     s = generate_rules_string(charset, length=string_length)

#     gz = gzip_string(s, gzipbytes=True)

#     slen = len(s)
#     gzlen = len(gz)
    
#     print(f"{100*gzlen/slen:>8.0f}% {slen:>8} {gzlen:>8}")



import numpy as np
import random
import sys

def generate_random_string(length=10):
    """ Generate a random string of a given length """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choice(letters) for i in range(length))

def string_to_numbers(s):
    """ Convert string to a list of numbers based on character ordinal values """
    return [ord(char) for char in s]

def compute_dft(data):
    """ Compute the Discrete Fourier Transform of a list of numbers """
    return np.fft.fft(data)

def main():
    # Generate a random string
    random_string = generate_rules_string(charset, string_lengths[-1])
    print("Random String:", random_string)
    
    # Convert string to numbers
    numerical_data = string_to_numbers(random_string)
    print("Numerical Representation:", numerical_data)
    
    # Compute DFT
    dft_result = compute_dft(numerical_data)
    print("DFT Result:", dft_result)
    
    # Compare sizes
    original_size = sys.getsizeof(random_string)
    dft_size = sys.getsizeof(dft_result)
    print("Size of Original String (bytes):", original_size)
    print("Size of DFT Result (bytes):", dft_size)

if __name__ == "__main__":
    main()
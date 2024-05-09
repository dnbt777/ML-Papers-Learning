from generate_data import *
from compress_data import *
import string

charset = "abc"
#charset = string.ascii_lowercase
string_lengths = [
    10, 100, 1000, 10000, 100_000, #1_000_000,
]
print("compression ratio | str len | gzipped len")
for string_length in string_lengths:
    
    s = generate_rules_string(charset, length=string_length)

    gz = gzip_string(s, gzipbytes=True)

    slen = len(s)
    gzlen = len(gz)
    
    print(f"{100*gzlen/slen:>8.0f}% {slen:>8} {gzlen:>8}")

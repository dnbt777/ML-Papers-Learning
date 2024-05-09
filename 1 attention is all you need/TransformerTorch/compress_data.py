import gzip
import base64

def compress_string(s):
    compressed = ""
    i = 0
    while i < len(s):
        count = 1
        while i + 1 < len(s) and s[i] == s[i+1]:
            i += 1
            count += 1
        if count == 1:
            compressed += s[i]
        elif count == 2:
            compressed += s[i].upper()
        else:
            compressed += s[i] + str(count)
        i += 1
    return compressed



def gzip_string(s, gzipbytes=True):
    # Convert the input string to bytes
    input_bytes = s.encode('utf-8')
    
    # Create a buffer to hold the compressed data
    buf = gzip.compress(input_bytes)
    
    # Encode the compressed data in base64 to make it a readable string
    base64_encoded = base64.b64encode(buf).decode('utf-8')
    if gzipbytes:
        return buf
    return base64_encoded
    

def main():
    s = "asdojiasdoisdj712387931278913278931297813297801327809132ddsuasdu89asdu89dsaasdjnadsjndsajdsajdsajasdjidasijdsanjidsanijsdanijosdanijodsa"
    gs = gzip_string(s)
    print(gs)

if __name__ == "__main__":
    main()
import collections
c = collections.Counter(open('data/cleaned.txt', 'r').read())

allowed_chars = []
for key in c:
    if c[key] > 300 and not key.isupper() and key != '\n':
        allowed_chars.append(key)
with open('data/allowed_chars.txt', 'w') as f:
    f.write("".join(allowed_chars))


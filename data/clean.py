from sys import argv, stdin

if __name__ == "__main__":
    if len(argv) < 2:
        print "USAGE:", argv[0], "allowed_chars.txt < data.txt"
    allowed = open(argv[1]).read()

    for line in stdin:
        print ''.join(filter(lambda x: x in allowed, line.lower()))

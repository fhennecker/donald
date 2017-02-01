import json

jsonfiles = [str(i) for i in range(2015,2018)]
outfile = open('data/cleaned.txt', 'w')

for jsonfile in jsonfiles:
    data = json.loads(open('data/'+jsonfile+'.json', 'r').read())
    for tweet in data:
        outfile.write(tweet['text']+'\n')


    


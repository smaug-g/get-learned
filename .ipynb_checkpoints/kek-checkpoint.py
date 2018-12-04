regex = re.compile('[^a-zA-Z0-9]')

def titleToList(title):
    return regex.sub(' ', title).lower().split()

# id_list = (id, [list of words in title])
id_list = list(map(lambda x: (x[0], titleToList(x[1])) , train_id_title))

allWords = [(word, 1) for title in id_list for word in title[1]]        

getFirst = lambda x: x[0]
        
allWords.sort(key=getFirst)

def reducer(word1count, word2count):
    return (word1count[0], word1count[1] + word2count[1])

# (word, count)
allCounts = list(reduce(reducer, group) for _,group in groupby(allWords, key=getFirst))

sortedCounts = sorted(allCounts, key=lambda x: x[1], reverse=True)

len(sortedCounts)
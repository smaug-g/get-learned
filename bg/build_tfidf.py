import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_nonimage_data(csvName, test=False, csvSize=3094, vectorizer = None):
    # all labels
    y = []

    # will have (id, title)
    train_id_title = []

    # contains imdbid, year, runtime, score
    otherData = np.zeros([csvSize, 4])

    with open(csvName) as trainCSV:
        reader = csv.reader(trainCSV, delimiter=',')
        i = 0
        for row in reader:
            # skip the first row
            if row[1] == 'imdbId':
                continue
            train_id_title.append((id, row[2]))
            if not test:
                y.append(int(row[6]))
            otherData[i, :] = np.array([float(row[1]), float(row[3]), float(row[4]), float(row[5])])
            i += 1

    y = np.array(y)

    # we're treating titles as TF-IDF vectors
    titles = [title[1] for title in train_id_title]
    if not vectorizer:
        vectorizer = TfidfVectorizer(strip_accents='unicode')
        Xtfidf = np.asarray(vectorizer.fit_transform(titles).todense())
    else:
        Xtfidf = np.asarray(vectorizer.transform(titles).todense())

    return Xtfidf, otherData, y, vectorizer
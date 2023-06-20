import numpy as np
import pandas as pd

import json
from youtubesearchpython import SearchVideos

def getMusicName(elem):
    return '{} - {}'.format(elem['artist'], elem['song_title'])


# Function to search a YouTube Video
def youtubeSearchVideo(music, results=1):
    searchJson = SearchVideos(music, offset=1, mode="json", max_results=results).result()
    searchParsed = json.loads(searchJson)
    searchParsed = searchParsed['search_result'][0]
    return {'title': searchParsed['title'], \
            'duration': searchParsed['duration'], \
            'views': searchParsed['views'], \
            'url': searchParsed['link'] }



dfSongs = pd.read_csv('archive/data.csv', index_col=0)
rows, cols = dfSongs.shape
print('Number of songs: {}'.format(rows))

#Each song has 16 Atributes
print('Number of attributes per song: {}'.format(cols))

# Song information about 
#print(dfSongs.info())

# First songs
print(dfSongs[['song_title', 'artist']].head(5))

# Select a song
anySong = dfSongs.loc[0]
# Get the song name
anySongName = getMusicName(anySong)
print('name:', anySongName)

# Search in YouTube
pesquisa = youtubeSearchVideo(anySongName)

# K-query
def knnQuery(queryPoint, arrCharactPoints, k):
    tmp = arrCharactPoints.copy(deep=True)
    tmp['dist'] = tmp.apply(lambda x: np.linalg.norm(x-queryPoint), axis=1)
    tmp = tmp.sort_values('dist')
    return tmp.head(k).index

# Range query
def rangeQuery(queryPoint, arrCharactPoints, radius):
    tmp = arrCharactPoints.copy(deep=True)
    tmp['dist'] = tmp.apply(lambda x: np.linalg.norm(x-queryPoint), axis=1)
    tmp['radius'] = tmp.apply(lambda x: 1 if x['dist'] <= radius else 0, axis=1)
    return tmp.query('radius == 1').index


# Execute k-NN removing the 'query point'
def querySimilars(df, columns, idx, func, param):
    arr = df[columns].copy(deep=True)
    queryPoint = arr.loc[idx]
    arr = arr.drop([idx])
    response = func(queryPoint, arr, param)
    return response




# Selecting song and attributes
songIndex = 1936 # query point, selected song
columns = ['acousticness','danceability','energy','instrumentalness','liveness','speechiness','valence']

# Selecting query parameters
func, param = knnQuery, 3 # k=3

# Querying
response = querySimilars(dfSongs, columns, songIndex, func, param)




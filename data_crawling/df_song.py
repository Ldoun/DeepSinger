import pandas as pd
from pprint import pprint as pp

appropriate = []
songs = pd.read_csv('./song2.csv')
artist_counter = {}
for i in range(len(songs)):
    try:
        artist_counter[songs.loc[i]['artist']] += 1
    except:
        artist_counter[songs.loc[i]['artist']] = 1

song_count = 0
for key,value in artist_counter.items():
    if value > 10:
        appropriate.append(key)
        print(str(value) + " " + key)
        song_count += value

print("all: " + str(song_count))

data_list = songs.loc[appropriate]

print(data_list)
data_list.to_csv('data_list.csv')
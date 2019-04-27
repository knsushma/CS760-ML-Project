import pandas as pd
import csv
import re

# Collecting Track IDs
file_features = "./essentia_features.csv"
with open(file_features, mode='r') as infile:
    records = csv.reader(infile)
    track_ids = [rows[-1] for rows in records]


#Genre-top Extraction
file_track = "./fma_metadata/tracks.csv"
with open(file_track, mode='r') as infile:
    reader = csv.reader(infile)
    track_genre_map = {rows[0]:rows[40] for rows in reader}

file_writer = "./essentia_features_genre_top.csv"
with open(file_writer, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for track_id in track_ids:
       writer.writerow([track_id, track_genre_map[track_id]])



# Artist Extraction
file_track = "./fma_metadata/tracks.csv"
with open(file_track, mode='r') as infile:
    reader = csv.reader(infile)
    track_artist_map = {rows[0]:rows[26] for rows in reader}

file_writer = "./essentia_features_artist.csv"
with open(file_writer, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for track_id in track_ids:
       writer.writerow([track_id, track_artist_map[track_id]])

def is_not_blank(s):
    return bool(s and s.strip())


#Year Extraction
file_track = "./fma_metadata/tracks.csv"
with open(file_track, mode='r') as infile:
    reader = csv.reader(infile)
    header1 = next(reader)
    header2 = next(reader)
    header3 = next(reader)
    track_year_map = {row[0]:row[3] for row in reader}

result = {}
for key, value in track_year_map.items():
    try:
        result[key] = pd.to_datetime(value).year
    except Exception:
        result[key] = 'nan'

file_writer = "./essentia_features_year.csv"
with open(file_writer, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for t_d in track_ids[1:]:
       writer.writerow([t_d, result[t_d]])


#Genre Extraction
file_genre = "./fma_metadata/genres.csv"
with open(file_genre, mode='r') as infile:
    reader = csv.reader(infile)
    genre_map = {rows[0]:rows[3] for rows in reader}

file_track = "./fma_metadata/tracks.csv"
with open(file_track, mode='r') as infile:
    reader = csv.reader(infile)
    track_map = {rows[0]:rows[42] for rows in reader}

track_genre_map = {}
for k,v in track_map.items():
    genre_ids = re.findall('\d+', v )
    genres = []
    for genre_id in genre_ids:
        genres.append(genre_map[genre_id])
        track_genre_map[k] = genres

result = {}
for i in track_ids:
    result[i] = track_genre[i]

file_writer = "./essentia_features_genre.csv"
with open(file_writer, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in result.items():
       writer.writerow([key, value])
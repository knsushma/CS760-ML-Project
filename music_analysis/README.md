FMA - Project Idea

Given a dataset containing features extracted from audio samples, the following problems could be explored(can choose one or combination of the following) : 
* Song recommendation
* Genre classification
* Artist classification
* Year prediction

Metadata information associated with the audio samples:
* track metadata : ID, title, artist, genres, tags and play counts of audio samples
* Can retrieve tracks standalone or from their albums/artists using the metadata csv
* Genres metadata: hierarchy of genres with corresponding parent id to traverse the tree structure of genres

Features associated with the audio samples:
* Total of 518 features associated with the audio signal summarized with seven statistics (mean, standard deviation, skew, kurtosis, median, minimum, maximum) for features like:
Mel Frequency Cepstral Coefficient (MFCC), RMSE, Zero crossing rate
* Some audio samples have extra features extracted from Spotify like:
* Extra metadata of the songs like album and artist location
* Song-related info like danceability, acousticness, energy, instrumentalness, liveness, speechiness, tempo, valence
* Scores and ranks of various tracks for features like the discovery, familiarity and hottness of the artist and the song.

The github page also has complete info on how to extract songs and features(using APIs) and dump it to csv if we want to. There is a show of some baseline models implemented to perform genre recognition. 

Github page for code and dataset : https://github.com/mdeff/fma


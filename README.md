## Spotify Playlist Recommender

[![spr_web_app](https://github.com/nsanka/RecSys/blob/ec8f014f98016710bc50062b7638d2fe4498f4a2/images/spr_web_app.png)](https://share.streamlit.io/nsanka/recsys/main/streamlit/app.py)

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/nsanka/RecSys)

authors: Naga Sanka, David Hernandez and Sheila Pietono
date: Dec 18, 2021

<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Description](#description)
* [Data](#data)
* [Code](#code)
* [Summary](#summary)
* [Blogs](#blogs)
* [Contributions](#contributions)
* [Contributing](#contributing)
* [License](#license)

## **Description:**

The purpose of this project was to build a recommendation system to allow users to discover music based on their listening preferences. 
Users can explore connections to music by providing playlist they already enjoy or favorites they already love.
<br>

## **Data:**
* We used the dataset provided by Spotify to enable research in music recommendations and can be accessed [here](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files)
* This dataset includes public playlists created by US Spotify users between January 2010 and November 2017.
* It has 1 million Spotify playlists, over 2 million unique tracks, nearly 300,000 artists and 734,000 albums.
* We obtained audio features for all tracks through Spotify's API. Features include danceability, tempo, liveness, speechiness, etc.


## **Features and Target Variables:**<br>
<br>
We used primarily an unsupervised learning clustering approach for this project, we predict the cluster for a given user, then use only that cluster to find the similar playlists and thereby use the playlists for song recommendations:<br>

### **Data cleaning and pre-processing:**
* read playlists/track info from json files
* EDA
* extract audio features for each track
* calculate average features for the playlist
* normalize the features
* project data into 2D space using TSNE
* clustering using density and centroid models
* identify optimum k using Silhouette/Davies-Bouldin/Calinski-Harabasz index

<!-- CODE -->
## **Code:**<br>
### **code/Get_MPD_Data.ipynb**<br>
* This notebook is used to create the main .json file containing the playlists to train the model and to generate the recommendations. <br>
* The loop_slices() function will go through as many slices as desired to extract the necessary information from the playlists, <br>
* It is recommended to use 20 slices to run locally and scale it as needed with a bigger instances such as AWS.<br>

### **code/Playlist_Recommendation.ipynb**<br>
* This notebook will go through the entire analysis and development for the model and the recommendations. It describes what methods are used and how those were selected.<br>
* Seven different models were selected from different families and a 2D projection with TSNE was done. At 20,000 playlists, KMeans with k=17 is the best performer.<br>
* This notebook will generate the model and the playlist dataset to be used. All the models and datasets are saved locally.<br>

### **code/read_spotify_million_playlists.py**<br>
* This is the primary code that we used to read all the million playlists information<br>
* This code exports sqlite database tables that are eventually used in the streamlit app<br>

### **streamlit/app.py**<br>
* This is the code used to build the [streamlit web application](https://share.streamlit.io/nsanka/recsys/main/streamlit/app.py)<br>
* This calls the class defined in spotify_client.py to get recommendations<br>

### **streamlit/spotipy_client.py**<br>
* This code was primarily used to generate the song recommendations based on user input<br>
* This code also has a class to connect to Spotify API using user access token<br>
* It takes machine learning models generated	above and user input from web app to recommend top n songs<br>
* It also has functions to create visualizations<br>

### **streamlit/style.css**<br>
* This is used to define web app CSS styles<br>

<!-- SUMMARY -->
## **Summary:**
The final product is a streamlit app which allows users to do the following:
* explore the similar or dissimilar songs related to the users music
* explore the top playlists that are similar to their preference
* see the genres they listen to the most
* obtain recommended songs to listen to based on users favorites collected in a time frame of last month, 6 months or all time
* obtain recommended songs to listen to based on any playlist

<!-- BLOGS -->
## **Blogs:**
[Part 1: Create Development Environment](https://nsanka.medium.com/music-recommender-system-part-1-86936d673c31?sk=4278ddfebc850599db2fca4a5f2a2104)<br>
[Part 2: Get the music dataset and perform Exploratory Data Analysis](https://nsanka.medium.com/music-recommender-system-part-2-ff4c3f54cba3?sk=2ad792ce8d7cf1433a8a50cebf2915e3)<br>
[Part 3: Build and train machine learning models](https://medium.com/@david.de.hernandez/modeling-data-for-a-spotify-recommender-system-3056997a0fc5?sk=7ce613a3d5cbd4a69e73804983d49f91)<br>
[Part 4: Evaluate the effect of dataset size on machine learning models](https://medium.com/@david.de.hernandez/scaling-data-on-an-aws-instance-for-an-spotify-recommender-system-fa71476e0d7c?sk=470e010f9d03865462400e65e07ab8ad)<br>
[PART 5: Pushing the Project to Cloud Computing (AWS Instance)](https://medium.com/@sheila_vp/part-4-pushing-the-project-to-cloud-computing-aws-instance-ce3b28ce8b6e)<br>
[Final Part: Deploy ML Based Recommender System into Production](https://medium.com/@nsanka/music-recommender-system-part-5-4278bf89411c)<br>

<!-- CONTRIBUTIONS -->
## **Contributions:**<br>
<br>
This project marks the completion of a Master's degree in Applied Data Science at Univeristy of Michigan.<br>
Naga Sanka - Data extraction and manipulation, Update code to recommendation predictions. Deploy the model to a web app.<br>
David Hernandez - Machine Learning Modeling and Recommender System for different dataset sizes.<br>
Sheila Pietono - Exploratory data analysis and Scale the data to an AWS instance.<br>

<!-- FUTURE IMPROVEMENTS -->
## **Future Improvements:**<br>
<br>
This data set has much potential to keep working on it and I would like to list all the potential ramifications and future work that can be done with it.<br>
We performed quite a few QC checks of the data to make sure recommendations made sense by comparing average audio features between the playlists.<br>

There is still future work to be done on this project as this app is currently still in beta mode. These are the proposed future improvements:<br>
* properly link to Spotify API so that users are not requied to copy/paste the access token
* implement collaborative based recommendation in addition to content based recommendations
* use deep learning for recommender system
* use the user feedback to improve the recommendations

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the GNU GENERAL PUBLIC LICENSE. See `LICENSE` for more information.

# Streamlit
from altair.vegalite.v4.api import value
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import time
import base64
import pandas as pd
# Plot
import altair as alt
import os
import sys

# Get the current working directory
cwd = os.getcwd()
sys.path.insert(1, cwd)
import config

# Spotify Credentials
os.environ["SPOTIPY_CLIENT_ID"] = config.SPOTIPY_CLIENT_ID
os.environ["SPOTIPY_CLIENT_SECRET"] = config.SPOTIPY_CLIENT_SECRET
os.environ['SPOTIPY_REDIRECT_URI'] = config.SPOTIPY_REDIRECT_URI  # Needed for user authorization
css_file = os.path.join(cwd, 'streamlit', 'style.css')
log_file = os.path.join(cwd, 'data', 'read_spotify_mpd_log.txt')
feedback_db_file = os.path.join(cwd, 'data', 'user_feedback.db')

# Pickled models
model_path = os.path.join(cwd, 'models', 'KMeans_K17_20000_sample_model.sav')
tsne_path = os.path.join(cwd, 'models', 'openTSNETransformer.sav')
scaler_path = os.path.join(cwd, 'models', 'StdScaler.sav')
playlists_db_path = os.path.join(cwd, 'data', 'spotify_20K_playlists.db')
train_data_scaled_path = os.path.join(cwd, 'data' , 'scaled_data.csv')

# Spotipy
from spotipy_client import *

# Thanks to streamlitopedia for the following code snippet
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

SPR_SPOTIFY_URL = 'https://cdn-icons-png.flaticon.com/512/2111/2111624.png'

# Initial page config
st.set_page_config(
    page_title='Spotify Playlist Recommender',
    page_icon=SPR_SPOTIFY_URL,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define CSS for Style
local_css(css_file)

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'about_us'
if 'output' not in st.session_state:
    st.session_state.output = 'Start Logging'

if 'example_url' not in st.session_state:
    st.session_state.example_url = 'Example: https://open.spotify.com/embed/playlist/37i9dQZF1DX0kbJZpiYdZl'
def update_playlist_url():
    st.session_state.example_url = st.session_state.playlist_url

if 'authorize' not in st.session_state:
    st.session_state.authorize = False
if 'username' not in st.session_state:
    st.session_state.username = 'Enter Spotify Username'
if 'fav_songs' not in st.session_state:
    st.session_state.fav_songs = None
def set_authorize():
    st.session_state.authorize = True
def save_spotify_user():
    st.session_state.username = st.session_state.user

if 'user_op' not in st.session_state:
    st.session_state.user_op = 'Playlist'
def update_user_option():
    st.session_state.user_op = st.session_state.user_selection

if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
if 'got_rec' not in st.session_state:
    st.session_state.got_rec = False
if 'rec_type' not in st.session_state:
    st.session_state.rec_type = 'playlist'
if 'rec_uris' not in st.session_state:
    st.session_state.rec_uris = []
if 'wordcloud_fig' not in st.session_state:
    st.session_state.wordcloud_fig = None
def get_recommendations(rec_type):
    st.session_state.got_rec = False
    st.session_state.got_feedback = False
    st.session_state.app_mode = 'recommend'
    st.session_state.rec_type = rec_type

if 'feedback_db' not in st.session_state:
    st.session_state.feedback_db = User_FeedbackDB(feedback_db_file)
if 'got_feedback' not in st.session_state:
    st.session_state.got_feedback = False
def add_feedback(feedback):
    rec_type = st.session_state.rec_type
    rec_name = st.session_state.rec_type
    username = ''
    ml_model_options = ''
    if st.session_state.rec_type != 'playlist':
        rec_type = 'favorite'
        username = st.session_state.username
    fb_list = [feedback, rec_type, rec_name, ml_model_options, username]
    st.session_state.feedback_db.add_user_feedback(fb_list)
    st.session_state.got_feedback = True

def log_output(new_text):
    new_log = st.session_state.output
    if new_text != 'None':
        new_log = new_text + '\n' + st.session_state.output
    if st.session_state.display_output:
        st.session_state.output = st.session_state.log_holder.text_area('',value=new_log, height=500)

# Sidebar
def spr_sidebar():
    with st.sidebar:
        st.image(SPR_SPOTIFY_URL, width=60)
        st.info('**Spotify Playlist Recommender**')
        data_button = st.button("About Dataset")
        model_button = st.button('User Input')
        rec_button = st.button('Recommendations')
        blog_button = st.button('Blog Posts')
        about_button = st.button("About Our Team")
        st.checkbox('Display Output', True, key='display_output')
        st.session_state.log_holder = st.empty()
        #log_output('None')
        if data_button:
            st.session_state.app_mode = 'dataset'
        if model_button:
            st.session_state.app_mode = 'model'
        if rec_button:
            st.session_state.app_mode = 'recommend'
        if blog_button:
            st.session_state.app_mode = 'blog'
        if about_button:
            st.session_state.app_mode = 'about_us'

def dataset_page():
    st.markdown("<br>", unsafe_allow_html=True)
    """
    # Spotify Million Playlist Dataset
    For this project we are using The Million Playist Dataset, as it name implies, the dataset consists of one million playlists and each playlists 
    contains n number of songs and some metadata is included as well such as name of the playlist, duration, number of songs, number of artists, etc.
    
    It is created by sampling playlists from the billions of playlists that Spotify users have created over the years. 
    Playlists that meet the following criteria were selected at random:
    - Created by a user that resides in the United States and is at least 13 years old
    - Was a public playlist at the time the MPD was generated
    - Contains at least 5 tracks
    - Contains no more than 250 tracks
    - Contains at least 3 unique artists
    - Contains at least 2 unique albums
    - Has no local tracks (local tracks are non-Spotify tracks that a user has on their local device
    - Has at least one follower (not including the creator
    - Was created after January 1, 2010 and before December 1, 2017
    - Does not have an offensive title
    - Does not have an adult-oriented title if the playlist was created by a user under 18 years of age
    
    As you can imagine a million anything is too large to handle and we are going to be using 2% of the data (20,000 playlists) to create the models and the scaling to an AWS instance.
    
    ### Enhancing the data:
    Since this dataset is released by Spotify, it already includes a track id that can be used to generate API calls and 
    access the multiple information that is provided from Spotify for a given song, artist or user.
    These are some of the features that are available to us for each song and we are going to use them to enhance our dataset and to help matching the user's favorite playlist.
    
    ##### Some of the available features are the following, they are measured mostly in a scale of 0-1:
    - **acousticness:** Confidence measure from 0.0 to 1.0 on if a track is acoustic.   
    - **danceability:** Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, 
    rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.   
    - **energy:** Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, 
    energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. 
    Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.   
    - **instrumentalness:** Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or 
    spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. 
    Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.   
    - **liveness:** Detects the presence of an audience in the recording. Higher liveness values represent an increased probability 
    that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.   
    - **loudness:** The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful 
    for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical 
    strength (amplitude). Values typical range between -60 and 0 db.   
    - **speechiness:** Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording 
    (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably 
    made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in 
    sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.   
    - **tempo:** The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the 
    speed or pace of a given piece and derives directly from the average beat duration.   
    - **valence:** A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound 
    more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).   
    
    Information about features: [link](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)
    """
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader('Total VS New Tracks in each json file')
    st.plotly_chart(get_num_tracks_fig(log_file, 'total'), use_container_width=True)
    st.subheader('Existing VS New Tracks in each json file')
    st.plotly_chart(get_num_tracks_fig(log_file, 'existing'), use_container_width=True)

def playlist_page():
    st.subheader("User Playlist")
    uri_link = st.session_state.playlist_url
    if 'Example:' in uri_link:
        uri_link = uri_link[9:]
    components.iframe(uri_link, height=300)
    return
    spotify = SpotifyAPI(client_id, client_secret)
    Data = spotify.search({"name": f"{uri_link}"}, search_type="playlist")

    need = []
    for i, item in enumerate(Data['tracks']['items']):
        track = item['album']
        track_id = item['id']
        song_name = item['name']
        popularity = item['popularity']
        need.append((i, track['artists'][0]['name'], track['name'], track_id, song_name, track['release_date'], popularity))
    
    Track_df = pd.DataFrame(need, index=None, columns=('Item', 'Artist', 'Album Name', 'Id', 'Song Name', 'Release Date', 'Popularity'))
    access_token = spotify.access_token

    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    endpoint = "https://api.spotify.com/v1/audio-features/"

    Feat_df = pd.DataFrame()
    for id in Track_df['Id'].iteritems():
        track_id = id[1]
        lookup_url = f"{endpoint}{track_id}"
        ra = requests.get(lookup_url, headers=headers)
        audio_feat = ra.json()
        Features_df = pd.DataFrame(audio_feat, index=[0])
        Feat_df = Feat_df.append(Features_df)

    Full_Data = Track_df.merge(Feat_df, left_on="Id", right_on="id")
    print(Full_Data.columns)

    Sort_DF = Full_Data.sort_values(by=['Popularity'], ascending=False)

    Name_of_Feat = 'acousticness'
    chart_df = Sort_DF[['Artist', 'Album Name', 'Song Name', 'Release Date', 'Popularity', f'{Name_of_Feat}']]

    # Streamlit Chart
    feat_header = Name_of_Feat.capitalize()

    st.header(f'{feat_header}' " vs. Popularity")
    c = alt.Chart(chart_df).mark_circle().encode(
        alt.X('Popularity', scale=alt.Scale(zero=False)), y=f'{Name_of_Feat}', color=alt.Color('Popularity', scale=alt.Scale(zero=False)), 
        size=alt.value(200), tooltip=['Popularity', f'{Name_of_Feat}', 'Song Name', 'Album Name'])

    st.altair_chart(c, use_container_width=True)      

    st.header("Table of Attributes")
    st.table(chart_df)

def insert_songs(placeholder, track_uris):
    with placeholder.container():
        for uri in track_uris:
            uri_link = "https://open.spotify.com/embed/track/" + uri + "?utm_source=generator&theme=0"
            components.iframe(uri_link, height=80)

def favs_page():
    Name = st.session_state.spr.sp.me()['display_name']
    st.subheader(Name + '\'s Favorite Songs')
    if st.session_state.fav_songs is None:
        st.session_state.fav_songs = st.session_state.spr.get_tracks_from_playlist_or_user_favorites()['uri']

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Last Month")
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Get Recommendations", key='lm', on_click=get_recommendations, args=('last_month',))
        st.markdown("<br>", unsafe_allow_html=True)
        left_songsholder = st.empty()
        insert_songs(left_songsholder, st.session_state.fav_songs.last('1M'))

    with middle_column:
        st.subheader("6 Months")
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Get Recommendations", key='6m', on_click=get_recommendations, args=('6_months',))
        st.markdown("<br>", unsafe_allow_html=True)
        middle_songsholder = st.empty()
        insert_songs(middle_songsholder, st.session_state.fav_songs.last('6M'))

    with right_column:
        st.subheader("All Time")
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Get Recommendations", key='at', on_click=get_recommendations, args=('all_time',))
        st.markdown("<br>", unsafe_allow_html=True)
        right_songsholder = st.empty()
        insert_songs(right_songsholder, st.session_state.fav_songs)

def model_page():
    st.header("Select your preference")
    Types_of_Features = ("Playlist", 'Favorites')
    st.session_state.user_selection = st.session_state.user_op
    st.selectbox("Feature", Types_of_Features, key='user_selection', on_change=update_user_option)

    if st.session_state.user_selection == "Playlist":
        st.session_state.playlist_url = st.session_state.example_url
        st.text_input("Playlist URI", key='playlist_url', on_change=update_playlist_url)
        playlist_uri = st.session_state.playlist_url.split('/')[-1]
        st.session_state.spr = SpotifyRecommendations(playlist_uri=playlist_uri)
        st.session_state.spr.log_output = log_output
        playlist_page()
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Get Recommendations", key='pl', on_click=get_recommendations, args=('playlist',))
        with st.expander("Here's how to find any Playlist URL in Spotify"):
            st.write(""" 
                - Search for Playlist on the Spotify app
                - Right Click on the Playlist you like
                - Click "Share"
                - Choose "Copy link to playlist"
            """)
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(os.path.join(cwd, 'images', 'spotify_get_playlist_uri.png'))
    else:
        st.session_state.user = st.session_state.username
        st.text_input('Spotify Username', key='user', on_change=save_spotify_user)
        if st.session_state.authorize:
            st.session_state.spr = SpotifyRecommendations(sp_user = st.session_state.user)
            st.session_state.spr.log_output = log_output
            favs_page()
        else:
            st.button("Login with Spotify", on_click=set_authorize)

def load_spr_ml_model():
    st.session_state.ml_model = SPR_ML_Model(model_path, tsne_path, scaler_path, playlists_db_path, train_data_scaled_path)
    
def rec_page():
    if st.session_state.rec_type == 'playlist':
        st.subheader('Recommendations based on Playlist:')
    elif st.session_state.rec_type == 'last_month':
        st.subheader('Recommendations based on your Last Month Favorites:')
    elif st.session_state.rec_type == '6_months':
        st.subheader('Recommendations based on your Six Months Favorites:')
    else:
        st.subheader('Recommendations based on your All Time Favorites:')
    status_holder = st.empty()
    rec_songsholder = st.empty()
    user_fbholder = st.empty()

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        fb_plotholder = st.empty()
    with middle_column:
        rec_page_status = st.empty()
    with right_column:
        wordcloud_holder = st.empty()

    if st.session_state.ml_model is None:
        with status_holder:
            with st.spinner('Loading ML Model...'):
                load_spr_ml_model()
            st.success('ML Model Loaded!')
    else:
        log_output('ML Model already loaded')
    
    if st.session_state.got_rec == False:
        spr = st.session_state.spr
        spr.set_ml_model(st.session_state.ml_model)
        with status_holder:
            with st.spinner('Getting Recommendations...'):
                spr.len_of_favs = st.session_state.rec_type
                spr.log_output = log_output
                st.session_state.rec_uris = spr.get_songs_recommendations(n=10)
                st.session_state.wordcloud_fig = spr.get_spotify_wrapped()
                st.session_state.got_rec = True
            st.success('Here are top 10 recommendations!')
    else:
        log_output('Showing already found recommendations')
        time.sleep(1)
        log_output('For new recommendations, Click Get Recommentations in User Input')
        time.sleep(1)

    insert_songs(rec_songsholder, st.session_state.rec_uris)

    if st.session_state.got_feedback == False:
        with user_fbholder:
            c1, c2, c3, c4 = st.columns((1, 1, 1, 1))
            with c1:
                st.button("Love it", key='love', on_click=add_feedback, args=('Love it',))
            with c2:
                st.button("Like it", key='like', on_click=add_feedback, args=('Like it',))
            with c3:
                st.button("Okay", key='okay', on_click=add_feedback, args=('Okay',))
            with c4:
                st.button("Hate it", key='hate', on_click=add_feedback, args=('Hate it',))

    
    with fb_plotholder:
        try:
            fig = st.session_state.feedback_db.get_feedback_plot()
            if fig:
                st.subheader('User Feedback:')
                st.plotly_chart(fig, use_container_width=True)
        except:
            pass

    
    wordcloud_holder.pyplot(st.session_state.wordcloud_fig)

def blog_page():
    st.markdown("<br>", unsafe_allow_html=True)
    """
    ## Creating Recommender System using Machine Learning
    ### Part 1: Create Development Environment
    #### Introduction:

    Now a days we all see many automated recommender systems everywhere, a few well known ones are Netflix, Amazon, Youtube, LinkedIn, etc. 
    In this series, let's see how to build a recommender system using machine learning from scratch. As part of this series, 
    I would like to show how we can create a framework for applying different machine learning algorithms on a real world music dataset to 
    predict the playlist/songs recommendations. We will use four main approaches such as content based filtering, collaborative filtering, 
    model based methods and deep neural networks...
    
    [Read more on Medium...](https://nsanka.medium.com/music-recommender-system-part-1-86936d673c31?sk=4278ddfebc850599db2fca4a5f2a2104)
    """
    st.markdown("<br>", unsafe_allow_html=True)
    """
    ### Part 2: Get the music dataset and perform Exploratory Data Analysis
    #### Recap:

    In the previous article, we created Development Environment with all the necessary Python libraries.

    In this article, let's get the dataset that we use which is the dataset provided as part of the Spotify Million Playlist Dataset (MPD) Challenge. 
    In order to prepare the dataset to use in machine learning models, we need to perform some data cleaning and data manipulation tasks. 
    We will also explore the dataset to know the features and combine with additional data fields obtained via the Spotify API...
    
    [Read more on Medium...](https://nsanka.medium.com/music-recommender-system-part-2-ff4c3f54cba3?sk=2ad792ce8d7cf1433a8a50cebf2915e3)
    """
    st.markdown("<br>", unsafe_allow_html=True)
    """
    ### Part 3: Get the music dataset and perform Exploratory Data Analysis
    #### The Data

    For this project we are using The Million Playlist Dataset (MPD) released by Spotify. As it name implies, the dataset consists of one million 
    playlists and each playlists contains n number of songs and additional metadata is included as well such as title of the playlist, duration, 
    number of songs, number of artists, etc...

    [Read more on Medium...](https://medium.com/@david.de.hernandez/3056997a0fc5)
    """
    st.markdown("<br>", unsafe_allow_html=True)
    """
    ### Final Part: Deploy ML Based Recommender System into Production
    #### Recap:

    In the previous article, we created few machine learning models to find out the best recommendations based on user Spotify favorite music or playlist.

    This article documents how we can put it all together for public use. We don't want our ML models to die in a Jupyter Notebook. We want our models to be 
    integrated into a product/application and available to any user. The complete deployment process involves three major steps....

    [Read more on Medium...](https://medium.com/@nsanka/music-recommender-system-part-5-4278bf89411c)
    """
    st.markdown("<br>", unsafe_allow_html=True)

    # Example code to use JS
    #html_string = '''
    #              <h1>Music Recommender System — Part 1</h1>
    #              <script language="javascript">
    #               document.querySelector("h1").style.color = "red";
    #               console.log("Streamlit runs JavaScript");
    #               alert("Streamlit runs JavaScript");
    #              </script>
    #              '''
    #components.html(html_string)

def about_page():
    # Display header.
    st.markdown("<br>", unsafe_allow_html=True)
    """
    # About Our Team

    [![Star](https://img.shields.io/github/stars/nsanka/RecSys.svg?logo=github&style=social)](https://github.com/nsanka/RecSys/stargazers)
    &nbsp[![Follow](https://img.shields.io/twitter/follow/nsanka11?style=social)](https://www.twitter.com/nsanka11)
    """
    st.markdown("<br>", unsafe_allow_html=True)

def main():
    spr_sidebar()
    if st.session_state.app_mode == 'dataset':
        dataset_page()

    if st.session_state.app_mode == 'model':
        model_page()
    
    if st.session_state.app_mode == 'recommend':
        rec_page()

    if st.session_state.app_mode == 'blog':
        blog_page()
        
    if st.session_state.app_mode == 'about_us':
        about_page()

# Run main()
if __name__ == '__main__':
    main()

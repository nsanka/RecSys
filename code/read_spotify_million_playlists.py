import os
import re
import sys
import json
import pprint
import pandas as pd
import sqlite3
from sqlite3 import Error
import multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
from zipfile import ZipFile
import fnmatch
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

pd.set_option('display.max_rows', None)
zip_file = 'data/spotify_million_playlist_dataset.zip'
db_file = 'data/spotify_million_playlists.db'
log_file = 'data/read_spotify_mpd_log.txt'

sys.path.insert(1, os.getcwd())
import config
# Spotify credentials
os.environ["SPOTIPY_CLIENT_ID"] = config.SPOTIPY_CLIENT_ID
os.environ["SPOTIPY_CLIENT_SECRET"] = config.SPOTIPY_CLIENT_SECRET
os.environ['SPOTIPY_REDIRECT_URI'] = config.SPOTIPY_REDIRECT_URI

def write_log(text):
    with open(log_file, 'a') as lf:
        lf.write(str(text) + '\n')

def create_connection(db_file):
    """ create a database connection to the SQLite database specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        write_log('Connection to ' + db_file)
    except Error as e:
        write_log(e)
        print(e)
    return conn

def create_table(conn, create_table_sql, table_name):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        cur = conn.cursor()
        cur.execute(create_table_sql)
        write_log('Created table: ' + table_name)
    except Error as e:
        write_log(e)
        print(e)

def create_all_tables():
    sql_create_tracks_table = """ CREATE TABLE IF NOT EXISTS tracks (
                                    artist_name text,
                                    track_uri text NOT NULL,
                                    artist_uri text,
                                    track_name text NOT NULL,
                                    album_uri text,
                                    album_name text,
                                    track_id integer NOT NULL
                                    ); """

    sql_create_playlists_table = """CREATE TABLE IF NOT EXISTS playlists (
                                    name text NOT NULL,
                                    collaborative text,
                                    pid integer NOT NULL,
                                    modified_at integer,
                                    num_tracks integer,
                                    num_albums integer,
                                    num_followers integer,
                                    num_edits integer,
                                    duration_ms integer,
                                    num_artists integer
                                );"""

    sql_create_ratings_table = """CREATE TABLE IF NOT EXISTS ratings (
                                    pid integer NOT NULL,
                                    track_id integer NOT NULL,
                                    pos integer,
                                    num_followers integer,
                                    FOREIGN KEY (pid) REFERENCES playlists (pid),
                                    FOREIGN KEY (track_id) REFERENCES tracks (track_id)
                                );"""

    sql_create_features_table = """ CREATE TABLE IF NOT EXISTS features (
                                    track_id integer,
                                    danceability real,
                                    energy real,
                                    key real,
                                    loudness real,
                                    mode real,
                                    speechiness real,
                                    acousticness real,
                                    instrumentalness real,
                                    liveness real,
                                    valence real,
                                    tempo real,
                                    duration_ms integer,
                                    time_signature integer
                                    ); """

    # create a database connection
    conn = create_connection(db_file)

    # create tables
    if conn is not None:
        # create tracks table
        create_table(conn, sql_create_tracks_table, 'tracks')

        # create playlists table
        create_table(conn, sql_create_playlists_table, 'playlists')

        # create ratings table
        create_table(conn, sql_create_ratings_table, 'ratings')

        # create features table
        create_table(conn, sql_create_features_table, 'features')

    else:
        print("Error! cannot create the database connection.")

def select_track_by_trackuri(conn, track_uri):
    """
    Query tracks by track_uri
    :param conn: the Connection object
    :param track_uri:
    :return: track_id
    """
    cur = conn.cursor()
    cur.execute("SELECT track_id FROM tracks WHERE track_uri=?", (track_uri,))
    rows = cur.fetchall()
    track_id = 0
    if len(rows) > 0:
        track_id = rows[0][0]        
    return track_id

def get_max_track_id(conn, table_name):
    cur = conn.cursor()
    # Get Max track_id in database
    cur.execute("select max(track_id) from " + table_name)
    rows = cur.fetchall()
    max_track_id = rows[0][0]
    if max_track_id is None:
        max_track_id = 0
    return max_track_id

def create_playlist(conn, playlist, pid):
    """
    Create a new playlist
    :param conn:
    :param playlist:
    :return:
    """
    sql = ''' INSERT INTO playlists(name,collaborative,pid,modified_at,num_tracks,num_albums,num_followers,num_edits,duration_ms,num_artists)
              VALUES(?,?,?,?,?,?,?,?,?) '''
    try:
        cur = conn.cursor()
        cur.execute(sql, playlist)
        conn.commit()
        write_log('Added playlist: ' + pid)
    except:
        write_log('Failed to add playlist: ' + pid)

def get_all_playlist_ids(conn):
    cur = conn.cursor()
    # Get all pids of playlists in database
    cur.execute("select pid from playlists")
    rows = cur.fetchall()
    # rows will have [(0,), (1,)...]
    pids = []
    if len(rows) > 0:
        pids = [row[0] for row in rows]
    return pids

def get_playlist(conn, pid):
    """
    Query playlists by pid
    :param conn: the Connection object
    :param pid:
    :return: playlist
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM playlists WHERE pid=?", (pid,))
    rows = cur.fetchall()
    playlist = None
    if len(rows) > 0:
        playlist = rows[0]        
    return playlist

def get_table_df(conn, table_name):
    write_log('Reading table from database: ' + table_name)
    table_df = pd.read_sql('select * from ' + table_name, conn)
    print(table_df.head())
    return table_df

def get_average_audio_features(conn, pid):
    track_ids_df = pd.read_sql('select track_id from ratings where pid=' + str(pid), conn)
    features_df = get_table_df(conn, 'features')
    features_df = features_df.merge(track_ids_df, on='track_id')
    print('Playlist ', pid, 'has', len(features_df), 'tracks')
    average_df = features_df.drop(columns='track_id').mean()
    print(average_df)
    return average_df

def create_audio_features(cnt_uris=100):
    conn = create_connection(db_file)
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    max_track_id = get_max_track_id(conn, 'tracks')
    min_track_id = get_max_track_id(conn, 'features')

    print('features min track_id:', min_track_id, 'tracks max track_id', max_track_id)
    for idx in range(min_track_id, max_track_id, cnt_uris):
        #print('getting audio features for track_id ', idx+1, ' to ', idx+cnt_uris)
        write_log('Getting audio features for track_ids: ' + str(idx+1) + '-' + str(idx+cnt_uris))
        cur = conn.cursor()
        cur.execute('''select track_id, track_uri from tracks where (track_id > ?) and (track_id <= ?)''', (idx, idx+cnt_uris))
        rows = cur.fetchall()
        uris = [row[1] for row in rows]
        for _ in range(10):
            try:
                feats_list = sp.audio_features(uris)
            except Exception as e: 
                print(e)
            else:
                break
        else:
            print('All 10 attempts failed, try after sometime')
            write_log('All 10 attempts failed, try after sometime')
            break

        track_id_list = range(idx+1, idx+cnt_uris+1)
        # Remove rows where the features are None
        track_id_list = [track_id_list[feats_list.index(item)] for item in feats_list if item]
        print('got features for track_ids: ', track_id_list[0], '-', track_id_list[-1])
        feats_list = [item for item in feats_list if item]
        feats_df = pd.DataFrame(feats_list)
        columns = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
        feats_df = feats_df[columns]
        feats_df.insert(loc=0, column='track_id', value=track_id_list)
        write_log('Adding audio features for track_ids: ' + str(track_id_list[0]) + '-' + str(track_id_list[-1]))
        #print(feats_df.head())
        feats_df.to_sql(name='features', con=conn, if_exists='append', index=False)
    if conn:
        conn.close()

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def to_date(epoch):
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d")

def print_most_common(txt, df, col, num, is_date=False):
    print()
    print(txt)
    print("%s %s" % ('  count', col))
    for name, count in df[col].value_counts()[:num].items():
        if is_date:
            print("%7d %s" % (count, to_date(name)))
        else:
            print("%7d %s" % (count, name))

def show_summary():
    write_log('Printing Summary Statistics')
    playlists_df, tracks_df, features_df = read_all_tables()
    total_playlists = len(playlists_df)
    #total_tracks = len(tracks_df)
    total_tracks = playlists_df['num_tracks'].sum()
    #total_descriptions = len(playlists_df[playlists_df['description'].notna()])

    titles = set(playlists_df["name"].tolist())
    playlists_df["nname"] = playlists_df["name"].map(normalize_name)
    ntitles = set(playlists_df["nname"].tolist())

    albums = set(tracks_df["album_uri"].tolist())
    tracks = set(tracks_df["track_uri"].tolist())
    artists = set(tracks_df["artist_uri"].tolist())

    print()
    print("number of playlists", total_playlists)
    print("number of tracks", total_tracks)
    print("number of unique tracks", len(tracks))
    print("number of unique features", len(features_df))
    print("number of unique albums", len(albums))
    print("number of unique artists", len(artists))
    print("number of unique titles", len(titles))
    #print("number of playlists with descriptions", total_descriptions)
    print("number of unique normalized titles", len(ntitles))
    print("avg playlist length", float(total_tracks) / total_playlists)
    print_most_common("top playlist titles", playlists_df, "nname", 20)
    tracks_df['full_name'] = tracks_df["track_name"] + " by " + tracks_df["artist_name"]
    print_most_common("top tracks", tracks_df, 'full_name', 20)
    print_most_common("top artists", tracks_df, "artist_name", 20)
    print_most_common("numedits histogram", playlists_df, "num_edits", 20)
    print_most_common("last modified histogram", playlists_df, "modified_at", 20, True)
    print_most_common("playlist length histogram", playlists_df, "num_tracks", 20)
    print_most_common("num followers histogram", playlists_df, "num_followers", 20)

def process_json_data(json_data, num_playlists):
    conn = create_connection(db_file)

    # Get Max track_id in tracks table
    max_track_id = get_max_track_id(conn, 'tracks')
    existing_pids = get_all_playlist_ids(conn)
    
    # Get all playlists in the file
    playlists_df = pd.json_normalize(json_data['playlists'])
    playlists_df.drop(['tracks', 'description'], axis=1, inplace=True)
    #print(playlists_df.head())

    # Remove playlists if they are in database
    playlists_df = playlists_df[~playlists_df['pid'].isin(existing_pids)]
    # Get only num_playlists if requested
    if num_playlists > 0:
        playlists_df = playlists_df.iloc[:num_playlists]
    if len(playlists_df) == 0:
        print('All playlists from this file are in database')
        return
    #print(playlists_df.head(10))
    print('Adding playlists to database:', playlists_df['pid'].min(), playlists_df['pid'].max())
    write_log('Adding all playlists to database from file: ')
    write_log('Adding playlists: ' + str(playlists_df['pid'].min()) + '-' + str(playlists_df['pid'].max()))
    playlists_df.to_sql(name='playlists', con=conn, if_exists='append', index=False)

    # Get all the tracks in the file
    tracks_df = pd.json_normalize(json_data['playlists'], record_path=['tracks'], meta=['pid', 'num_followers'])
    #print(tracks_df.head())
    tracks_df = tracks_df[tracks_df['pid'].isin(playlists_df['pid'].values)]
    tracks_df['track_uri'] = tracks_df['track_uri'].apply(lambda uri: uri.split(':')[2])
    tracks_df['album_uri'] = tracks_df['album_uri'].apply(lambda uri: uri.split(':')[2])
    tracks_df['artist_uri'] = tracks_df['artist_uri'].apply(lambda uri: uri.split(':')[2])
    print('Total tracks/ratings in this file: ', len(tracks_df))
    write_log('Total tracks/ratings in this file: ' + str(len(tracks_df)))

    print('Get track_id for existing tracks from database, create one for new tracks')
    write_log('Get track_id for existing tracks from database, create one for new tracks')
    all_tracks_df = pd.read_sql('select track_id, track_uri from tracks', conn)
    tracks_df = tracks_df.merge(all_tracks_df, how='left', on='track_uri').fillna(0)
    #print('Total tracks after merge: ', len(tracks_df))
    #print(tracks_df.head(100))
    print('Tracks already exist', len(tracks_df[tracks_df['track_id'] != 0]['track_uri'].unique()))
    write_log('Tracks already exist: ' + str(len(tracks_df[tracks_df['track_id'] != 0]['track_uri'].unique())))
    tracks_df['track_id1'] = tracks_df[tracks_df["track_id"] == 0][['track_uri']].groupby('track_uri').ngroup()+max_track_id+1
    tracks_df['track_id'] = tracks_df['track_id'] + tracks_df['track_id1'].fillna(0)
    tracks_df['track_id'] = tracks_df['track_id'].astype('int64')
    #print('Total tracks with new track_id: ', len(tracks_df))
    print('Created new track_ids', len(tracks_df[tracks_df['track_id1'].notna()]['track_uri'].unique()))
    write_log('Created new track_ids: ' + str(len(tracks_df[tracks_df['track_id1'].notna()]['track_uri'].unique())))

    # Save ratings to the database
    ratings_df = tracks_df[['pid', 'track_id', 'pos', 'num_followers']]
    #print(ratings_df.head())
    print('Adding all ratings to database from file: ' + ' ' + str(len(ratings_df)))
    write_log('Adding all ratings to database from file: ' + ' ' + str(len(ratings_df)))
    ratings_df.to_sql(name='ratings', con=conn, if_exists='append', index=False)

    # Save unique tracks to the database
    tracks_df = tracks_df[tracks_df['track_id1'].notna()]
    tracks_df.drop(['pos', 'duration_ms', 'pid', 'num_followers', 'track_id1'], axis=1, inplace=True)
    tracks_df = tracks_df.drop_duplicates(subset='track_uri', keep="first")
    print('Total unique tracks: ', len(tracks_df))
    print('Adding tracks to database:', max_track_id+1, tracks_df['track_id'].max())
    write_log('Adding tracks to database: ' + str(max_track_id+1) + '-' + str(tracks_df['track_id'].max()))
    #print(tracks_df.tail())
    tracks_df.to_sql(name='tracks', con=conn, if_exists='append', index=False)

    if conn:
        conn.close()

def extract_mpd_dataset(zip_file, num_files=0, num_playlists=0):
    with ZipFile(zip_file) as zipfiles:
        file_list = zipfiles.namelist()

        #get only the csv files
        json_files = fnmatch.filter(file_list, "*.json")
        json_files = [f for i,f in sorted([(int(filename.split('.')[2].split('-')[0]), filename) for filename in json_files])]

        cnt = 0
        # Init multiprocessing.Pool()
        #print("Number of processors: ", mp.cpu_count())
        #pool = mp.Pool(mp.cpu_count())

        for filename in json_files:
            cnt += 1
            print('\nFile: ' + filename)
            write_log('\nFile: ' + filename)

            with zipfiles.open(filename) as json_file:
                json_data = json.loads(json_file.read())
                process_json_data(json_data, num_playlists)
                #pool.apply(process_json_data, args=(filename, num_playlists))

            if (cnt == num_files) and (num_files > 0):
                break
        # Close muliprocessing poolS
        #pool.close()

def read_all_tables():
    conn = create_connection(db_file)
    print()
    playlists_df = get_table_df(conn, 'playlists')
    print(list(playlists_df.columns))
    print()
    tracks_df = get_table_df(conn, 'tracks')
    print(list(tracks_df.columns))
    print()
    features_df = get_table_df(conn, 'features')
    print(list(features_df.columns))
    # Ratings table is too big to read full, it has 343,960,399 rows
    #ratings_df = get_table_df(conn, 'ratings')

    # Get average features for playlist: pid
    #average_features_df = get_average_audio_features(conn, 0)
    return playlists_df, tracks_df, features_df

if __name__ == '__main__':
    start_time = datetime.now()
    write_log("Start Time =" + start_time.strftime("%H:%M:%S"))

    # Create playlists, tracks, ratings, features tables in database
    create_all_tables()
    
    # add tracks and playlists for each json file in zipfile
    extract_mpd_dataset(zip_file, 0, 0)
    
    # get audio features for all tracks
    create_audio_features()

    # Print the summary statistics
    show_summary()
    
    end_time = datetime.now()
    write_log("End Time =" + end_time.strftime("%H:%M:%S"))
    write_log("Total Time: " + str(end_time - start_time) + '\n')

import os
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

zip_file = 'data/spotify_million_playlist_dataset.zip'
db_file = 'data/spotify_mpd.db'
log_file = 'data/read_log.txt'

sys.path.insert(1, '/Users/nsanka/Downloads/RecSys')
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

        return conn
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

def create_audio_features(conn, cnt_uris=100):
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    max_track_id = get_max_track_id(conn, 'tracks')
    min_track_id = get_max_track_id(conn, 'features')

    print('features min track_id:', min_track_id, 'tracks max track_id', max_track_id)
    for idx in range(min_track_id, max_track_id, cnt_uris):
        #print('getting audio features for track_id ', idx+1, ' to ', idx+cnt_uris)
        write_log('getting audio features for track_id ' + str(idx+1) + ' to ' + str(idx+cnt_uris))
        cur = conn.cursor()
        cur.execute('''select track_id, track_uri from tracks where (track_id > ?) and (track_id <= ?)''', (idx, idx+cnt_uris))
        rows = cur.fetchall()
        uris = [row[1] for row in rows]
        for attempt in range(10):
            print('Attempt: ', attempt)
            try:
                feats_list = sp.audio_features(uris)
            except Exception as e: 
                print(e)
            else:
                break
        else:
            print('All 10 attempts failed, try after sometime')
            break

        track_id_list = range(idx+1, idx+cnt_uris+1)
        # Remove rows where the features are None
        track_id_list = [track_id_list[feats_list.index(item)] for item in feats_list if item]
        write_log('got features for track_ids: ' + str(track_id_list[0]) + '-' + str(track_id_list[-1]))
        print('got features for track_ids: ', track_id_list[0], '-', track_id_list[-1])
        feats_list = [item for item in feats_list if item]
        feats_df = pd.DataFrame(feats_list)
        columns = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
        feats_df = feats_df[columns]
        feats_df.insert(loc=0, column='track_id', value=track_id_list)
        #print(feats_df.head())
        feats_df.to_sql(name='features', con=conn, if_exists='append', index=False)

def process_json_data_loop(filename, num_playlists):
    conn = create_all_tables()
    print('File: ' + filename)
    write_log('File: ' + filename)

    # Get Max track_id in tracks table
    max_track_id = get_max_track_id(conn, 'tracks')

    playlists = []
    ratings = []
    tracks = []
    cnt = 0
    with ZipFile(zip_file) as zipfiles:
        with zipfiles.open(filename) as json_file:
            json_data = json.loads(json_file.read())
            for playlist in json_data['playlists']:
                pl = get_playlist(conn, playlist['pid'])
                if pl != None:
                    write_log('Playlist already exists in database: ' + str(playlist['pid']))
                    print('Playlist already exists in database: ' + str(playlist['pid']))
                    # Playlist already exists in table
                    continue
                else:
                    write_log('Processing Playlist pid: ' + str(playlist['pid']))
                    print('Processing Playlist pid: ' + str(playlist['pid']))
                cnt += 1
                new_playlist = [playlist['pid'], playlist['name'], playlist['num_albums'], playlist['num_artists'], playlist['num_edits'], 
                                playlist['num_tracks'], playlist['collaborative'], playlist['duration_ms'], playlist['modified_at']]
                playlists.append(new_playlist)

                for track in playlist['tracks']:
                    track_uri = track['track_uri'].split(':')[2]
                    write_log('Processing Track: ' + track_uri)
                    # Check if track_uri exists in the tracks list
                    track_id = next((t[0] for t in tracks if track_uri in t), None)
                    if not track_id:
                        write_log('Track not found in tracks list: ' + track_uri)
                        # Check track_uri exists in the database
                        track_id = select_track_by_trackuri(conn, track_uri)
                        if track_id == 0:
                            write_log('Track not found in database: ' + track_uri)
                            album_uri = track['album_uri'].split(':')[2]
                            artist_uri = track['artist_uri'].split(':')[2]
                            # (max_track_id + 1) tracks already exist in database
                            track_id = len(tracks) + (max_track_id + 1)
                            new_track = [track_id, track['track_name'], track_uri, track['album_name'], album_uri, track['artist_name'], artist_uri]
                            write_log('Adding Track to database: ' + str(track_id))
                            tracks.append(new_track)
                    
                    new_rating = [playlist['pid'], track_id, track['pos'], playlist['num_followers']]
                    write_log('Adding Rating for pid:' + str(playlist['pid']) + ' track_id: ' + str(track_id))
                    ratings.append(new_rating)

                if (cnt == num_playlists) and (num_playlists > 0):
                    break

            playlist_cols = ['pid','name','num_albums','num_artists','num_edits','num_tracks','collaborative','duration_ms','modified_at']
            playlists_df = pd.DataFrame(playlists, columns=playlist_cols)
            #print(playlists_df.head())
            write_log('Adding all playlists to database from file: ' + filename)
            playlists_df.to_sql(name='playlists', con=conn, if_exists='append', index=False)

            rating_cols = ['pid', 'track_id', 'pos', 'num_followers']
            ratings_df = pd.DataFrame(ratings, columns=rating_cols)
            #print(ratings_df.head())
            write_log('Adding all ratings to database from file: ' + filename)
            ratings_df.to_sql(name='ratings', con=conn, if_exists='append', index=False)

            track_cols = ['track_id', 'track_name', 'track_uri', 'album_name', 'album_uri', 'artist_name', 'artist_uri']
            tracks_df = pd.DataFrame(tracks, columns=track_cols)
            #print(tracks_df.tail())
            write_log('Adding all tracks to database from file: ' + filename)
            tracks_df.to_sql(name='tracks', con=conn, if_exists='append', index=False)

            # get audio features for all tracks
            create_audio_features(conn)

    if conn:
        conn.close()

def process_json_data(filename, num_playlists):
    conn = create_all_tables()
    print('File: ' + filename)
    write_log('File: ' + filename)

    # Get Max track_id in tracks table
    max_track_id = get_max_track_id(conn, 'tracks')
    existing_pids = get_all_playlist_ids(conn)

    with ZipFile(zip_file) as zipfiles:
        with zipfiles.open(filename) as json_file:
            json_data = json.loads(json_file.read())

            playlists_df = pd.json_normalize(json_data['playlists'])
            # Remove playlists if they are in database
            playlists_df = playlists_df[~playlists_df['pid'].isin(existing_pids)]
            # Get only num_playlists if requested
            if num_playlists > 0:
                playlists_df = playlists_df.iloc[:num_playlists]
            if len(playlists_df) == 0:
                print('Added all playlists from this file')
                return
            playlists_df.drop(['tracks', 'description'], axis=1, inplace=True)
            #print(playlists_df.head())
            print('Processing playlists:', playlists_df['pid'].min(), playlists_df['pid'].max())
            write_log('Adding all playlists to database from file: ' + filename)
            playlists_df.to_sql(name='playlists', con=conn, if_exists='append', index=False)

            tracks_df = pd.json_normalize(json_data['playlists'], record_path=['tracks'], meta=['pid', 'num_followers'])
            if num_playlists > 0:
                tracks_df = tracks_df[tracks_df['pid'].isin(playlists_df['pid'].values)]
            tracks_df['track_uri'] = tracks_df['track_uri'].apply(lambda uri: uri.split(':')[2])
            tracks_df['album_uri'] = tracks_df['album_uri'].apply(lambda uri: uri.split(':')[2])
            tracks_df['artist_uri'] = tracks_df['artist_uri'].apply(lambda uri: uri.split(':')[2])
            print('Get track_id for existing tracks from database')
            #all_tracks_df = pd.read_sql('select track_id, track_uri from tracks', conn)
            #print(len(all_tracks_df))
            #print(all_tracks_df.tail())
            tracks_df['track_id'] = tracks_df['track_uri'].apply(lambda uri: select_track_by_trackuri(conn, uri))
            tracks_df['track_id1'] = tracks_df[tracks_df["track_id"]==0].groupby('track_id').cumcount()+max_track_id+1
            tracks_df['track_id'] = tracks_df['track_id'] + tracks_df['track_id1'].fillna(0)
            tracks_df['track_id'] = tracks_df['track_id'].astype('int64')

            ratings_df = tracks_df[['pid', 'track_id', 'pos', 'num_followers']]
            #print(ratings_df.head())
            write_log('Adding all ratings to database from file: ' + filename)
            ratings_df.to_sql(name='ratings', con=conn, if_exists='append', index=False)

            tracks_df = tracks_df[tracks_df['track_id1'].notna()]
            tracks_df.drop(['pos', 'duration_ms', 'pid', 'num_followers', 'track_id1'], axis=1, inplace=True)
            print('Processing tracks:', max_track_id, tracks_df['track_id'].max())
            #print(tracks_df.tail())
            write_log('Adding all tracks to database from file: ' + filename)
            tracks_df.to_sql(name='tracks', con=conn, if_exists='append', index=False)

            # get audio features for all tracks
            create_audio_features(conn)

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
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())

    for filename in json_files:
        cnt += 1
        process_json_data(filename, num_playlists)
        pool.apply(process_json_data, args=(filename, num_playlists))
        if (cnt == num_files) and (num_files > 0):
            break
    # Close muliprocessing poolS
    pool.close()

if __name__ == '__main__':
    start_time = datetime.now()
    write_log("Start Time =" + start_time.strftime("%H:%M:%S"))
    # add tracks and playlists for each json file in zipfile
    extract_mpd_dataset(zip_file, 0, 500)
    end_time = datetime.now()
    write_log("End Time =" + end_time.strftime("%H:%M:%S"))
    write_log("Time Take: "+ str(end_time - start_time))

# -*- coding: utf-8 -*-
import streamlit as st
from streamlit import caching
import numpy as np
import pandas as pd
import os
import spotipy
import matplotlib.pyplot as plt

from code_complementaire.MultiApp import MultiApp
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheHandler
from spotify_collector.secrets_spotify import client_id,client_secret,redirect_uri
from spotify_collector.spotify_connector import get_spotipy
from code_complementaire.extraction_spotipy import *
from code_complementaire.playlist_soufflee import *
from code_complementaire.analyses import *
from code_complementaire.recommendation_annees import *
from code_complementaire.nuage_artist import *

@st.cache(allow_output_mutation=True)
def get_spotipy_ready():
    scope = "playlist-read-private user-read-email user-read-private playlist-modify-private user-top-read"
    sp = get_spotipy(scope, client_id,client_secret,redirect_uri)
    res=sp.current_user
    return sp

def accueil():
    caching.clear_cache()
    st.title('SpotData')
    st.write('Par J.Delaplace, E.Lei, C.Nothhelfer, P.Vehrlé')
    st.write('SpotData est une WebApp vous proposant de faire analyser vos playlists Spotify, et de découvrir encore plus de musiques qui vous correspondent.')
    st.write("Vous allez être redirigé vers une page d'autentification. Une fois cette dernière finie, revenez sur cette page.")
    textPlaceholder = st.empty()
    click = textPlaceholder.button("Se connecter à Spotify")
    if click:
        textPlaceholder.text("Vous êtes connecté. Vous pouvez naviguer sur le reste de l'application.")
        sp = get_spotipy_ready()
        res = sp.current_user()
    return None

def apres_auth():
    #Récupération des informations de l'utilisateur
    sp = get_spotipy_ready()
    user_info = sp.current_user()

    st.title('SpotData')
    st.header("Bienvenue, {}".format(user_info['display_name']))
    # st.image('Desktop/spotify_profil_mockup.png')

    #Affichage des informations des l'utilisatuer (si elles existent)
    st.subheader('Mes informations')
    if user_info["images"]!=[]: #photo
        st.image(user_info["images"][0]["url"])
    if user_info["email"]!=[]: #email
        st.write("E-mail : {}".format(user_info["email"]))
    if user_info["country"]!=[]: #pays
        st.write("Pays : {}".format(country(user_info["country"])))

    #Récupération des playlists ey de leur ID
    _, playlists = get_playlists(sp.current_user_playlists()["items"])
    st.subheader('Pour commencer, une vision d\'ensemble de votre musique')
    
    #Affichage des statistiques golbales
    st.write('  - _Nombre de playlists_ : {}'.format(len(playlists)))
    all_track = all_tracks(playlists,sp)
    st.write('  - _Nombre de titres enregistrés_ : {}'.format(len(all_track)))
    st.write('  - _Nombre d\'artistes écoutés_ : {}'.format(len(all_artists(playlists,sp))))
    
    #Affichage des tops artistes
    top_artists = sp.current_user_top_artists(limit=5)
    if top_artists["total"]!=0: #On n'affiche pas les top artistes si l'utilisateur n'en a pas
        st.write('  - _Artistes les plus écoutés_ : {}'.format(top_artist_to_string(top_artists)))
    
    #Affichage du top track
    top_tracks = sp.current_user_top_tracks(limit=1)
    if top_tracks["total"]!=0:
        st.write('  - _Titre le plus écouté_ : ___{}___ ({})'.format(top_tracks["items"][0]["name"], top_tracks["items"][0]["artists"][0]['name']))
    return None


def analyse():
    sp = get_spotipy_ready()
    playlists = sp.current_user_playlists()
    name_playlists, id_playlists = get_playlists(playlists["items"])
    
    st.title('SpotData')
    st.header('Analyse de vos playlists')
    
    st.subheader("Quelle playlist souhaitez-vous faire analyser ?")
    crtPlaylist=st.selectbox('Vos playlists: ',name_playlists)
    crtId = name_to_id(name_playlists, id_playlists, crtPlaylist)
    
    img_url = sp.playlist(crtId)['images'][0]['url']
    
    dataPL = creat_df_audiofeatures(crtId, sp)
    totTime = dataPL['length'].sum()//1000  # en secondes
    st.write('__Nombre de pistes__ : {}'.format(dataPL.shape[0]))
    st.write('__Durée de la playlist__ : {} h {} min {} s'.format(totTime//3600, totTime//60-(totTime//3600)*60, totTime - totTime//3600*3600 - (totTime//60-(totTime//3600)*60)*60))
    
    tabAF = create_work()
    tabTags = gen_tags(tabAF, dataPL)
    display_plotly([gen_wind_rose(tabTags[tabTags['to analyse']], img_url, crtPlaylist)])
    # st.write('(Texte d\'analyse=>Playlist sport/tranquille etc.-pas prioritaire-)')
    
    # display_plotly([gen_corr_scatter(dataPL, tabTags[tabTags['to analyse']].index)])
    
    a=st.multiselect('Audio-Features',['Années de sortie','Acousticness','Danceability','Energy','Instrumentalness','Liveness','Popularity','Speechiness','Valence'])
    if crtPlaylist!='Select' and a!=[]:
        # test_plotly('df_example_01-Copy1.csv',a)
        display_plotly(gen_hists(dataPL, tabTags, a))
    
    return None

def recommandation():
    #Récupération des noms de playlists et de leur ID
    sp = get_spotipy_ready()
    playlists = sp.current_user_playlists()
    name_playlists, id_playlists = get_playlists(playlists["items"])
    st.title('SpotData')
    st.header('Recommandation de playlists')
    st.subheader('A partir de vos playlists, nous vous en proposons des nouvelles, de plusieurs manières différentes.')

    #Choix de la playlist
    st.subheader('A partir de quelle playlist souhaitez-vous en obtenir une nouvelle?')
    playlist_to_change=st.selectbox('Vos playlists: ',['<select>']+name_playlists)
    if playlist_to_change!='<select>': #Une playlist a été selectionnée

        #On récupère l'ID de la playlist à changer
        id_to_change=name_to_id(name_playlists,id_playlists,playlist_to_change)
        
        #Choix du type de recommandation
        recommandation_type=st.selectbox('Quelle type de recommandation voulez-vous ?',['<select>',"Playlist soufflée", "Recommandation par années"])
        
        #recommandation de playlist soufflée
        if recommandation_type=="Playlist soufflée":
            st.subheader('Playlist soufflée')
            st.write("Pour chaque morceau de la playlist, ce type de recommandation va chercher un autre morceau avec le même artiste et certaines audiofeatures (choisies par l'utilisateur) similaires.")
            st.subheader("Choississez des audiofeatures à garder similaires dans la nouvelle playlist")
            st.write("N'en choisissez pas trop, la recommandation serait bien plus compliquée ! (et nous serions obligés de vous créer une playlist avec moins de musiques ...)")
            audiofeatures_chosen=st.multiselect('Audio-Features',['danceability','energy','speechiness','acousticness','instrumentalness','popularity','valence'])
            if audiofeatures_chosen!=[]: #Des audiofeatures ont été choisits

                #On a la playlist recommandée
                nouvelle_playlist = recommandation_souflee(id_to_change,audiofeatures_chosen,sp)

                #Affichage des morceaux recommandés
                affichage_playlist(nouvelle_playlist,sp)

                # Création de la playlist sur Spotify
                ajout_playlist_sur_spotify(nouvelle_playlist,sp,playlist_to_change)

        elif recommandation_type=="Recommandation par années":
            st.subheader('Recommandation par années')
            st.write("Chaque morceau de la playlist sera remplacé par un morceau paru autour de la date selectionnée avec des audiofeatures similaires")

            st.subheader("Choississez une année cible")
            year = st.text_input('Année choisie', value="1970")

            st.subheader("Choississez un delta d'années")
            st.write("Par exemple, si vous sélectionner 1970 précédement et 5 ici, nous vous recommanderons des titres entre 1965 et 1975.")
            delta = st.slider ("Delta d'années", min_value=0, max_value=10,value=5,step = 1)

            nouvelle_playlist = recommendation_year(id_to_change,year,delta,sp)
            affichage_playlist_annees(nouvelle_playlist,sp)
            ajout_playlist_sur_spotify(nouvelle_playlist,sp,playlist_to_change)
        
        else:
            st.subheader('Playlist soufflée')
            st.write("Pour chaque morceau de la playlist, ce type de recommandation va chercher un autre morceau avec le même artiste et certaines audiofeatures (choisies par l'utilisateur) similaires.")
            st.subheader('Recommandation par années')
            st.write("Chaque morceau de la playlist sera remplacé par un morceau paru autour de la date selectionnée avec des audiofeatures similaires")
    return None

def glossaire():
    st.title('SpotData')
    st.header('Glossaire')
    st.write('Vous trouverez sur cette page une définition des différents audiofeatures, donnée par la documentation de Spotify. La plupart de ces indices sont mesurés entre 0 et 1.')

    st.markdown("___Acousticness___")
    st.write("L'acousticness mesure l'acoustique du morceau. Plus il est proche de 1, plus il y a de chance que le morceau soit acoustique (avec des instruments non synthétiques).")
    st.write("Exemple : _Believer_ d'Imagine Dragons est principalement constitué de sons synthétiques, alors que _All of Me_ de John Legend est, lui, très acoustique.")
    st.markdown("___Danceability___")
    st.write("La dansabilité mesure si un morceau est adapté à la danse à partir d'élements musicaux comme le tempo, la stabilité du rythme, la force de la rythmique. A 0, un morceau est très peu adapté à la danse alors qu'à 1 est un morceau très dansable.")
    st.write("Exemple : _Fix You_ de Coldplay ne permet pas de danser énormément, tandis que _Billie Jean_ de Michael Jackson bien plus.")
    st.markdown("___Energy___")
    st.write("L'énergie mesure la perception de l'intensité et de l'activité. Typiquement, les morceaux énergiques semblent rapides, forts et bruyants. Par exemple, le death metal est à haute énergie alors qu'un prélude de Bach est bas dans cette échelle. Les caractéristiques qui servent à créer cette donnée sont la sonorité perçue, le timbre, la vitesse d'apparition et l'entropie générale.")
    st.write("Exemple : _when the party's over_ de Billie Eilish est très peu énergique tandis que _Highway to Hell_ de ACDC score 0.913 en énergie.")
    st.markdown("___Instrumentalness___")
    st.write("Cet indice mesure si un morceau contient des voix. Plus l'indice est proche de 1, plus il est certain qu'il n'y a aucune voix. Une valeur au dessus de 0.5 tend à être un morceau instrumental mais la probabilité augmente quand on s'approche de 1.")
    st.write("Exemple : _Ain't No Moutain High Enough_ de Marvin Gaye a une instrumentalité de 0, alors que les _Préludes_ de Rachmaninoff sont plutôt de l'ordre de 0.8.")
    st.markdown("___Liveness___")
    st.write("La liveness permet de détecter la présence d'un public dans l'enregistrement. Plus l'indice est proche de 1, plus il est probable que le morceau est un live. Au dessus de 0.8, il est quasiment certain que l'enregistrement a été fait en live.")
    st.markdown("___Loudness___")
    st.write("Le bruit est le seul paramètre qui n'est pas mesuré entre 0 et 1 mais sur une échelle de -60 dB à 0dB. Cette mesure est moyennée sur l'ensemble.")
    st.markdown("___Speechiness___")
    st.write("Cet indice détecte la présence de parties parlées. Les enregistrements parlés (type poèmes, talk show, podcast) seront proches de 1. Au dessus de 0.66, l'enregistrement est probablement entièrement consitué de paroles. Entre 0.33 et 0.66, il y aura des paroles et de la musique, soit en alternance, soit en même temps comme le rap. En dessous de 0.33, il est probable que l'enregistrement soit de la musique, sans parties parlées.")
    st.write("Exemple : _All I Want for Christmas Is You_ de Mariah Carey est très peu \"speechy\", tandis que _How Could You Leave Us_ de NF, étant majoritairement du rap, a un score de 0.758.")
    st.markdown("___Valence___")
    st.write("Plus la valence est haute, plus le morceau est joyeux. A l'inverse, plus la valence est faible, plus le morceau est triste.")
    st.write("Exemple : _Happier_ de Ed Sheeran a une valence proche de 0, tandis que _Uptown Funk_ de Mark Ronson possède une valence proche de 1.")

    st.subheader("Pour avoir plus d'information,")
    st.write("[Visitez ce site](https://rpubs.com/PeterDola/SpotifyTracks)")
    # st.table(pd_gloss)
    return None

def apropos():
    st.title('SpotData')
    st.header('A propos')
    st.write('SpotData est une WebApp développée par 4 élèves de l\'école Mines ParisTech (J.Delaplace, E.Lei, C.Nothhelfer, P.Vehrlé) dans le cadre d\'un projet d\'informatique.')
    st.write("Le but est d\'analyser des playlists Spotify et de proposer des recommandations grâce au module python Spotipy. Cette application est développée avec Streamlit.")
    st.write('[Dépot Github du projet](https://github.com/EliseLune/SpotData)')
    st.write('[Site des Mines](https://www.minesparis.psl.eu/)')
    return None

st.set_page_config(page_title='SpotData')

#Mise en lien de plusieurs pages et créations de la barre verticale de menu
app = MultiApp()
app.add_app("Se déconnecter", accueil)
app.add_app("Accueil", apres_auth)
app.add_app("Analyse", analyse)
app.add_app("Recommandations", recommandation)
app.add_app("Glossaire",glossaire)
app.add_app("A propos",apropos)
app.run()
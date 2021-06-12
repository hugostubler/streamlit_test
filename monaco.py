import pandas as pd 
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import datetime
#from sklearn import metrics
#from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pydeck
st.title('Winter series Monaco Day 1 ')

st.write("race 1")

import gpxpy
gpx = gpxpy.parse(open('activity_6224982513.gpx'))

print("{} track(s)".format(len(gpx.tracks)))
track = gpx.tracks[0]

print("{} segment(s)".format(len(track.segments)))
segment = track.segments[0]

print("{} point(s)".format(len(segment.points)))

data = []
segment_length = segment.length_3d()
for point_idx, point in enumerate(segment.points):
    data.append([point.longitude, point.latitude,
                 point.elevation, point.time, segment.get_speed(point_idx)])
    
from pandas import DataFrame

columns = ['lon', 'lat', 'Altitude', 'Time', 'Speed']
df = DataFrame(data, columns=columns)
#df.head()
#st.map(df, radiusMinPixels=3)

st.pydeck_chart(pydeck.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pydeck.ViewState(
        latitude=df['lat'][200],
         longitude=df['lon'][200],
         zoom=13.5,
     ),
    layers=[
         pydeck.Layer(
             'ScatterplotLayer',
             data=df,
             get_position='[lon, lat]',
             get_color='[200, 30, 0, 160]',
             get_radius=8,
         ),
     ],
 ))



Upwind = []
Downwind = []
for t in range(len(df)-1):
    if df.lon[t+1] - df.lon[t] > 0: 
        Upwind.append(t)
    elif df.lon[t+1] - df.lon[t] < 0: 
        Downwind.append(t)

def tribord_babord(liste):
    u = df.iloc[liste][df.Speed > 2.2].lat.reset_index()['lat']
    tribord = []
    babord = []
    for t in range(len(u)-1):
        if u[t+1]-u[t] > 0:
            tribord.append(t)
        else: 
            babord.append(t)
    return tribord, babord


mode = st.sidebar.selectbox(
    'select mode',
     ["upwind", "downwind"]
)

average = int(st.slider("Select a moving average: ", min_value=1,   
                       max_value=15,value=5, step=1))


upwind = df.iloc[Upwind][df.Speed > 2.2].Speed.rolling(window=int(f"{average}")).mean()
downwind = df.iloc[Downwind][df.Speed > 3].Speed.rolling(window=int(f"{average}")).mean()

starboard_upwind = df.iloc[tribord_babord(Upwind)[0]][df.Speed > 3.5].Speed.rolling(window=int(f"{average}")).mean().reset_index()['Speed']
port_upwind = df.iloc[tribord_babord(Upwind)[1]][df.Speed > 3.5].Speed.rolling(window=int(f"{average}")).mean().reset_index()['Speed']
starboard_downwind = df.iloc[tribord_babord(Downwind)[0]][df.Speed > 2.2].Speed.rolling(window=int(f"{average}")).mean().reset_index()['Speed']
port_downwind = df.iloc[tribord_babord(Downwind)[1]][df.Speed > 2.2].Speed.rolling(window=int(f"{average}")).mean().reset_index()['Speed']

UP = pd.DataFrame(starboard_upwind, port_upwind, columns=["starboard", "port"])
DOWN = pd.DataFrame(starboard_downwind, port_downwind, columns=["starboard", "port"])


if mode == "upwind":
    st.write("Upwind speed")
    st.line_chart(upwind*2)
    fig, ax = plt.subplots()
    ax.plot(starboard_downwind)
    ax.plot(port_downwind)
    st.pyplot(fig)

    u = df.iloc[Upwind][df.Speed > 2.2].lat.reset_index()['lat']
    tack = []
    for t in range(2,len(df.iloc[Upwind][df.Speed > 2.2])-1):
        if u[t-1]>u[t]:
            if u[t+1]>u[t]:
                tack.append(t)
        if u[t-1]<u[t]:
            if u[t+1]<u[t]:
                tack.append(t)

    MEAN = pd.DataFrame()
    for t in tack: 
        MEAN[f"{t}"] = df.Speed[t-10 : t+10].rolling(window=3).mean().reset_index()["Speed"]
    
    fig, ax = plt.subplots()
    ax.plot(MEAN.mean(axis = 1)*2)
    st.pyplot(fig)
    




if mode == "downwind":
    st.write("Downwind speed")
    st.line_chart(downwind)
    fig, ax = plt.subplots()
    ax.plot(starboard_upwind)
    ax.plot(port_upwind)
    st.pyplot(fig)


MEAN = pd.DataFrame()
for t in tack: 
    MEAN[f"{t}"] = df.Speed[t-10 : t+10].rolling(window=3).mean().reset_index()["Speed"]
MEAN.mean(axis = 1).plot()
   





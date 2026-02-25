import pylast
import pandas as pd

Client_ID = "f3bf1e5bc918d0723152a0a83344b14f"
Client_Secret = "91a593127b723eec50f2e2b5f0380eec"
Username = 'gsucheck'
network = pylast.LastFMNetwork(api_key=Client_ID, api_secret=Client_Secret, username=Username)

user = network.get_user(Username)

# Recent Tracks

recents = user.get_recent_tracks(limit=1000)

data = []

for item in recents:
    artist = network.get_artist(item.track.artist.name)
    tags = artist.get_top_tags(limit=5)
    tag_names = [tag.item.name.lower() for tag in tags]

    data.append({
        'track' : item.track.title,
        'artist' : item.track.artist.name,
        'timestamp' : int(item.timestamp),
        'tags' : tag_names
    })

df = pd.DataFrame(data)
df.head()

df.to_csv('listening_history.csv', index=False)
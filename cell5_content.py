# -- 5. Load lyrics from Genius Song Lyrics + Spotify datasets --
# Attach both datasets in Kaggle sidebar -> Add Data:
#   1. "Genius Song Lyrics" by carlosgdcj
#   2. "900K Spotify" by devdope

import json, os, re
import pandas as pd
from pathlib import Path

out_path = f'data/raw/{GENRE}.jsonl'
os.makedirs('data/raw', exist_ok=True)

TRAP_ARTISTS = {
    'future', 'young thug', 'gunna', 'lil baby', '21 savage', 'roddy ricch',
    'offset', 'playboi carti', 'travis scott', 'metro boomin', 'lil uzi vert',
    'dababy', 'polo g', 'lil durk', 'nba youngboy', 'juice wrld', 'pop smoke',
    'trippie redd', 'ynw melly', 'lil tecca', 'kodak black', 'a boogie wit da hoodie',
    'nav', 'quavo', 'takeoff', 'migos', 'moneybagg yo', 'key glock',
    'pooh shiesty', 'big30', 'rylo rodriguez', 'nle choppa',
}

TRAP_TAGS = {'trap', 'rap', 'hip hop', 'hip-hop', 'drill', 'mumble rap'}

def is_trap_row(artist, genre_tag):
    if genre_tag and any(g in str(genre_tag).lower() for g in TRAP_TAGS):
        return True
    return any(a in str(artist).lower() for a in TRAP_ARTISTS)

def clean_lyrics(raw):
    text = re.sub(r'\[.*?\]', '', str(raw))
    text = text.strip()
    return text

songs = []

csv_files = sorted(Path('/kaggle/input').rglob('*.csv'))
print(f'CSV files found: {len(csv_files)}')
for f in csv_files:
    print(f'  {f}')

if not csv_files:
    raise RuntimeError(
        'No dataset attached!\n'
        'Kaggle sidebar -> Add Data -> search "Genius Song Lyrics" by carlosgdcj'
    )

for fpath in csv_files:
    try:
        print(f'\nReading {fpath.name}...')
        df = pd.read_csv(fpath, on_bad_lines='skip')
        print(f'  Shape: {df.shape}')
        print(f'  Columns: {list(df.columns)}')

        cols_lower = {c.lower(): c for c in df.columns}

        lyric_col  = cols_lower.get('lyrics') or cols_lower.get('lyric') or cols_lower.get('text')
        artist_col = cols_lower.get('artist') or cols_lower.get('artist_name')
        title_col  = cols_lower.get('title') or cols_lower.get('song') or cols_lower.get('song_name')
        genre_col  = cols_lower.get('tag') or cols_lower.get('genre')
        lang_col   = cols_lower.get('language')

        if lyric_col is None:
            print(f'  No lyrics column - skipping')
            continue

        print(f'  lyrics={lyric_col}, artist={artist_col}, title={title_col}, genre={genre_col}')

        if lang_col:
            df = df[df[lang_col].str.lower().fillna('') == 'en']
            print(f'  After English filter: {len(df)} rows')

        for _, row in df.iterrows():
            lyrics = clean_lyrics(row[lyric_col])
            if len(lyrics.split()) < 80:
                continue

            artist    = str(row[artist_col]) if artist_col else 'unknown'
            title     = str(row[title_col])  if title_col  else 'unknown'
            genre_tag = str(row[genre_col])  if genre_col  else ''

            if GENRE == 'trap' and not is_trap_row(artist, genre_tag):
                continue

            songs.append({'artist': artist, 'title': title, 'genre': GENRE, 'lyrics': lyrics})

            if len(songs) >= 15000:
                break

        print(f'  Matched so far: {len(songs)} songs')
        if len(songs) >= 15000:
            break

    except Exception as e:
        print(f'  Error reading {fpath.name}: {e}')

with open(out_path, 'w', encoding='utf-8') as f:
    for r in songs:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'\nTotal songs saved: {len(songs)}')
print(f'Saved to: {out_path}')
if len(songs) < 1000:
    print('WARNING: few songs - check dataset is attached and has a lyrics column')

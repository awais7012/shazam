import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter
import hashlib
import pickle
import os
from pathlib import Path
import requests
from tqdm import tqdm
import json

# -------------------------
# Parameters
# -------------------------
SAMPLE_RATE = 22050
FFT_WINDOW = 2048
HOP_SIZE = 512
PEAK_NEIGHBORHOOD_SIZE = 20
FAN_VALUE = 5
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
FINGERPRINT_REDUCTION = 20

DATABASE_FILE = "fingerprint_database.pkl"
SONGS_FOLDER = "songs"
CLIPS_FOLDER = "clips"

# -------------------------
# Audio Processing (Same as before)
# -------------------------
def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def get_spectrogram(y):
    S = np.abs(librosa.stft(y, n_fft=FFT_WINDOW, hop_length=HOP_SIZE))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

def get_peaks(S):
    local_max = maximum_filter(S, size=(PEAK_NEIGHBORHOOD_SIZE, PEAK_NEIGHBORHOOD_SIZE))
    threshold = np.percentile(S, 75)
    peaks = (S == local_max) & (S > threshold)
    peak_coords = np.argwhere(peaks)
    peak_amplitudes = S[peaks]
    sorted_indices = np.argsort(peak_amplitudes)[::-1]
    n_peaks = min(len(peak_coords), len(peak_coords) // FINGERPRINT_REDUCTION + 100)
    peak_coords = peak_coords[sorted_indices[:n_peaks]]
    return [(int(freq), int(time)) for freq, time in peak_coords]

def generate_hashes(peaks, fan_value=FAN_VALUE):
    hashes = []
    peaks_sorted = sorted(peaks, key=lambda x: x[1])
    for i, anchor in enumerate(peaks_sorted):
        for j in range(1, fan_value + 1):
            if (i + j) < len(peaks_sorted):
                target = peaks_sorted[i + j]
                t_delta = target[1] - anchor[1]
                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    hash_input = f"{anchor[0]}|{target[0]}|{t_delta}"
                    h = hashlib.sha1(hash_input.encode('utf-8')).hexdigest()[:20]
                    hashes.append((h, anchor[1]))
    return hashes

# -------------------------
# API Music Downloaders
# -------------------------

class MusicDownloader:
    """Download music from various sources"""
    
    @staticmethod
    def search_and_download_jamendo(query, num_songs=5, output_folder=SONGS_FOLDER):
        """
        Download from Jamendo API (Free music, no API key needed)
        """
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n🎵 Searching Jamendo for: '{query}'")
        
        # Jamendo API endpoint (free, no auth required for basic search)
        url = "https://api.jamendo.com/v3.0/tracks/"
        params = {
            'client_id': 'b6747d04',  # Public demo client_id
            'format': 'json',
            'limit': num_songs,
            'search': query,
            'audioformat': 'mp32'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'results' not in data or len(data['results']) == 0:
                print("❌ No songs found!")
                return []
            
            downloaded_files = []
            
            print(f"✓ Found {len(data['results'])} songs. Downloading...\n")
            
            for track in data['results']:
                song_name = f"{track['artist_name']}_{track['name']}".replace(' ', '_')
                song_name = ''.join(c for c in song_name if c.isalnum() or c == '_')[:50]
                output_path = os.path.join(output_folder, f"{song_name}.mp3")
                
                if os.path.exists(output_path):
                    print(f"✓ {song_name} already exists")
                    downloaded_files.append(output_path)
                    continue
                
                audio_url = track['audio']
                
                try:
                    print(f"⬇️  Downloading: {track['name']} by {track['artist_name']}")
                    audio_response = requests.get(audio_url, stream=True, timeout=30)
                    audio_response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        for chunk in audio_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"   ✓ Saved to {output_path}")
                    downloaded_files.append(output_path)
                    
                except Exception as e:
                    print(f"   ✗ Failed: {e}")
            
            return downloaded_files
            
        except Exception as e:
            print(f"❌ API Error: {e}")
            return []
    
    @staticmethod
    def search_and_download_freemusicarchive(genre='electronic', num_songs=5, output_folder=SONGS_FOLDER):
        """
        Download from Free Music Archive curated list
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Curated free songs by genre
        songs_by_genre = {
            'electronic': [
                ('https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Kevin_MacLeod/Impact/Kevin_MacLeod_-_Wallpaper.mp3', 'Kevin_MacLeod_Wallpaper'),
                ('https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Broke_For_Free/Directionless_EP/Broke_For_Free_-_01_-_Night_Owl.mp3', 'Broke_For_Free_Night_Owl'),
            ],
            'classical': [
                ('https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chris_Zabriskie/Preludes/Chris_Zabriskie_-_01_-_Prelude_No_1.mp3', 'Chris_Zabriskie_Prelude'),
            ],
            'acoustic': [
                ('https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Jason_Shaw/Audionautix_Acoustic/Jason_Shaw_-_Acoustic_Alchemy.mp3', 'Jason_Shaw_Acoustic'),
                ('https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Scott_Holmes/Inspiring__Upbeat_Music/Scott_Holmes_-_Upbeat_Party.mp3', 'Scott_Holmes_Upbeat'),
            ]
        }
        
        songs = songs_by_genre.get(genre.lower(), songs_by_genre['electronic'])
        downloaded_files = []
        
        print(f"\n🎵 Downloading {min(num_songs, len(songs))} {genre} songs from FMA...\n")
        
        for url, name in songs[:num_songs]:
            output_path = os.path.join(output_folder, f"{name}.mp3")
            
            if os.path.exists(output_path):
                print(f"✓ {name} already exists")
                downloaded_files.append(output_path)
                continue
            
            try:
                print(f"⬇️  Downloading: {name}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"   ✓ Saved")
                downloaded_files.append(output_path)
            except Exception as e:
                print(f"   ✗ Failed: {e}")
        
        return downloaded_files

# -------------------------
# Database Management
# -------------------------
class FingerprintDatabase:
    def __init__(self):
        self.database = defaultdict(list)
        self.song_list = []
    
    def add_song(self, song_id, file_path):
        """Add a song to the database"""
        y, sr = load_audio(file_path)
        if y is None:
            return False
        
        S = get_spectrogram(y)
        peaks = get_peaks(S)
        hashes = generate_hashes(peaks)
        
        for h, t in hashes:
            self.database[h].append((song_id, t))
        
        self.song_list.append(song_id)
        return True
    
    def scan_folder(self, folder_path):
        """Scan folder and add all audio files"""
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Folder {folder_path} does not exist!")
            return
        
        audio_files = [f for f in folder.glob('**/*') if f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print(f"❌ No audio files found in {folder_path}")
            return
        
        print(f"\n📂 Found {len(audio_files)} audio files. Building database...")
        
        for audio_file in tqdm(audio_files, desc="Processing"):
            song_id = audio_file.stem
            success = self.add_song(song_id, str(audio_file))
            if not success:
                print(f"  ✗ Failed: {song_id}")
        
        print(f"✓ Database built with {len(self.song_list)} songs!\n")
    
    def match_query(self, file_path, min_match_count=5):
        """Match a query audio clip"""
        y, sr = load_audio(file_path)
        if y is None:
            return None, 0
        
        S = get_spectrogram(y)
        peaks = get_peaks(S)
        hashes = generate_hashes(peaks)
        
        matches = defaultdict(list)
        
        for h, query_time in hashes:
            if h in self.database:
                for song_id, song_time in self.database[h]:
                    offset = song_time - query_time
                    matches[song_id].append(offset)
        
        if not matches:
            return None, 0
        
        best_song = None
        best_count = 0
        
        for song_id, offsets in matches.items():
            offset_counts = Counter(offsets)
            most_common_offset, count = offset_counts.most_common(1)[0]
            
            if count > best_count:
                best_count = count
                best_song = song_id
        
        if best_count < min_match_count:
            return None, 0
        
        return best_song, best_count
    
    def save(self, filename=DATABASE_FILE):
        """Save database to disk"""
        with open(filename, 'wb') as f:
            pickle.dump({'database': dict(self.database), 'song_list': self.song_list}, f)
        print(f"💾 Database saved to {filename}")
    
    def load(self, filename=DATABASE_FILE):
        """Load database from disk"""
        if not os.path.exists(filename):
            print(f"❌ Database file {filename} not found")
            return False
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.database = defaultdict(list, data['database'])
            self.song_list = data['song_list']
        
        print(f"✓ Database loaded with {len(self.song_list)} songs")
        return True

# -------------------------
# Main Interface
# -------------------------
def main():
    print("=" * 70)
    print("  🎵 AUDIO FINGERPRINTING SYSTEM (Shazam-like)")
    print("=" * 70)
    
    db = FingerprintDatabase()
    
    # Try to load existing database
    if os.path.exists(DATABASE_FILE):
        print("\n[1] Load existing database")
        print("[2] Build new database from API")
        print("[3] Build from local folder")
        choice = input("\nChoose (1/2/3): ").strip()
        
        if choice == '1':
            db.load()
        elif choice == '2':
            build_from_api(db)
        else:
            build_from_folder(db)
    else:
        print("\n❌ No existing database found.")
        print("\n[1] Download songs from API and build database")
        print("[2] Scan local folder")
        choice = input("\nChoose (1/2): ").strip()
        
        if choice == '1':
            build_from_api(db)
        else:
            build_from_folder(db)
    
    # Matching mode
    if len(db.song_list) > 0:
        matching_mode(db)
    else:
        print("\n❌ No songs in database. Exiting.")

def build_from_api(db):
    """Download songs from API and build database"""
    print("\n" + "=" * 70)
    print("  📥 DOWNLOAD FROM API")
    print("=" * 70)
    print("\n[1] Jamendo API - Search any music (recommended)")
    print("[2] Free Music Archive - By genre")
    
    choice = input("\nChoose (1/2): ").strip()
    
    if choice == '1':
        query = input("Enter search query (artist, genre, song): ").strip()
        if not query:
            query = "electronic"
        
        num = input("How many songs? (default: 5): ").strip()
        num = int(num) if num.isdigit() else 5
        
        downloader = MusicDownloader()
        files = downloader.search_and_download_jamendo(query, num)
        
        if files:
            print(f"\n✓ Downloaded {len(files)} songs!")
            db.scan_folder(SONGS_FOLDER)
            db.save()
        else:
            print("\n❌ No songs downloaded!")
    
    else:
        genre = input("Enter genre (electronic/classical/acoustic): ").strip()
        if not genre:
            genre = "electronic"
        
        num = input("How many songs? (default: 3): ").strip()
        num = int(num) if num.isdigit() else 3
        
        downloader = MusicDownloader()
        files = downloader.search_and_download_freemusicarchive(genre, num)
        
        if files:
            print(f"\n✓ Downloaded {len(files)} songs!")
            db.scan_folder(SONGS_FOLDER)
            db.save()

def build_from_folder(db):
    """Build database from local folder"""
    folder = input(f"Enter folder path (default: {SONGS_FOLDER}): ").strip()
    if not folder:
        folder = SONGS_FOLDER
    db.scan_folder(folder)
    db.save()

def matching_mode(db):
    """Interactive matching mode"""
    print("\n" + "=" * 70)
    print("  🔍 MATCHING MODE")
    print("=" * 70)
    
    while True:
        print("\nOptions:")
        print("  [1] Match a single file")
        print("  [2] Match all files in clips folder")
        print("  [3] Exit")
        
        choice = input("\nChoose (1/2/3): ").strip()
        
        if choice == '1':
            file_path = input("Enter audio file path: ").strip()
            if os.path.exists(file_path):
                print(f"\n🔍 Matching: {file_path}")
                match, confidence = db.match_query(file_path)
                if match:
                    print(f"\n✅ MATCH FOUND: {match} (confidence: {confidence})")
                else:
                    print("\n❌ No match found")
            else:
                print("❌ File not found!")
        
        elif choice == '2':
            if not os.path.exists(CLIPS_FOLDER):
                print(f"❌ Clips folder '{CLIPS_FOLDER}' not found!")
            else:
                match_all_clips(db, CLIPS_FOLDER)
        
        elif choice == '3':
            print("\n👋 Goodbye!")
            break

def match_all_clips(db, clips_folder):
    """Match all clips in a folder"""
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
    clips = [f for f in Path(clips_folder).glob('*') if f.suffix.lower() in audio_extensions]
    
    if not clips:
        print(f"❌ No audio clips found in {clips_folder}")
        return
    
    print(f"\n🔍 Matching {len(clips)} clips...\n")
    
    for clip in clips:
        match, confidence = db.match_query(str(clip))
        if match:
            print(f"✅ {clip.name} -> {match} (confidence: {confidence})")
        else:
            print(f"❌ {clip.name} -> No match")

if __name__ == "__main__":
    main()
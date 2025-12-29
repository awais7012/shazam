"""
Audio Fingerprinting System - Fixed Matching
This creates fingerprints from a folder of songs and matches audio clips
"""

import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter
import hashlib
import pickle
import os
from pathlib import Path
from tqdm import tqdm

# -------------------------
# Parameters (TUNED FOR BETTER MATCHING)
# -------------------------
SAMPLE_RATE = 22050
FFT_WINDOW = 4096  # Increased for better frequency resolution
HOP_SIZE = FFT_WINDOW // 4
PEAK_NEIGHBORHOOD_SIZE = 10
FAN_VALUE = 15  # More connections for better matching
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
FINGERPRINT_REDUCTION = 10  # Keep more peaks

DATABASE_FILE = "fingerprint_database.pkl"
SONGS_FOLDER = "songs"
CLIPS_FOLDER = "clips"

# -------------------------
# Audio Processing
# -------------------------
def load_audio(file_path):
    """Load audio file"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def get_spectrogram(y):
    """Generate spectrogram"""
    S = np.abs(librosa.stft(y, n_fft=FFT_WINDOW, hop_length=HOP_SIZE))
    return S

def get_peaks(S):
    """Find peaks in spectrogram - IMPROVED"""
    # Apply maximum filter
    struct_size = (PEAK_NEIGHBORHOOD_SIZE, PEAK_NEIGHBORHOOD_SIZE)
    local_max = maximum_filter(S, size=struct_size)
    
    # Peaks are local maxima above median
    background = np.median(S)
    peaks = (S == local_max) & (S > background * 2)  # 2x median threshold
    
    # Get peak coordinates and amplitudes
    peak_coords = np.argwhere(peaks)
    if len(peak_coords) == 0:
        return []
    
    peak_amplitudes = S[tuple(peak_coords.T)]
    
    # Sort by amplitude and keep strongest peaks
    sorted_indices = np.argsort(peak_amplitudes)[::-1]
    
    # Dynamic peak limit based on signal length
    max_peaks = len(S[0]) * 5  # ~5 peaks per time frame
    n_peaks = min(len(peak_coords), max_peaks)
    
    peak_coords = peak_coords[sorted_indices[:n_peaks]]
    
    return [(int(freq), int(time)) for freq, time in peak_coords]

def generate_hashes(peaks, fan_value=FAN_VALUE):
    """Generate fingerprint hashes - IMPROVED"""
    if len(peaks) == 0:
        return []
    
    hashes = []
    peaks_sorted = sorted(peaks, key=lambda x: x[1])  # sort by time
    
    for i, anchor in enumerate(peaks_sorted):
        # Connect to multiple future peaks
        for j in range(1, fan_value + 1):
            if (i + j) < len(peaks_sorted):
                target = peaks_sorted[i + j]
                t_delta = target[1] - anchor[1]
                
                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    # Create hash: anchor_freq|target_freq|time_delta
                    hash_input = f"{anchor[0]}|{target[0]}|{t_delta}"
                    h = hashlib.sha1(hash_input.encode('utf-8')).hexdigest()[:20]
                    hashes.append((h, anchor[1]))
    
    return hashes

# -------------------------
# Database Class
# -------------------------
class FingerprintDatabase:
    def __init__(self):
        self.database = defaultdict(list)
        self.song_list = []
    
    def add_song(self, song_id, file_path):
        """Add a song to the database"""
        print(f"  Processing: {song_id}")
        
        y, sr = load_audio(file_path)
        if y is None:
            return False
        
        S = get_spectrogram(y)
        peaks = get_peaks(S)
        
        if len(peaks) == 0:
            print(f"    ⚠️ No peaks found!")
            return False
        
        hashes = generate_hashes(peaks)
        
        if len(hashes) == 0:
            print(f"    ⚠️ No hashes generated!")
            return False
        
        print(f"    ✓ Peaks: {len(peaks)}, Hashes: {len(hashes)}")
        
        # Add to database
        for h, t in hashes:
            self.database[h].append((song_id, t))
        
        self.song_list.append(song_id)
        return True
    
    def scan_folder(self, folder_path):
        """Scan folder and add all audio files"""
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'}
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Folder '{folder_path}' does not exist!")
            print(f"   Please create it and add your songs there.")
            return False
        
        audio_files = [f for f in folder.glob('**/*') if f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print(f"❌ No audio files found in '{folder_path}'")
            print(f"   Supported formats: {', '.join(audio_extensions)}")
            return False
        
        print(f"\n📂 Found {len(audio_files)} audio files")
        print(f"{'='*60}")
        
        success_count = 0
        for audio_file in audio_files:
            song_id = audio_file.stem
            if self.add_song(song_id, str(audio_file)):
                success_count += 1
        
        print(f"{'='*60}")
        print(f"✓ Successfully processed {success_count}/{len(audio_files)} songs")
        print(f"✓ Total fingerprints in database: {len(self.database)}\n")
        return True
    
    def match_query(self, file_path, min_match_count=5):
        """Match a query audio clip - IMPROVED"""
        print(f"\n🔍 Analyzing: {Path(file_path).name}")
        
        y, sr = load_audio(file_path)
        if y is None:
            return None, 0, {}
        
        S = get_spectrogram(y)
        peaks = get_peaks(S)
        
        if len(peaks) == 0:
            print("   ⚠️ No peaks found in clip")
            return None, 0, {}
        
        hashes = generate_hashes(peaks)
        
        if len(hashes) == 0:
            print("   ⚠️ No hashes generated from clip")
            return None, 0, {}
        
        print(f"   Generated {len(hashes)} fingerprints from clip")
        
        # Match against database
        matches = defaultdict(list)
        matched_hashes = 0
        
        for h, query_time in hashes:
            if h in self.database:
                matched_hashes += 1
                for song_id, song_time in self.database[h]:
                    offset = song_time - query_time
                    matches[song_id].append(offset)
        
        print(f"   Found {matched_hashes} matching fingerprints")
        
        if not matches:
            print("   ❌ No matches in database")
            return None, 0, {}
        
        # Analyze matches
        results = {}
        for song_id, offsets in matches.items():
            offset_counts = Counter(offsets)
            most_common_offset, count = offset_counts.most_common(1)[0]
            
            # Calculate match percentage
            match_percentage = (count / len(hashes)) * 100
            
            results[song_id] = {
                'count': count,
                'offset': most_common_offset,
                'percentage': match_percentage
            }
        
        # Sort by count
        sorted_results = sorted(results.items(), key=lambda x: x[1]['count'], reverse=True)
        
        if sorted_results[0][1]['count'] < min_match_count:
            print(f"   ❌ Best match below threshold ({sorted_results[0][1]['count']} < {min_match_count})")
            return None, 0, results
        
        best_song = sorted_results[0][0]
        best_count = sorted_results[0][1]['count']
        
        return best_song, best_count, results
    
    def save(self, filename=DATABASE_FILE):
        """Save database to disk"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'database': dict(self.database),
                    'song_list': self.song_list
                }, f)
            print(f"💾 Database saved to '{filename}'")
            return True
        except Exception as e:
            print(f"❌ Error saving database: {e}")
            return False
    
    def load(self, filename=DATABASE_FILE):
        """Load database from disk"""
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.database = defaultdict(list, data['database'])
                self.song_list = data['song_list']
            print(f"✓ Loaded database with {len(self.song_list)} songs")
            print(f"  Total fingerprints: {len(self.database)}")
            return True
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False

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
        print(f"\n✓ Found existing database")
        db.load()
        
        print("\n[1] Use existing database")
        print("[2] Rebuild database from songs folder")
        choice = input("\nChoose (1/2): ").strip()
        
        if choice == '2':
            db = FingerprintDatabase()  # Reset
            if db.scan_folder(SONGS_FOLDER):
                db.save()
    else:
        print(f"\n📂 Building new database from '{SONGS_FOLDER}' folder...")
        if not db.scan_folder(SONGS_FOLDER):
            print("\n❌ Cannot proceed without songs. Exiting.")
            return
        db.save()
    
    if len(db.song_list) == 0:
        print("\n❌ No songs in database. Exiting.")
        return
    
    # Matching mode
    matching_mode(db)

def matching_mode(db):
    """Interactive matching mode"""
    print("\n" + "=" * 70)
    print("  🔍 MATCHING MODE")
    print("=" * 70)
    print(f"\nDatabase contains {len(db.song_list)} songs:")
    for i, song in enumerate(db.song_list, 1):
        print(f"  {i}. {song}")
    
    while True:
        print("\n" + "-" * 70)
        print("Options:")
        print("  [1] Match a single audio file")
        print("  [2] Match all files in 'clips' folder")
        print("  [3] Show database info")
        print("  [4] Exit")
        
        choice = input("\nChoose: ").strip()
        
        if choice == '1':
            file_path = input("Enter audio file path: ").strip()
            
            if not os.path.exists(file_path):
                print("❌ File not found!")
                continue
            
            match, confidence, all_results = db.match_query(file_path)
            
            if match:
                print(f"\n{'='*60}")
                print(f"✅ MATCH FOUND!")
                print(f"{'='*60}")
                print(f"Song: {match}")
                print(f"Confidence: {confidence} matching fingerprints")
                print(f"Match Quality: {all_results[match]['percentage']:.1f}%")
                
                if len(all_results) > 1:
                    print(f"\nOther possible matches:")
                    for song, data in sorted(all_results.items(), 
                                            key=lambda x: x[1]['count'], 
                                            reverse=True)[1:4]:
                        print(f"  • {song}: {data['count']} matches ({data['percentage']:.1f}%)")
            else:
                print(f"\n❌ No confident match found")
                if all_results:
                    print(f"\nWeak matches detected:")
                    for song, data in sorted(all_results.items(), 
                                            key=lambda x: x[1]['count'], 
                                            reverse=True)[:3]:
                        print(f"  • {song}: {data['count']} matches ({data['percentage']:.1f}%)")
        
        elif choice == '2':
            match_folder(db, CLIPS_FOLDER)
        
        elif choice == '3':
            show_database_info(db)
        
        elif choice == '4':
            print("\n👋 Goodbye!")
            break

def match_folder(db, clips_folder):
    """Match all clips in a folder"""
    if not os.path.exists(clips_folder):
        print(f"\n❌ Clips folder '{clips_folder}' not found!")
        print(f"   Create this folder and add audio clips to match.")
        return
    
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'}
    clips = [f for f in Path(clips_folder).glob('*') 
             if f.suffix.lower() in audio_extensions]
    
    if not clips:
        print(f"\n❌ No audio clips found in '{clips_folder}'")
        return
    
    print(f"\n{'='*60}")
    print(f"Matching {len(clips)} clips...")
    print(f"{'='*60}")
    
    results = []
    for clip in clips:
        match, confidence, _ = db.match_query(str(clip))
        results.append((clip.name, match, confidence))
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    for clip_name, match, confidence in results:
        if match:
            print(f"✅ {clip_name:30} → {match} ({confidence})")
        else:
            print(f"❌ {clip_name:30} → No match")

def show_database_info(db):
    """Show database statistics"""
    print(f"\n{'='*60}")
    print(f"DATABASE INFO")
    print(f"{'='*60}")
    print(f"Total songs: {len(db.song_list)}")
    print(f"Total unique fingerprints: {len(db.database)}")
    
    # Calculate average fingerprints per song
    total_entries = sum(len(v) for v in db.database.values())
    avg_per_song = total_entries / len(db.song_list) if db.song_list else 0
    print(f"Average fingerprints per song: {avg_per_song:.0f}")
    
    print(f"\nSongs in database:")
    for i, song in enumerate(db.song_list, 1):
        print(f"  {i}. {song}")

if __name__ == "__main__":
    main()
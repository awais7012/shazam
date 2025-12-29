"""
Create Test Clips from Songs
This extracts random clips from your songs to test the fingerprinting system
Run this AFTER you have songs in the 'songs' folder

Usage: python create_test_clips.py
"""

import librosa
import soundfile as sf
import os
from pathlib import Path
import random
import numpy as np

SONGS_FOLDER = "songs"
CLIPS_FOLDER = "clips"
CLIP_DURATION = 10  # seconds
NUM_CLIPS_PER_SONG = 2

def create_clip(song_path, output_folder, clip_duration=CLIP_DURATION):
    """
    Extract a random clip from a song
    """
    try:
        # Load the full song
        y, sr = librosa.load(song_path, sr=22050, mono=True)
        
        total_duration = len(y) / sr
        
        if total_duration < clip_duration:
            print(f"  ⚠️ Song too short ({total_duration:.1f}s), using full song")
            clip = y
            start_time = 0
        else:
            # Random start position (avoid first and last 5 seconds)
            safe_start = int(5 * sr)
            safe_end = int((total_duration - clip_duration - 5) * sr)
            
            if safe_end <= safe_start:
                start_sample = 0
            else:
                start_sample = random.randint(safe_start, safe_end)
            
            start_time = start_sample / sr
            end_sample = start_sample + int(clip_duration * sr)
            
            # Extract clip
            clip = y[start_sample:end_sample]
        
        # Optional: Add slight noise to make it more realistic
        # noise = np.random.normal(0, 0.002, len(clip))
        # clip = clip + noise
        
        # Create output filename
        song_name = Path(song_path).stem
        clip_name = f"{song_name}_clip_{int(start_time)}s.wav"
        output_path = os.path.join(output_folder, clip_name)
        
        # Save clip
        sf.write(output_path, clip, sr)
        
        return output_path, start_time, clip_duration
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, 0, 0

def main():
    print("=" * 70)
    print("  🎵 TEST CLIP GENERATOR")
    print("=" * 70)
    
    # Check if songs folder exists
    if not os.path.exists(SONGS_FOLDER):
        print(f"\n❌ Songs folder '{SONGS_FOLDER}' not found!")
        print("   Please create it and add songs first.")
        return
    
    # Find all audio files
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'}
    songs = [f for f in Path(SONGS_FOLDER).glob('**/*') 
             if f.suffix.lower() in audio_extensions]
    
    if not songs:
        print(f"\n❌ No audio files found in '{SONGS_FOLDER}'")
        return
    
    print(f"\n✓ Found {len(songs)} songs")
    
    # Create clips folder
    os.makedirs(CLIPS_FOLDER, exist_ok=True)
    
    # Generate clips
    print(f"\n📂 Creating {NUM_CLIPS_PER_SONG} clip(s) per song...")
    print(f"   Clip duration: {CLIP_DURATION} seconds")
    print(f"   Output folder: '{CLIPS_FOLDER}'")
    print(f"\n{'='*60}")
    
    total_created = 0
    
    for song_path in songs:
        song_name = song_path.stem
        print(f"\n🎵 {song_name}")
        
        for i in range(NUM_CLIPS_PER_SONG):
            result = create_clip(str(song_path), CLIPS_FOLDER, CLIP_DURATION)
            
            if result[0]:
                clip_path, start_time, duration = result
                print(f"  ✓ Created clip: {Path(clip_path).name}")
                print(f"    From: {start_time:.1f}s to {start_time + duration:.1f}s")
                total_created += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Created {total_created} test clips in '{CLIPS_FOLDER}'")
    print(f"\nNow run the main fingerprinting script to test matching!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
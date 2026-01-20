
import os
from pathlib import Path
from tqdm import tqdm

def generate_esd_dataset():
    # Configuration
    # Assumes ESD dataset is located at ./ESD relative to this script
    base_dir = Path("ESD")
    output_dir = Path("dataset_emotion")
    
    # Emotion configuration
    # Map directory names to instruction text
    # Happy/Surprise -> Happy instruction
    # Neutral -> Neutral instruction
    emotion_instruct_map = {
        "Happy": "请以开心高兴的语气用普通话说<|endofprompt|>",
        "Surprise": "请以开心高兴的语气用普通话说<|endofprompt|>",
        "Neutral": "请以正常中立的语气用普通话说<|endofprompt|>"
    }
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize data containers
    wav_scp = []
    text_data = [] # (uttid, text)
    utt2spk = []
    instruct_data = []
    
    # Iterate through speaker directories
    # Check if base_dir exists
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist.")
        return

    # Filter for directories that look like speaker IDs (e.g., 0001, 0002)
    speaker_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if not speaker_dirs:
        print(f"No speaker directories found in {base_dir}")
        return
    
    print(f"Found {len(speaker_dirs)} speakers.")

    valid_count = 0
    missing_text_count = 0
    
    for spk_dir in tqdm(speaker_dirs, desc="Processing Speakers"):
        spk_id = spk_dir.name
        
        # Load text transcript
        # File is usually named {spk_id}.txt inside the folder
        text_file = spk_dir / f"{spk_id}.txt"
        transcript_map = {}
        
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # ESD format: ID    Text    Emotion
                    # Try splitting by tab first, then whitespace
                    # Usually: 0001_000001     打远一看，它们的确很是美丽，    中立
                    parts = line.split()
                    if len(parts) >= 2:
                        uttid = parts[0]
                        content = parts[1]
                        transcript_map[uttid] = content
        else:
            print(f"Warning: Transcript file not found for {spk_id}: {text_file}")
            continue

        # Process target emotion folders
        for emotion, instruction in emotion_instruct_map.items():
            emo_dir = spk_dir / emotion
            if not emo_dir.exists():
                # Some speakers might not have all emotions
                continue
            
            # List all wav files
            wav_files = sorted(emo_dir.glob("*.wav"))
            
            for wav_path in wav_files:
                uttid = wav_path.stem
                
                # Check if we have text for this uttid
                if uttid not in transcript_map:
                    # Sometimes files exist without transcript?
                    missing_text_count += 1
                    continue
                
                transcript = transcript_map[uttid]
                
                # Add to lists
                # Use absolute path for wav.scp
                wav_scp.append(f"{uttid} {wav_path.absolute()}")
                text_data.append(f"{uttid} {transcript}")
                utt2spk.append(f"{uttid} {spk_id}")
                instruct_data.append(f"{uttid} {instruction}")
                
                valid_count += 1

    # Write output files
    print(f"Writing output files to {output_dir}...")
    
    with open(output_dir / "wav.scp", "w", encoding="utf-8") as f:
        f.write("\n".join(wav_scp) + "\n")
        
    with open(output_dir / "text", "w", encoding="utf-8") as f:
        f.write("\n".join(text_data) + "\n")
        
    with open(output_dir / "utt2spk", "w", encoding="utf-8") as f:
        f.write("\n".join(utt2spk) + "\n")
        
    with open(output_dir / "instruct.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(instruct_data) + "\n")
        
    # Generate spk2utt
    spk_map = {}
    for entry in utt2spk:
        uid, sid = entry.split()
        if sid not in spk_map:
            spk_map[sid] = []
        spk_map[sid].append(uid)
    
    with open(output_dir / "spk2utt", "w", encoding="utf-8") as f:
        for sid in sorted(spk_map.keys()):
            f.write(f"{sid} {' '.join(sorted(spk_map[sid]))}\n")

    print("=" * 50)
    print(f"Processing Complete!")
    print(f"Total processed files: {valid_count}")
    print(f"Missing transcripts: {missing_text_count}")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 50)

if __name__ == "__main__":
    generate_esd_dataset()

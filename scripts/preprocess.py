import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import shutil

def calculate_window_variance(actions: List[np.ndarray], center_idx: int, window_size: int) -> np.ndarray:
    """
    Calculate the component-wise variance of actions within a window around center_idx.
    
    Args:
        actions: List of action sequences
        center_idx: Center index of the window
        window_size: Half-width of the window (n in [i-n, i+n])
    
    Returns:
        Array of variances for each action component
    """
    start_idx = max(0, center_idx - window_size)
    end_idx = min(len(actions), center_idx + window_size + 1)
    
    if end_idx - start_idx <= 1:
        return np.array([0.0])
    
    # Extract actions in the window
    window_actions = actions[start_idx:end_idx]
    
    # Convert to a 2D array where each row is a flattened action
    try:
        # Flatten each action and stack them
        flattened_actions = []
        for action in window_actions:
            if isinstance(action, np.ndarray) and action.size > 0:
                flattened_actions.append(action.flatten())
            else:
                # Handle empty or invalid actions
                if len(flattened_actions) > 0:
                    # Use the same size as previous actions, filled with zeros
                    flattened_actions.append(np.zeros_like(flattened_actions[0]))
                else:
                    # If this is the first action and it's invalid, skip
                    continue
        
        if len(flattened_actions) <= 1:
            return np.array([0.0])
        
        # Stack into 2D array (rows are time steps, columns are action components)
        action_matrix = np.stack(flattened_actions)
        
        # Calculate variance across time (axis=0) for each component separately
        component_variances = np.var(action_matrix, axis=0)
        return component_variances
        
    except Exception as e:
        # If there's any issue with the calculation, return array of zeros
        return np.array([0.0])

def find_movement_boundaries(actions: List[np.ndarray], threshold: float, window_size: int) -> Tuple[int, int]:
    """
    Find the start and end indices where noticeable movement begins and ends using component-wise variance.
    
    Args:
        actions: List of action sequences
        threshold: Variance threshold for detecting noticeable movement
        window_size: Half-width of the window for variance calculation (n in [i-n, i+n])
    
    Returns:
        Tuple of (start_index, end_index) for the trimmed episode
    """
    if len(actions) <= 2 * window_size + 1:
        return 0, len(actions)
    
    # Calculate component-wise variance for each position
    component_variances = []
    for i in range(len(actions)):
        variances = calculate_window_variance(actions, i, window_size)
        component_variances.append(variances)
    
    # Find start index - first position where ANY component variance exceeds threshold
    start_idx = 0
    for i in range(len(component_variances)):
        if np.any(component_variances[i] > threshold):
            start_idx = i
            break
    
    # Find end index - last position where ANY component variance exceeds threshold
    end_idx = len(actions)
    for i in range(len(component_variances) - 1, -1, -1):
        if np.any(component_variances[i] > threshold):
            end_idx = i + 1
            break
    
    return start_idx, end_idx

def trim_video_file(input_video_path: Path, output_video_path: Path, start_frame: int, end_frame: int) -> bool:
    """
    Trim an MP4 video file to keep only frames from start_frame to end_frame.
    
    Args:
        input_video_path: Path to input video file
        output_video_path: Path to output video file
        start_frame: Starting frame index (inclusive)
        end_frame: Ending frame index (exclusive)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Open input video
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            print(f"  Error: Could not open video {input_video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate frame indices
        if start_frame >= total_frames or end_frame <= start_frame:
            print(f"  Warning: Invalid frame range [{start_frame}, {end_frame}) for video with {total_frames} frames")
            cap.release()
            return False
        
        # Ensure output directory exists
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read and write frames in the desired range
        frame_idx = start_frame
        while frame_idx < end_frame and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_idx += 1
        
        # Clean up
        cap.release()
        out.release()
        
        print(f"  Video trimmed: {total_frames} -> {end_frame - start_frame} frames")
        return True
        
    except Exception as e:
        print(f"  Error trimming video {input_video_path}: {e}")
        return False

def find_corresponding_files(data_file_path: Path, dataset_dir: Path) -> List[Path]:
    """
    Find corresponding video files for a given data file.
    
    Args:
        data_file_path: Path to the main data parquet file
        dataset_dir: Root dataset directory
    
    Returns:
        List of MP4 file paths
    """
    # Parse the path to extract chunk number and episode number
    # Data path format: <dataset-dir>/data/chunk-000/episode_000000.parquet
    chunk_dir = data_file_path.parent.name      # e.g., chunk-000
    episode_name = data_file_path.stem          # e.g., episode_000000 (stem gets name without extension)
    
    # Construct the base directory for this specific chunk's videos
    # Video path format: <dataset-dir>/videos/chunk-000/<group>/episode_000000.mp4
    video_chunk_dir = dataset_dir / 'videos' / chunk_dir
    
    # Find all corresponding MP4 files within this chunk's video directory
    mp4_paths = []
    if video_chunk_dir.exists():
        # Search recursively within the chunk's video directory for the episode file.
        # This will find it in any <group> sub-directory.
        # Pattern from video_chunk_dir: **/episode_000000.mp4
        search_pattern = f"**/{episode_name}.mp4"
        potential_files = video_chunk_dir.rglob(search_pattern)
        mp4_paths = list(potential_files)

        if not mp4_paths:
            print(f"  Warning: No .mp4 files found matching pattern '{search_pattern}' in {video_chunk_dir}")
    else:
        print(f"  Warning: Video chunk directory not found: {video_chunk_dir}")
            
    return mp4_paths

def process_episode_file(file_path: Path, threshold: float, window_size: int, dataset_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Process a single episode file and remove static actions at beginning/end."""
    print(f"Processing: {file_path}")
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    if df.empty:
        return df
    
    # Convert action column to numpy arrays if they're not already
    actions = []
    for action in df['action']:
        if isinstance(action, (list, tuple)):
            actions.append(np.array(action))
        elif isinstance(action, np.ndarray):
            actions.append(action)
        else:
            # Handle string representation or other formats
            try:
                actions.append(np.array(action))
            except:
                print(f"Warning: Could not convert action to numpy array: {action}")
                actions.append(np.array([]))
    
    # Find movement boundaries
    start_idx, end_idx = find_movement_boundaries(actions, threshold, window_size)
    
    # Trim the dataframe
    trimmed_df = df.iloc[start_idx:end_idx].copy()
    
    # Reset frame_index to be continuous
    if not trimmed_df.empty:
        trimmed_df.loc[:, 'frame_index'] = range(len(trimmed_df))
    
    print(f"  Original length: {len(df)}, Trimmed length: {len(trimmed_df)}")
    print(f"  Removed {start_idx} frames from start, {len(df) - end_idx} frames from end")
    
    # Process corresponding video files
    mp4_paths = find_corresponding_files(file_path, dataset_dir)
    print(f"  Found {len(mp4_paths)} corresponding MP4 files")
    
    # Process MP4 files
    for mp4_path in mp4_paths:
        try:
            print(f"  Processing video: {mp4_path}")
            
            # Determine output path
            rel_path = mp4_path.relative_to(dataset_dir)
            output_mp4_path = output_dir / rel_path
            
            # Trim the video
            if trim_video_file(mp4_path, output_mp4_path, start_idx, end_idx):
                print(f"  Saved video to: {output_mp4_path}")
            else:
                print(f"  Failed to process video: {mp4_path}")
                
        except Exception as e:
            print(f"  Error processing video {mp4_path}: {e}")
    
    return trimmed_df

def update_metadata_files(dataset_dir: Path, output_dir: Path, episode_length_map: Dict[int, int], total_processed_frames: int) -> None:
    """
    Update metadata files after preprocessing to reflect new episode lengths.
    
    Args:
        dataset_dir: Original dataset directory
        output_dir: Output dataset directory
        episode_length_map: Mapping of episode_index to new length
        total_processed_frames: Total number of frames after processing
    """
    print("Updating metadata files...")
    
    # Update episodes.jsonl
    original_episodes_path = dataset_dir / 'meta' / 'episodes.jsonl'
    output_episodes_path = output_dir / 'meta' / 'episodes.jsonl'
    
    if original_episodes_path.exists():
        with open(original_episodes_path, 'r') as f:
            episodes = [json.loads(line.strip()) for line in f if line.strip()]
        
        # Update episode lengths
        for episode in episodes:
            episode_idx = episode['episode_index']
            if episode_idx in episode_length_map:
                episode['length'] = episode_length_map[episode_idx]
                print(f"  Updated episode {episode_idx}: {episode['length']} frames")
        
        # Write updated episodes.jsonl
        output_episodes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_episodes_path, 'w') as f:
            for episode in episodes:
                f.write(json.dumps(episode) + '\n')
        print(f"  Updated {output_episodes_path}")
    else:
        print(f"  Warning: {original_episodes_path} not found")
    
    # Update info.json
    original_info_path = dataset_dir / 'meta' / 'info.json'
    output_info_path = output_dir / 'meta' / 'info.json'
    
    if original_info_path.exists():
        with open(original_info_path, 'r') as f:
            info = json.load(f)
        
        # Update total frames
        info['total_frames'] = total_processed_frames
        print(f"  Updated total_frames: {total_processed_frames}")
        
        # Write updated info.json
        with open(output_info_path, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"  Updated {output_info_path}")
    else:
        print(f"  Warning: {original_info_path} not found")
    
    # Copy other metadata files that don't need updating
    files_to_copy = ['tasks.jsonl', 'modality.json', 'stats.json', 'episodes_stats.jsonl']
    
    for filename in files_to_copy:
        src_path = dataset_dir / 'meta' / filename
        dst_path = output_dir / 'meta' / filename
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"  Copied {filename}")
        else:
            print(f"  Warning: {filename} not found")

def main(args):
    dataset_dir = Path(args.dataset_dir)
    threshold = args.threshold
    window_size = args.window_size
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir.parent / f"{dataset_dir.name}_processed"
    
    # Paths to data and video directories
    data_dir = dataset_dir / 'data'
    video_dir = dataset_dir / 'video'
    
    print(f"Processing dataset directory: {dataset_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Video directory: {video_dir}")
    print(f"Variance threshold: {threshold}")
    print(f"Window size: {window_size} (neighborhood: [-{window_size}, +{window_size}])")
    print(f"Output directory: {output_dir}")
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist")
        return
    
    if not data_dir.exists():
        print(f"Error: Data subdirectory {data_dir} does not exist")
        return
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files in the expected structure
    parquet_files = list(data_dir.rglob("episode_*.parquet"))
    
    if not parquet_files:
        print(f"No episode_*.parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} episode files to process")
    
    total_original_frames = 0
    total_processed_frames = 0
    episode_length_map = {}  # Track new episode lengths
    
    for file_path in parquet_files:
        # Get relative path from dataset_dir to maintain directory structure
        rel_path = file_path.relative_to(dataset_dir)
        output_file_path = output_dir / rel_path
        
        # Create output subdirectories
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process the episode file
        try:
            original_df = pd.read_parquet(file_path)
            processed_df = process_episode_file(file_path, threshold, window_size, dataset_dir, output_dir)
            
            total_original_frames += len(original_df)
            total_processed_frames += len(processed_df)
            
            # Extract episode index from filename and store new length
            episode_filename = file_path.stem  # e.g., "episode_000000"
            if episode_filename.startswith("episode_"):
                episode_index = int(episode_filename.split("_")[1])
                episode_length_map[episode_index] = len(processed_df)
            
            # Save processed file
            if not processed_df.empty:
                processed_df.to_parquet(output_file_path, index=False)
                print(f"  Saved to: {output_file_path}")
            else:
                print(f"  Warning: Episode became empty after processing, skipping save")
                episode_length_map[episode_index] = 0
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
        
        print()  # Add blank line between episodes for readability
    
    # Update metadata files
    update_metadata_files(dataset_dir, output_dir, episode_length_map, total_processed_frames)
    
    # Copy other top-level files
    files_to_copy = ['README.md', '.gitattributes']
    for filename in files_to_copy:
        src_path = dataset_dir / filename
        dst_path = output_dir / filename
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename}")
    
    print(f"\nProcessing complete!")
    print(f"Total original frames: {total_original_frames}")
    print(f"Total processed frames: {total_processed_frames}")
    print(f"Frames removed: {total_original_frames - total_processed_frames}")
    print(f"Reduction: {(total_original_frames - total_processed_frames) / total_original_frames * 100:.2f}%")
    print(f"Updated metadata files in: {output_dir / 'meta'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process robot dataset by removing static actions at episode boundaries.\n\n" +
                   "Expected file structure:\n" +
                   "  <dataset_dir>/data/chunk-000/episode_000000.parquet\n" +
                   "  <dataset_dir>/videos/chunk-000/<group_name>/episode_000000.mp4",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Root directory of the dataset (e.g., datasets/my-dataset)')
    parser.add_argument('--threshold', type=float, default=0.002,
                       help='Variance threshold for detecting noticeable movement (default: 0.002)')
    parser.add_argument('--window_size', type=int, default=10,
                       help='Half-width of the window for variance calculation (default: 10)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: <dataset_dir>_processed)')
    args = parser.parse_args()

    main(args)
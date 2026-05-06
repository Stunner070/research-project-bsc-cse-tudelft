from pathlib import Path

root = Path("/scratch/sofyanali/celebvhq/videos")
print("Exists:", root.exists())
print("MP4 count:", len(list(root.rglob("*.mp4"))))

#output to txt file
with open("data_paths.txt", "w") as f:
    f.write(f"Exists: {root.exists()}\n")
    f.write(f"MP4 count: {len(list(root.rglob('*.mp4')))}\n")
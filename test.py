import os

# Path to the current image_paths.txt
paths_file = 'vector_data/image_paths.txt'

# Read the current paths
with open(paths_file, 'r') as f:
    image_paths = f.read().splitlines()

# Convert absolute paths to relative paths
# Assuming images are in the 'images/' directory
new_paths = []
for path in image_paths:
    # Extract the filename from the absolute path
    filename = os.path.basename(path)
    # Create the relative path
    relative_path = f"/images/{filename}"
    new_paths.append(relative_path)

# Write the updated paths back to the file
with open(paths_file, 'w') as f:
    f.write('\n'.join(new_paths))

print(f"Updated {paths_file} with relative paths.")
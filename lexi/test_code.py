from pathlib import Path

# Get the current working directory
cwd = Path.cwd()

# Get the location of this file
file_path = Path(__file__)

# Get the parent directory of this file
parent_dir = file_path.parent

# Print all the paths
print(f"Current working directory: {cwd}")
print(f"File path: {file_path}")
print(f"Parent directory: {parent_dir}")

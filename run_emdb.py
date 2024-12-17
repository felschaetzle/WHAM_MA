import subprocess


import subprocess

# Define the command as a list of strings
command = [
    "python", 
    "demo.py", 
    "--video", 
    "/mnt/hdd/emdb_dataset/P5/40_indoor_walk_big_circle/raw_short.mov",
    "--visualize"
]

# Run the command
result = subprocess.run(command)

# Print the output and error (if any)
print("Output:\n", result.stdout)
print("Error:\n", result.stderr)
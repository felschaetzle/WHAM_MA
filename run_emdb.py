import subprocess


import subprocess

# Define the command as a list of strings
command = [
    "python", 
    "demo.py", 
    "--subject",
    "P4",
    "--sequence",
    "35"
]

# Run the command
result = subprocess.run(command)

# Print the output and error (if any)
print("Output:\n", result.stdout)
print("Error:\n", result.stderr)
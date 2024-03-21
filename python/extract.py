import numpy as np

# Open the file
with open('openfoam_files/50_B/U', 'r') as file:
    lines = file.readlines()

# Find the index where the internalField starts
start_index = lines.index("internalField   nonuniform List<vector> \n")

# Read the number of vectors
num_vectors = int(lines[start_index + 1])

# Extract the lines containing vectors
vector_lines = lines[start_index + 2:start_index + 2 + num_vectors+1]

# Initialize a list to store valid vectors
valid_vectors = []

# Parse the vector lines and store valid vectors
for line in vector_lines:
    print(line)
    vector = list(map(float, line.strip("()\n").split()))
    if len(vector) == 3:  # Check if the vector has 3 elements
        valid_vectors.append(vector)

# Convert the list of valid vectors into a NumPy array
vectors = np.array(valid_vectors)

U = vectors[:,0]
U = np.reshape(U, (int(np.sqrt(len(U))),int(np.sqrt(len(U)))))

print(U.shape)

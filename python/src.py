import numpy as np

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


def extract(case = 'A'):
    # Open the file
    with open('your_file_path.txt', 'r') as file:
        lines = file.readlines()

    # Find the index where the internalField starts
    start_index = lines.index("internalField   nonuniform List<vector> \n")

    # Read the number of vectors
    num_vectors = int(lines[start_index + 1])

    # Extract the lines containing vectors
    vector_lines = lines[start_index + 2:start_index + 2 + num_vectors]

    # Initialize a list to store valid vectors
    valid_vectors = []

    # Parse the vector lines and store valid vectors
    for line in vector_lines:
        vector = list(map(float, line.strip("()\n").split()))
        if len(vector) == 3:  # Check if the vector has 3 elements
            valid_vectors.append(vector)

    # Convert the list of valid vectors into a NumPy array
    vectors = np.array(valid_vectors)

    # Print the extracted vectors
    print(vectors)
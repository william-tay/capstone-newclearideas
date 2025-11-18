# remove_attributes.py
# Removes all columns after source and target from an edge list

input_file = "../tech-as-topology/tech-as-topology-visual.edges"
output_file = "output_unweighted.edges"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # Skip malformed lines
        if len(parts) < 2:
            continue

        source, target = parts[0], parts[1]
        outfile.write(f"{source} {target}\n")

print(f"✅ Attributes removed. Clean file saved as '{output_file}'.")

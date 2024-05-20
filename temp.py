import os

def process_line(line):
    # Split the line by whitespace
    parts = line.split()
    if parts:
        try:
            # Extract the first number
            first_number = float(parts[0])
            # Modify the number (e.g., add 10)
            modified_number = first_number + 10
            # Replace the original number with the modified one
            parts[0] = str(modified_number)
            # Join the modified parts back into a line
            modified_line = ' '.join(parts)
            return modified_line
        except ValueError:
            # If the first part is not a valid number, return the original line
            return line
    else:
        # If the line is empty, return it unchanged
        return line

def edit_numbers_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            input_file = os.path.join(root, filename)
            output_file = os.path.join(root, f"modified_{filename}")
            try:
                with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                    for line in infile:
                        modified_line = process_line(line)
                        outfile.write(modified_line + '\n')
                print(f"Modified content written to {output_file}")
            except FileNotFoundError:
                print(f"File '{input_file}' not found.")

# Example usage:
folder_path = 'trainingsets/LoL features.v4-clean-ds.yolov8'  # Replace with your folder path
edit_numbers_in_folder(folder_path)
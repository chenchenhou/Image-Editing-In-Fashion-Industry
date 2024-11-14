import os
import json

INPUT_FILE = "editing_data_new.json"
OUTPUT_FILE = "editing_data_turbo.json"

# Read input data from input.json
with open(INPUT_FILE, "r") as infile:
    input_data = json.load(infile)

# Convert input data to desired format
output_data = {}
for item in input_data:
    for image_path, details in item.items():
        # Extract image name without directory and extension
        image_name = (
            os.path.basename(image_path).split(".")[0] + ".jpg"
        )  # Assuming output file extension as .jpg
        output_data[image_name] = {
            "src_prompt": details["src_prompt"].strip(),
            "tgt_prompt": [details["tgt_prompt"].strip()],
        }

# Write the converted data to output.json
with open(OUTPUT_FILE, "w") as outfile:
    json.dump(output_data, outfile, indent=4)

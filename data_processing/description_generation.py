import os
from openai import OpenAI
openai = OpenAI()

# Directory containing class folders
data_dir = "./"
output_dir = "./descriptions/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to generate prompt descriptions
def generate_descriptions(class_name, num_variants=50):
    prompt = f"Generate {num_variants} unique and descriptive prompts for the label '{class_name}'."
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
        max_tokens=150 * num_variants,
        n=1,
        stop=None
    )
    descriptions = response.choices[0].message.content.strip().split("\n")
    return descriptions[:num_variants]

# Iterate through each folder in the data directory
for class_name in os.listdir(data_dir):
    if class_name == "descriptions":
        continue
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        print(f"Generating descriptions for class: {class_name}")
        descriptions = generate_descriptions(class_name)
        
        # Save descriptions to a text file
        output_file = os.path.join(output_dir, f"{class_name}.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(descriptions))

print("Descriptions generated and saved successfully.")
import os
from rembg import remove
from tqdm import tqdm

input_folder = 'IMG'
output_folder = 'IMGBG'

# Folders to exclude
excluded_folders = {'A', 'blank'}

os.makedirs(output_folder, exist_ok=True)

print("📁 Scanning and processing images recursively (excluding A and blank)...")

for root, dirs, files in os.walk(input_folder):
    # Skip processing if current folder is in excluded list
    folder_name = os.path.basename(root)
    if folder_name in excluded_folders:
        continue

    for filename in tqdm(files, desc=f"Processing in {folder_name}"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(root, filename)

            # Create matching subfolder in output
            relative_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)

            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_subfolder, output_filename)

            try:
                with open(input_path, 'rb') as i:
                    input_data = i.read()

                output_data = remove(input_data)

                with open(output_path, 'wb') as o:
                    o.write(output_data)

            except Exception as e:
                print(f"❌ Error processing {input_path}: {e}")

print("\n✅ Done! Backgrounds removed (excluding A and blank folders). Output in:", output_folder)

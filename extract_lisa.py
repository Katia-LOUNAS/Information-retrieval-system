import shutil
import os

files_number = [0 , 5]
# Combine all document parts into a single file
combined_file_path = './Collection/combined_documents.txt'
with open(combined_file_path, 'wb') as combined_file:
    for part_number in range(6):
        for file_number in files_number:
            part_filename = f'./Collection/lisa/lisa{part_number}.{file_number}01'
            part_path = os.path.join(part_filename)
            with open(part_path, 'rb') as part_file:
                shutil.copyfileobj(part_file, combined_file)

    # Add the last two files directly
    for file_number in [627, 850]:
        part_filename = f'./Collection/lisa/lisa5.{file_number}'
        part_path = os.path.join(part_filename)
        with open(part_path, 'rb') as part_file:
            shutil.copyfileobj(part_file, combined_file)

print(f"All available document parts have been successfully combined into '{combined_file_path}'.")
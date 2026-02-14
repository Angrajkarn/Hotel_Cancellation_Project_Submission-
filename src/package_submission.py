import zipfile
import os
import sys

def package_project():
    zip_filename = 'Hotel_Cancellation_Project_Submission.zip'
    base_dir = r"c:\Users\Nivedita\.gemini\antigravity\playground\temporal-aurora\hotel_cancellations"
    output_path = os.path.join(base_dir, zip_filename)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                # Skip the zip file itself and hidden files
                if file == zip_filename or file.startswith('.'):
                    continue
                
                # Create relative path for archive
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)
                
                # Add to zip
                zipf.write(file_path, arcname=rel_path)
                    
    print(f"Project packaged into {output_path}")

if __name__ == "__main__":
    package_project()

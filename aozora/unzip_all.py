import os
import zipfile
import sys

def list_zip_files(directory):
    """指定したディレクトリ内の.zipファイルをリストアップする"""
    zip_files = []
    for fname in os.listdir(directory):
        if fname.endswith('.zip'):
            zip_files.append(fname)
    return zip_files

def main():
    if len(sys.argv) < 2:
        print("Usage: python unzip_all.py <directory>")
        sys.exit(1)
        
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"The directory '{directory}' does not exist.")
        sys.exit(1)
    
    zip_files = list_zip_files(directory)

    for zip_file in zip_files:
        zip_path = os.path.join(directory, zip_file)
        try:
            print(f'Extracting {zip_path.encode("utf-8", "surrogateescape")}...')

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extract_dir = os.path.splitext(zip_path)[0]
                zip_ref.extractall(extract_dir)
                print(f'Extracted {zip_path.encode("utf-8", "surrogateescape")} to {extract_dir}')
        except Exception as e:
            print(f'Error extracting {zip_path.encode("utf-8", "surrogateescape")}: {e}')

if __name__ == "__main__":
    main()
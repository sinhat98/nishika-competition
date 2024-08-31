import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python format_dirname.py <directory>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"The directory '{directory}' does not exist.")
        sys.exit(1)
    
    for path in directory.iterdir():
        if path.is_dir():
            # path内のファイルで「収集率」を含むファイル名をリストアップ
            files = list(path.glob('*収集率*'))
            file = files[0] if files else None
            if file:
                new_dirname = file.stem.split('_')[-1]
                new_path = path.with_name(new_dirname)
                path.rename(new_path)
                print(f'Renamed to {new_path}')
                
if __name__ == "__main__":
    main()
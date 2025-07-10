"""Testing interfaces and data models."""
from pathlib import Path
from nwm_explorer.data.routelink import download_routelinks, get_routelink_readers

def main():
    # Set root data directory
    root = Path("./data-new")

    # Download routelinks
    download_routelinks(root)

    # Look at data
    for d, r in get_routelink_readers(root).items():
        print(d)
        print(r.head().collect())

if __name__ == "__main__":
    main()

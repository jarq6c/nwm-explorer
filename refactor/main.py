"""Download and process RouteLink information."""
from modules.routelink import download_routelink

def main():
    routelink = download_routelink().collect()
    print(routelink)

if __name__ == "__main__":
    main()

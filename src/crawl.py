import argparse
import requests
import xml.etree.ElementTree as ET

def get_sitemap_links(sitemap_url, filter_path=""):
    """Parse `sitemap.xml` and return filtered page links"""
    try:
        response = requests.get(sitemap_url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch sitemap: {sitemap_url}")
            return []

        sitemap_xml = response.text
        root = ET.fromstring(sitemap_xml)
        namespaces = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Extract all links
        urls = [elem.text for elem in root.findall(".//ns:loc", namespaces)]
        
        # Apply filter if specified
        filtered_urls = [url for url in urls if filter_path in url] if filter_path else urls
        
        return filtered_urls
    except Exception as e:
        print(f"Error parsing sitemap {sitemap_url}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Scrape sitemap links from a website")
    parser.add_argument("--url", type=str, required=True, help="Sitemap URL to crawl")
    parser.add_argument("--filter", type=str, default="", help="Filter URLs containing this substring")

    args = parser.parse_args()
    
    # Get sitemap links
    filtered_pages = get_sitemap_links(args.url, args.filter)

    # Print results
    print(f"\n🔗 Found {len(filtered_pages)} Pages in Sitemap:")
    for url in sorted(filtered_pages):
        print(url)

if __name__ == "__main__":
    main()
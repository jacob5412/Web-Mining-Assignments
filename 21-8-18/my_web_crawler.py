import requests
import re
from bs4 import BeautifulSoup

MAX_COUNT = 10


def crawl_web(initial_url, search_for):
    crawled = set()
    to_crawl = [("<given>", initial_url)]
    counter = 0

    while to_crawl and counter < MAX_COUNT:
        counter += 1

        # Avoid repeating
        source, current_url = to_crawl.pop(0)
        if current_url in crawled:
            print("Already visited:", current_url)
            continue
        else:
            # If we see it once, don't bother again.
            crawled.add(current_url)

        # Connect and prepare to fetch URL
        try:
            print("Fetching:", current_url)
            r = requests.get(current_url)
        except Exception as e:
            print("   Skipping (no-fetch):", current_url)
            print(e)
            continue

        # Check it's an HTML page
        content_type = r.headers.get("Content-Type", "Nope")
        if not content_type.startswith("text/html"):
            print("   Skipping (non-html):", content_type)
            continue

        # Actually fetch the content
        text = r.content

        # Search for the phrase
        if search_for in text.decode("utf-8"):
            print("*** Found on", current_url)

        # Extract links and add them to the to_crawl list.
        soup = BeautifulSoup(text, features="lxml")
        links = soup.findAll("a", attrs={"href": re.compile("^https?://")})
        for link in sorted(links, key=lambda x: x.get("href")):
            if len(to_crawl) + counter >= MAX_COUNT:
                break
            to_crawl.append((current_url, link.get("href")))

    return crawled


if __name__ == "__main__":
    print(crawl_web("https://github.com", "blog"))

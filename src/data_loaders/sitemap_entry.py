from typing import NamedTuple, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from typing import List

# Define the NamedTuple with lastmod as an Optional[datetime]
class SitemapEntry(NamedTuple):
    url: str
    lastmod: Optional[datetime]  # Use datetime for timestamps


class Sitemap:
    def __init__(self, sitemap: str) -> None:
        self.sitemap = sitemap

    def load(self) -> List[SitemapEntry]:
        # Fetch the sitemap
        response = requests.get(self.sitemap)
        sitemap_content = response.content

        # Parse the sitemap XML
        soup = BeautifulSoup(sitemap_content, "xml")

        # Extract URLs and lastmod
        sitemap_entries = []
        for url_tag in soup.find_all("url"):
            loc = url_tag.find("loc").text
            lastmod_tag = url_tag.find("lastmod")
            # Parse lastmod with timestamp format
            lastmod = (
                datetime.strptime(lastmod_tag.text, "%Y-%m-%dT%H:%M:%S.%fZ")
                if lastmod_tag and lastmod_tag.text
                else None
            )
            sitemap_entries.append(SitemapEntry(url=loc, lastmod=lastmod))

        return sitemap_entries

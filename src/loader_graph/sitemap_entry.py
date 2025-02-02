from typing import NamedTuple, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from typing import List



class SitemapEntry(NamedTuple):
    """
    Represents a single entry in a sitemap.

    :param url: The URL of the sitemap entry.
    :param lastmod: The optional datetime of the last modification of the entry.
    :param score: An optional score associated with the entry. Defaults to 0.
    """
    url: str
    lastmod: Optional[datetime]
    score: float = 0


class Sitemap:
    """
    A class to fetch and parse XML sitemaps, extracting a list of sitemap entries.
    """
    def __init__(self, sitemap: str) -> None:
        """
        Initialize the Sitemap instance with a given sitemap URL.

        :param sitemap: The URL of the sitemap to fetch and parse.
        """
        self.sitemap = sitemap


    def load(self) -> List[SitemapEntry]:
        """
        Load and parse the sitemap to extract entries.
        Fetches the sitemap content from the provided URL, parses the XML to extract
        all URL entries along with their optional lastmod timestamps, and returns them
        as a list of SitemapEntry objects.

        :return: A list of parsed sitemap entries.
        """
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

from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
import requests
import time
from src.data_loaders.sitemap_entry import SitemapEntry


class DoclingHTMLLoader(BaseLoader):
    def __init__(self, sitemap_entry: SitemapEntry | list[SitemapEntry],
                 max_retries: int = 3, retry_delay: float = 1.0) -> None:
        self._file_paths = [se.url for se in sitemap_entry] if isinstance(sitemap_entry, list) \
            else [sitemap_entry.url]

        self._last_dates = [se.lastmod for se in sitemap_entry] if isinstance(sitemap_entry, list) \
            else [sitemap_entry.lastmod]

        self._converter = DocumentConverter(allowed_formats=[InputFormat.HTML])
        self._max_retries = max_retries
        self._retry_delay = retry_delay


    def lazy_load(self) -> Iterator[LCDocument]:
        for source, lastmod in zip(self._file_paths, self._last_dates):
            if self._is_valid_url(source):
                attempt = 0
                while attempt < self._max_retries:
                    try:
                        dl_doc = self._converter.convert(source).document
                        text = dl_doc.export_to_markdown()
                        yield LCDocument(page_content=text, metadata={"source": source, "lastmod": lastmod})
                        break
                    except Exception as e:
                        attempt += 1
                        if attempt < self._max_retries:
                            print(
                                f"Error processing {source}: {e}. "
                                f"Retrying in {self._retry_delay} seconds... (Attempt {attempt}/{self._max_retries})")
                            time.sleep(self._retry_delay)
                        else:
                            print(f"Failed to process {source} after {self._max_retries} attempts: {e}")
            else:
                print(f"Skipping invalid or inaccessible URL: {source}")


    @staticmethod
    def _is_valid_url(url: str) -> bool:
        try:
            response = requests.head(url, allow_redirects=True)
            # print(f"URL: {url}, STATUS: {response.status_code}")
            return response.status_code == 200
        except requests.RequestException as e:
            print(f"Error checking URL {url}: {e}")
            return False
from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
import requests
from src.data_loaders.sitemap_entry import SitemapEntry


class DoclingHTMLLoader(BaseLoader):
    def __init__(self, sitemap_entry: SitemapEntry | list[SitemapEntry]) -> None:
        self._file_paths = [se.url for se in sitemap_entry] if isinstance(sitemap_entry, list) \
            else [sitemap_entry.url]

        self._last_dates = [se.lastmod for se in sitemap_entry] if isinstance(sitemap_entry, list) \
            else [sitemap_entry.lastmod]

        self._converter = DocumentConverter(allowed_formats=[InputFormat.HTML])


    def lazy_load(self) -> Iterator[LCDocument]:
        for source, lastmod in zip(self._file_paths, self._last_dates):
            if self._is_valid_url(source):
                try:
                    dl_doc = self._converter.convert(source).document
                    text = dl_doc.export_to_markdown()
                    yield LCDocument(page_content=text, metadata={"source": source, "lastmod": lastmod})
                except Exception as e:
                    print(f"Error processing {source}: {e}")
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
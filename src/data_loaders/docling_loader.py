from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
import requests


class DoclingHTMLLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            if self._is_valid_url(source):
                try:
                    dl_doc = self._converter.convert(source).document
                    text = dl_doc.export_to_markdown()
                    yield LCDocument(page_content=text, metadata={"source": source})
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
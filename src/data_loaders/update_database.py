import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_loaders.document_processor import DocumentProcessor
from dotenv import load_dotenv
from src.data_loaders.sitemap_entry import Sitemap


if __name__ == '__main__':
    load_dotenv()

    sitemap = Sitemap(sitemap="https://tech.appunite.com/blog/blog-sitemap.xml")
    sitemap_entries = sitemap.load()

    document_processor = DocumentProcessor('au-blog-rag')
    document_processor.update_database(sitemap_entries)
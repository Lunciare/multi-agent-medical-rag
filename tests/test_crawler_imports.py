import sys
import logging
logging.basicConfig(level=logging.DEBUG)

print("Starting import tests")
try:
    import fitz
    print("fitz OK")
    import requests
    print("requests OK")
    import bs4
    print("bs4 OK")
    from playwright.async_api import async_playwright
    print("playwright OK")
    
    from scripts.data_processing.rag_crawler import TARGET_BOOKS
    print("TARGET_BOOKS loaded")
except Exception as e:
    print(f"Error: {e}")

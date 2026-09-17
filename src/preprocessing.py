"""
Text Preprocessing Module for TRUTHSCAN AI.
Handles missing values, wire-service watermark removal, boilerplate filtering,
and text normalization.

Ensures identical preprocessing between training and real-time inference.
"""

import re
import html


def clean_text(text: str) -> str:
    """
    Cleans raw text conservatively without destroying grammatical structure,
    named entities, or semantic meaning.
    
    Operations:
    1. Converts None/NaN to empty string.
    2. Unescapes HTML entities (&amp; -> &, &quot; -> ", etc.).
    3. Strips URL patterns (http, https, www, pic.twitter.com).
    4. Strips embedded multimedia tags (e.g., [youtube https://...]).
    5. Removes news agency datelines (e.g., 'WASHINGTON (Reuters) -', '(Reuters) -')
       to prevent models from overfitting on publisher watermarks rather than content.
    6. Removes social media attribution footers (e.g., 'Featured image via...').
    7. Normalizes all whitespace characters (newlines, tabs, multiple spaces) to single spaces.
    8. Strips leading and trailing whitespace.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities
    text = html.unescape(text)
    
    # 2. Remove URLs, pic.twitter links, and youtube embeds
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'pic\.twitter\.com/\S+', '', text)
    text = re.sub(r'\[youtube\s+[^\]]+\]', '', text, flags=re.IGNORECASE)
    
    # 3. Strip leading news agency dateline artifacts (e.g., "CITY (Reuters) - ", "(Reuters) - ")
    # This prevents shortcut learning where the model memorizes publisher branding
    text = re.sub(r'^[A-Z\s,./-]*\s*\([Rr]euters\)\s*[-—–:]\s*', '', text)
    text = re.sub(r'^\s*\(?\s*[Rr]euters\s*\)?\s*[-—–:]\s*', '', text)
    text = re.sub(r'\([Rr]euters\)', '', text)
    text = re.sub(r'\bReuters\b', '', text, flags=re.IGNORECASE)
    
    # 4. Remove publisher footer boilerplates common in online scraping
    text = re.sub(r'Featured\s+image\s+.*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Photo\s+by\s+.*?\)', '', text, flags=re.IGNORECASE)
    
    # 5. Replace non-breaking spaces and special whitespace
    text = text.replace('\xa0', ' ')
    
    # 6. Normalize multiple whitespace characters, tabs, and newlines into single spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def combine_title_and_text(title: str, text: str) -> str:
    """
    Combines headline and article text into a single cohesive string for model input.
    Cleans both components identically before combining.
    """
    clean_title = clean_text(title)
    clean_body = clean_text(text)
    
    if clean_title and clean_body:
        return f"{clean_title} {clean_body}"
    elif clean_title:
        return clean_title
    elif clean_body:
        return clean_body
    else:
        return ""

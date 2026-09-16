"""
Text Preprocessing Module for TRUTHSCAN AI.
Handles missing values, whitespace normalization, and non-aggressive text cleaning.
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
    3. Normalizes all whitespace characters (newlines, tabs, multiple spaces) to single spaces.
    4. Strips leading and trailing whitespace.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove URL patterns while preserving text context
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Replace non-breaking spaces and other unicode spaces
    text = text.replace('\xa0', ' ')
    
    # Normalize multiple whitespace characters, tabs, and newlines into single spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def combine_title_and_text(title: str, text: str) -> str:
    """
    Combines headline and article text into a single cohesive string for model input.
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

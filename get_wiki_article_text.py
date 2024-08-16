
import pandas as pd
from tqdm import tqdm
import requests
import time
import re
import argparse


def create_arg_parser() -> argparse.Namespace:
    """
    Creates and returns argument parser for command-line arguments.

    Returns:
        Namespace containing the command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-in", "--input_file", type=str, help="Input file", required=True
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Name of output file",
        default="output.jsonl",
    )
    args = parser.parse_args()
    return args

def strip_archive_suffix(title):
    """
    Remove any archive suffix from a Wikipedia page title.

    Args:
        title (str): Wikipedia page title.

    Returns:
        str: The cleaned title without the archive suffix.
    """
    pattern = r'\/Archive \d+$'
    return re.sub(pattern, '', title)

def get_wikipedia_article_content(page_title):
    """
    Fetch the main content of a Wikipedia page given its title from the Wikipedia api.

    Args:
        page_title (str): The title of the Wikipedia page to fetch.

    Returns:
        str: The text content of the Wikipedia page.
    
    Raises:
        Exception: If there is an error fetching the content from Wikipedia.
    """
    page_title = strip_archive_suffix(page_title)
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": page_title
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching data from Wikipedia API: {response.status_code}")

    data = response.json()
    pages = data.get('query', {}).get('pages', {})
    page = next(iter(pages.values()), {})

    content = page.get('extract', '')
    return content


def get_wiki_content(page_titles):
    """
    Retrieve the Wikipedia content for a list of page titles.

    Args:
        page_titles (list): A list of page titles to process.

    Returns:
        list: A list of texts extracted from the Wikipedia pages.
    """
    article_contents = []
    for page_title in tqdm(page_titles, desc='Processing page titles'):
        try:
            article_content = get_wikipedia_article_content(page_title)
            article_contents.append(article_content)
        except Exception as e:
            article_contents.append("")
            print(f"Error processing page title {page_title}: {e}")
    return article_contents

if __name__ == "__main__":
    args = create_arg_parser()

    # Load dataset
    input_data = pd.read_json(args.input_file, lines=True)
    
    # Select only the 'PAGE-TITLE' and 'PAGE-ID' columns
    page_data_df = input_data[['PAGE-TITLE', 'PAGE-ID']]

    # Remove duplicate rows based on 'PAGE-TITLE' and 'PAGE-ID'
    unique_page_data_df = page_data_df.drop_duplicates()

    # Process all page titles at once
    article_contents = get_wiki_content(unique_page_data_df['PAGE-TITLE'])

    # Add the article content to the DataFrame
    unique_page_data_df['ARTICLE-TEXT'] = article_contents

    # Save as new dataset
    unique_page_data_df.to_json(args.output_file, orient='records', lines=True)
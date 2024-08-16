import argparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset, load_dataset
import pandas as pd


def create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-in", "--input_file", type=str, help="Input file", required=True
    )
    parser.add_argument(
        "-a", "--article_file", type=str, help="Name of article file", required=True
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Name of output file",
        default="output.jsonl",
    )
    parser.add_argument(
        "-n",
        "--top_n_percent",
        type=int,
        help="Top percentage of content words that will be saved",
        default=5,
    )
    parser.add_argument(
        "-min",
        "--min_n",
        type=int,
        help="Minimum number of content words that will be saved",
        default=30,
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=bool,
        help="If true a new column TEXT-MASKED will be added to the comments where the content words are masked.",
        default=True,
    )
    args = parser.parse_args()
    return args


def preprocess_text(text):
    """
    Processes the article text by converting to lower case and removing non-alphabetic characters.
    
    Args:
        text: The original text of the article.
    
    Returns:
        The cleaned and processed text.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def extract_top_tfidf_words(example, vectorizer, feature_names, top_n_percent, min_n):
    """
    Identifies and extracts the top TF-IDF scoring words from an article's text.
    
    Args:
        example: A dictionary containing the article text under the key 'ARTICLE-TEXT'.
        vectorizer: A fitted TfidfVectorizer object.
        feature_names: List of feature names extracted by the vectorizer.
        top_n_percent: The percentage of total words in the article to consider as top words.
        min_n: The minimum number of words to extract.
    
    Returns:
        The example dictionary updated with a new key 'CONTENT-WORDS' containing a list of top words.
    """
    text = example["ARTICLE-TEXT"]
    if text.strip():
        row = vectorizer.transform([text]).toarray().flatten()
        unique_words = set(text.split())
        calculated_top_n = int(len(unique_words) * (top_n_percent / 100))
        top_n = max(min_n, min(len(unique_words), calculated_top_n))
        top_indices = row.argsort()[-top_n:][::-1]
        top_words = [feature_names[index] for index in top_indices]
        example["CONTENT-WORDS"] = top_words
    else:
        example["CONTENT-WORDS"] = []
    return example


def mask_content_words(text, content_words):
    """
    Masks content words in the provided text.
    
    Args:
        text: Original text from comments.
        content_words: List of words to be masked.
    
    Returns:
        Text with content words replaced by '[MASK]'.
    """
    words = text.split()

    # Iterate through each word and check if it contains any content word as a substring
    masked_words = [
        (
            "[MASK]"
            if any(
                re.search(re.escape(content_word), word, re.IGNORECASE)
                for content_word in content_words
            )
            else word
        )
        for word in words
    ]

    # Join the words back into a single string
    masked_text = " ".join(masked_words)
    return masked_text


def process_comments(example):
    """
    Applies content word masking to all comments in a discussion.
    
    Args:
        example: A dictionary with a 'COMMENTS' key containing a list of comment dictionaries.
    
    Returns:
        A dictionary with updated 'COMMENTS' where content words have been masked.
    """
    comments = example["COMMENTS"]
    content_words = example["CONTENT-WORDS"]

    for comment in comments:
        text_clean = comment.get("TEXT-CLEAN", "")
        comment["TEXT-MASKED"] = mask_content_words(text_clean, content_words)

    return {"COMMENTS": comments}


if __name__ == "__main__":
    args = create_arg_parser()

    # Load dataset
    dataset = pd.read_json(args.input_file, lines=True)

    # Load in article texts
    articles = load_dataset("json", data_files=args.article_file, split="train")

    # Preprocess the text
    articles = articles.map(
        lambda x: {"ARTICLE-TEXT": preprocess_text(x["ARTICLE-TEXT"])}
    )

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    preprocessed_texts = articles["ARTICLE-TEXT"]
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Extract top TF-IDF words for each article
    articles = articles.map(
        lambda x: extract_top_tfidf_words(
            x,
            vectorizer,
            feature_names,
            top_n_percent=args.top_n_percent,
            min_n=args.min_n,
        ),
        batched=False,
    )

    # Remove rows with an empty list of content words
    articles = articles.filter(lambda x: len(x["CONTENT-WORDS"]) > 0)

    # Convert the Hugging Face dataset back to a pandas DataFrame
    articles_df = articles.to_pandas()

    # Merge the CONTENT-WORDS column from articles onto the dataset using PAGE-TITLE as the key
    merged_data = pd.merge(
        dataset,
        articles_df[["PAGE-TITLE", "CONTENT-WORDS"]],
        on="PAGE-TITLE",
        how="inner",
    )

    # Process comments in the merged data
    merged_data = Dataset.from_pandas(merged_data)
    if args.mask:
        merged_data = merged_data.map(process_comments)

    # Save the merged data as a JSON file
    merged_data.to_json(args.output_file, orient="records", lines=True)

import argparse
from datasets import load_dataset
import re

def create_arg_parser() -> argparse.Namespace:
    """
    Creates and returns argument parser for command-line arguments.

    Returns:
        Namespace containing the command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-in",
        "--input_file",
        type=str,
        help="Name of input file.",
        required=True,
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Name of output file, without extension.",
        required=True,
    )
    parser.add_argument(
        "-ds",
        "--dev_size",
        type=int,
        help="Number of samples to save as separate development set.",
        default=200,
    )
    parser.add_argument(
        "-ts",
        "--test_size",
        type=int,
        help="Number of samples to save as separate test set.",
        default=100,
    )
    parser.add_argument(
        "-inter",
        "--interaction",
        type=bool,
        help="If set to true, only interactive discussions will be returned",
        default=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Number to use as seed for the shuffling process when selecting dev and/or test data",
        default=42,
    )
    args = parser.parse_args()
    return args


def clean_text(example):
    """
    Cleans the text of comments by removing unwanted patterns and signatures.

    Args:
        example (dict): A dictionary representing a single example from the dataset, containing a 'COMMENTS' field.

    Returns:
        dict: The modified example with cleaned text in the 'COMMENTS' field.
    """
    # Regular expressions to find patterns including an IP address or a username between '--' and '(talk)'
    # and the specific 'Preceding unsigned comment added by' pattern with IP and '(talk)'
    pattern1 = r'--\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|[\w\s-]+?)\s*\(talk\)$'
    pattern2 = r'Preceding unsigned comment added by (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|[\w\s-]+?)\s*$'
    pattern3 = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s*\(talk\)$'

    for i, comment in enumerate(example['COMMENTS']):
        text = str(comment['TEXT-CLEAN'])

        # Try the first pattern
        match = re.search(pattern1, text)
        if not match:
            # If no match, try the second pattern
            match = re.search(pattern2, text, flags=re.IGNORECASE)
        if not match:
            # If no match, try the second pattern
            match = re.search(pattern3, text)

        if match:
            # Extract the IP address or username from the matched group
            username = match.group(1).strip() 
            # If the USER field is empty, fill it with the extracted username
            if not example['COMMENTS'][i]['USER']:
                example['COMMENTS'][i]['USER'] = username

        # Remove the entire pattern 1 and 3 from the text and replace newlines with a space
        text = re.sub(pattern1, '', text)
        text = re.sub(pattern3, '', text)

        # Remove the 'unsigned' signature
        text = re.sub(r'—\s?(the\s+)?Preceding.*$', '', text, flags=re.IGNORECASE)

        # Remove any trailing dashes with optional surrounding whitespace
        text = re.sub(r'\s*--\s*$', '', text)

        # Replace newlines with spaces and reduce multiple spaces to a single space
        text = re.sub('\s+', ' ', text).strip()

        # Update the cleaned text back into the dataset
        example['COMMENTS'][i]['TEXT-CLEAN'] = text

    return example

def filter_discussions(example):
    """
    Filters out discussions that don't meet specific criteria.

    Args:
        example (dict): A dictionary representing a single example from the dataset, containing a 'COMMENTS' field.

    Returns:
        bool: True if the discussion meets the criteria, False otherwise.
    """
    # Extract the text from the first comment's 'TEXT-CLEAN' field
    text = str(example['COMMENTS'][0]['TEXT-CLEAN']).lower()
    first_user = example['COMMENTS'][0]['USER']

    # Check if first user is not null/None
    if not first_user:
        return False

    # Check if the text starts with 'Hello fellow Wikipedians'
    if text.startswith('hello fellow wikipedians'):
        return False

    # Check if the text contains both 'fair use' and 'image'
    if 'fair use' in text and 'image' in text:
        return False

    # Check if the text has less than 10 words
    if len(text.split()) < 10:
        return False

    # Check if the text starts with 'This article was automatically assessed'
    if text.startswith('this article was automatically assessed'):
        return False

    # Check if discussion is merge request
    if 'merge' in text:
        return False

    # Check if first comment is not a first level comment
    if text.startswith(':'):
        return False
    
    # Check for bot messages (as first comment)
    if 'automated bot run' in text:
        return False

    # If none of the conditions are met, return True
    return True


def find_interaction(example):
    """
        Determines if there is interaction in a discussion, defined by user 1 commenting after another user.

        Args:
            example (dict): A dictionary representing a single example from the dataset, containing a 'COMMENTS' field.

        Returns:
            bool: True if interaction is found, False otherwise.
    """
    users = [comment['USER'] for comment in example['COMMENTS']]
    if len(set(users)) > 1:
        user_two = None
        for i, user in enumerate(users):
            if i > 0:
                if user != users[0]:
                    user_two = i
                    break
        if user_two:
            if users[0] in users[user_two:]:
                return True
    return False


def remove_none_users(example):
    """
    Filters out discussions that contain comments with None as the USER field.

    Args:
        example (dict): A dictionary representing a single example from the dataset, containing a 'COMMENTS' field.

    Returns:
        bool: True if no None users are found, False otherwise.
    """
    users = set([comment['USER'] for comment in example['COMMENTS']])
    if None in users:
        return False
    return True

if __name__ == "__main__":
    args = create_arg_parser()
    dev_size = int(args.dev_size)
    test_size = int(args.test_size)
    seed = int(args.seed)

    data = load_dataset('json', data_files = args.input_file, split='train')
    data = data.map(clean_text)
    data = data.filter(filter_discussions)
    if args.interaction:
        data = data.filter(find_interaction)
        data = data.filter(remove_none_users)
    print(f'Final number of discussions: {len(data)}')

    if dev_size > 0 or test_size > 0:
        shuffled_data = data.shuffle(seed=seed)

    if dev_size > 0:
        dev_data = shuffled_data.select(range(dev_size))
        dev_data.to_json(args.output_file + '_dev.jsonl', orient='records', lines=True)

    if test_size > 0:
        test_data = shuffled_data.select(range(dev_size, dev_size + test_size))
        test_data.to_json(args.output_file + '_test.jsonl', orient='records', lines=True)

    data.to_json(args.output_file + '.jsonl', orient='records', lines=True)
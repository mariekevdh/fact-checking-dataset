import argparse
from datasets import load_dataset, Dataset
import pandas as pd

def create_arg_parser() -> argparse.Namespace:
    """
    Creates and returns argument parser for command-line arguments.

    Returns:
        Namespace containing the command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--result_files",
        nargs='+',
        type=str,
        help="Result files. Should contain at least the column 'output' and 'PRED-LABEL. Multiple files should be separated by a whitespace.",
    )
    parser.add_argument(
        "-m",
        "--main_data",
        type=str,
        help="Name of main data file, containing the columns PAGE-ID, PAGE-TITLE, DISCUSSION-TITE, DISCUSSION-ID, COMMENTS.",
        required=True,
    )
    parser.add_argument(
        "-out",
        "--output_path",
        type=str,
        help="Name of output file, without extension.",
        required=True,
    )
    args = parser.parse_args()
    return args

def get_summary(example):
    """
    Extracts the summary from the 'output' field of an example if available.
    
    Args:
        example (dict): A single record from a dataset containing an 'output' key.
    
    Returns:
        dict: The original example updated with a new key 'SUMMARY' that contains the extracted summary or an empty string if not found.
    """
    output_text = example['output']
    summary_end = -(len('Answer: yes'))
    summary_start = len('Summary: ')

    # Extract the summary text or return emtpy string if not found
    if output_text.lower().startswith('summary'):
        summary = output_text[summary_start:summary_end].strip()
    else:
        summary = ""

    example['SUMMARY'] =  summary
    return example


if __name__ == "__main__":
    args = create_arg_parser()
    result_files = args.result_files
    main_file = args.main_data
    output_path = args.output_path

    results = load_dataset('json', data_files=result_files, split='train')
    main_data = load_dataset('json', data_files=main_file, split='train')

    faulty_labels = results.filter(lambda example: 'answer:' not in example['output'].lower())
    print(f"Number of discussions that did not get a proper label: {len(faulty_labels)}") 

    factcheck_results = results.filter(lambda example: example['output'].lower().strip().endswith('answer: yes'))
    print(f'Total number of interaction discussions: {len(results)}')
    print(f'Number of fact-checking discussions: {len(factcheck_results)}')
    
    factcheck_results = factcheck_results.map(get_summary)
    summary_count_before = len(factcheck_results)
    factcheck_results = factcheck_results.filter(lambda example: example['SUMMARY'] != '')
    print(f'Number of discussions that do not contain a proper summary: {summary_count_before - len(factcheck_results)}')

    # Delete prompt and output
    factcheck_results = factcheck_results.remove_columns(['PROMPT', 'output', 'PRED-LABEL'])

    # Convert dataset to pandas dataframes for easier merging
    factcheck_results_df = factcheck_results.to_pandas()
    main_data_df = main_data.to_pandas()

    # Merge the DataFrames on DISCUSSION-ID
    merged_df = pd.merge(factcheck_results_df, main_data_df, on='DISCUSSION-ID', how='left')

    # Convert back to a Hugging Face dataset to preserve dictionary structure in COMMENTS column when saving the data
    merged_dataset = Dataset.from_pandas(merged_df)
    print(f'Final number of discussions in fact-checking dataset: {len(merged_dataset)}')

    merged_dataset.to_json(output_path, orient='records', lines=True)
    print(f'File saved as {output_path}')
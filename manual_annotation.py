from datasets import load_dataset, Dataset
import argparse
import os


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
        help="Input file.",
        required=True,
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        help="Output file.",
        required=True,
    )
    parser.add_argument(
        "-lc",
        "--label_column",
        type=str,
        help="Column name to save the annotations, or existing column with partial annotations.",
        default="LABEL",
    )
    args = parser.parse_args()
    return args


def clear_terminal():
    """
    Clears the terminal screen to provide a clean interface for user input.
    """
    os.system("cls" if os.name == "nt" else "clear")


def create_full_discussion(row):
    """
    Creates a string representation of a discussion including the Wikipedia page, discussion title, and the opening post.

    Args:
        row (dict): A dictionary containing the discussion data.
    
    Returns:
        str: Formatted string of the discussion.
    """
    return (
        "Wikipedia page: "
        + row["PAGE-TITLE"]
        + "\n\n"
        + "Discussion title: "
        + row["DISCUSSION-TITLE"]
        + "\n\nOpening post: "
        + row["COMMENTS"][0]["TEXT-CLEAN"]
    )


def annotate(row, label_column):
    """
    Annotates a single discussion row based on user input. The user is asked whether the discussion is focused on fact-checking.

    Args:
        row (dict): A dictionary representing a discussion with possible prior annotations.
        label_column (str): The column name where the annotation result will be stored.
    
    Returns:
        dict: The updated row with new annotations, or the command 'save' to indicate saving progress and stopping.
    """
    invalid = False
    if row[label_column].lower() not in ["yes", "no"]:
        full_discussion = create_full_discussion(row)
        while True:
            clear_terminal()
            print(f"{full_discussion}")
            print("\n------------------------------")
            if invalid:
                print("\nInvalid input. Please enter 'yes', 'no', or 'save'")
                invalid = False
            annotation = (
                input(
                    "\nIs the main focus of this discussion on fact-checking? Answer 'yes' or 'no' (or 'save' to save and stop): "
                )
                .strip()
                .lower()
            )
            if annotation in ["yes", "no", "save"]:
                break
            else:
                invalid = True
        if annotation == "save":
            return "save"
        row[label_column] = annotation
    return row


if __name__ == "__main__":
    args = create_arg_parser()
    label_column = args.label_column

    dataset = load_dataset("json", data_files=args.input_file, split="train")
    if label_column not in dataset.column_names:
        dataset = dataset.add_column(label_column, [""] * len(dataset))

    annotated_rows = []
    save = False
    for row in dataset:
        if not save:
            annotated_row = annotate(row, label_column)
            if annotated_row == "save":
                save = True
            else:
                annotated_rows.append(annotated_row)
        if save:
            annotated_rows.append(row)

    # Create dictionary from rows and create a new dataset with the annotations
    updated_dict = {
        key: [row[key] for row in annotated_rows] for key in annotated_rows[0].keys()
    }
    updated_dataset = Dataset.from_list(annotated_rows)

    updated_dataset.to_json(args.output_file, orient="records", lines=True)
    print(f"Annotations have been saved to {args.output_file}")

from datasets import load_dataset
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import re


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
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        help="Batch size",
        default=4,
    )
    parser.add_argument(
        "-s",
        "--selection",
        type=int,
        help="Chunk that will be processed. 1 means first 20000, 2 means second 20000, etc. If 0, all data will be processed.",
        default=0,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        help="Model name",
        default="microsoft/Phi-3-mini-4k-instruct",
    )
    parser.add_argument(
        "-p",
        "--prompt_type",
        type=str,
        help="Prompt type to use for generating labels",
        default="label",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=bool,
        help="If set to true, classification report will be printed. A column 'LABEL' is expected in the input data with the true labels",
        default=False,
    )
    args = parser.parse_args()
    return args


def normalize_text(text):
    """
    Normalize the text by removing URLs, replacing newlines with spaces, and reducing consecutive non-alphanumeric characters.

    Args:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    # Replace urls with [url]
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    cleaned_text = re.sub(url_pattern, "[url]", text)

    # Replace newlines with single whitespace
    normalized_text = re.sub("\n", " ", cleaned_text)

    # Replace consecutive occurrences of any characters that are not alphanumeric in the set with a single instance
    non_alphanumeric = r"([^a-zA-Z0-9])\1+"
    normalized_text = re.sub(non_alphanumeric, r"\1", normalized_text)

    return normalized_text


def add_phi_tag(prompt):
    """
    Add tags to the given prompt. The tags are necessary for Phi-3.

    Args:
        prompt (str): The original prompt.

    Returns:
        str: The modified prompt with a PHI tag appended.
    """
    return f"<|user|>{prompt}<|end|><|assistant|>"


def generate_prompt(example, prompt_type):
    """
    Generate a detailed prompt based on the discussion title and the first comment using various instruction templates.

    Args:
        example (dict): A dictionary containing 'DISCUSSION-TITLE' and 'COMMENTS' fields.
        prompt_type (str): The type of prompt to generate.

    Returns:
        dict: A dictionary containing the generated prompt.
    """
    title = normalize_text(str(example["DISCUSSION-TITLE"]))
    first_comment = normalize_text(example["COMMENTS"][0]["TEXT-CLEAN"])

    main_objective = "Objective:\nBased on the given title and opening post, determine whether the primary focus of the following Wikipedia talk page discussion is on fact-checking the current content of the article. Fact-checking refers to verifying the accuracy and truthfulness of the existing content in the Wikipedia article."
    discussion_excerpt = (
        f"Discussion Excerpt:\nTitle: {title}\nOpening post: {first_comment}"
    )

    def instruction_template(
        instruction_3=None, answer_instruction="basic", examples=True
    ):
        if answer_instruction == "basic":
            answer_instruction = "Answer 'Yes' or 'No'."
        instructions = ["Carefully read the provided discussion excerpt."]
        if examples:
            instructions.append(
                (
                    "Analyze whether the main theme of the discussion is primarily focused on fact-checking the current content of the article, using the following criteria:\n"
                    "    - Yes: If the focus is on assessing the verifiability or accuracy of any current information from the article, correcting misinformation, the validity or reliability of references, addressing the absence of references, providing references, or similar concepts.\n"
                    "    - No: If the focus is on the addition of new information, adding different viewpoints, requesting new information, style, formatting, content organization, relevance of content, or lacks enough context to make a classification.\n"
                    "Note: The examples provided above are not exhaustive. Use your judgment to determine if the discussion is primarily focused on fact-checking, even if the situation differs slightly from the examples."
                )
            )
        if instruction_3:
            instructions.append(instruction_3)
        instructions.append(answer_instruction)

        instruction_string = "Instructions:\n"
        for i, inst in enumerate(instructions):
            instruction_string += f"{i+1}. {inst}"
            if i < len(instructions)-1:
                instruction_string += "\n"

        return instruction_string

    def response_template(extra_response=None):
        if extra_response:
            return f"Response Template:\n{extra_response}\nAnswer: Yes/No"
        return "Response Template:\nAnswer: Yes/No"

    def prompt_template(
        instruction_3=None,
        extra_response=None,
        answer_instruction="basic",
        examples=True,
    ):
        return f"{main_objective}\n\n{instruction_template(instruction_3, answer_instruction, examples)}\n\n{response_template(extra_response)}\n\n{discussion_excerpt}"

    if prompt_type == "simple_label_only":
        prompt = prompt_template(examples=False)
    elif prompt_type == "examples_label_only":
        prompt = prompt_template()
    elif prompt_type == "explanation":
        instruction_3 = "Explain your classification."
        extra_response = "Explanation: [Briefly explain your answer.]"
        answer_instruction = "Based on your explanation, answer 'Yes' or 'No'."
        prompt = prompt_template(
            instruction_3=instruction_3,
            extra_response=extra_response,
            answer_instruction=answer_instruction,
        )
    elif prompt_type == "highlight":
        instruction_3 = (
            "Cite specific parts of the discussion that most influenced your decision."
        )
        extra_response = "Citation(s): [Cite specific parts of the discussion that most influenced your decision.]"
        answer_instruction = "Based on your citations, answer 'Yes' or 'No'."
        prompt = prompt_template(
            instruction_3=instruction_3,
            extra_response=extra_response,
            answer_instruction=answer_instruction,
        )
    elif prompt_type == "highlight_explain":
        instruction_3 = "Cite specific parts of the discussion that influenced your decision and explain their relevance to your classification."
        extra_response = "Explanation: [Briefly explain your classification, using citations from the discussion.]"
        prompt = prompt_template(
            instruction_3=instruction_3, extra_response=extra_response
        )
    elif prompt_type == "summarize":
        instruction_3 = "Summarize the main focus of this discussion."
        extra_response = "Summary: [Briefly summarize the discussion.]"
        answer_instruction = "Based on your summary, answer 'Yes' or 'No'."
        prompt = prompt_template(
            instruction_3=instruction_3,
            extra_response=extra_response,
            answer_instruction=answer_instruction,
        )
    elif prompt_type == "summarize_explain":
        instruction_3 = ["Summarize the main focus of this discussion.", "Based on your summary, briefly explain your answer."]
        extra_response = "Summary: [Briefly summarize the discussion.]\nExplanation: [Based on your summary, briefly explain your classification.]"
        prompt = prompt_template(
            instruction_3=instruction_3, extra_response=extra_response
        )
    elif prompt_type == "paraphrase":
        instruction_3 = "Paraphrase the main focus of this discussion."
        extra_response = "Paraphrase: [Briefly paraphrase the discussion.]"
        answer_instruction = "Based on your paraphrase, answer 'Yes' or 'No'."
        prompt = prompt_template(
            instruction_3=instruction_3,
            extra_response=extra_response,
            answer_instruction=answer_instruction,
        )
    elif prompt_type == "paraphrase_explain":
        instruction_3 = ["Paraphrase the main focus of this discussion.", "Based on you paraphrase, briefly explain your answer."]
        extra_response = "Paraphrase: [Briefly paraphrase the main focus of the discussion.]\nExplanation: [Based on your paraphrase, briefly explain your classification.]"
        prompt = prompt_template(
            instruction_3=instruction_3, extra_response=extra_response
        )

    return {"PROMPT": add_phi_tag(prompt)}


# Function to convert examples to model inputs with padding
def convert_to_tensors(example):
    """
    Convert the text to model input tensors.

    Args:
        example (dict): A dictionary containing the prompt text under the key "PROMPT".

    Returns:
        dict: A dictionary with keys "input_ids" and "attention_mask", containing the tensor representations.
    """
    tokens = tokenizer(
        example["PROMPT"],
        padding="max_length",
        truncation=True,
        max_length=1000,
        return_tensors="pt",
    )
    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask}


def generate_text(input_ids, attention_mask):
    """
    Generate text from the model based on the input tensors.

    Args:
        input_ids (torch.Tensor): The input IDs to the model.
        attention_mask (torch.Tensor): The attention mask for the model.

    Returns:
        list: A list of generated texts.
    """
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=500,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_length = input_ids.shape[1]
    return [
        tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
        for output in outputs
    ]


def get_labels(output_list):
    """
    Extract labels from the generated output texts.

    Args:
        output_list (list): A list of output texts from which to extract labels.

    Returns:
        list: A list of labels ('yes' or 'no') based on the contents of each output text.
    """
    labels = []
    for output in output_list:
        if (
            output.lower().strip().endswith("yes")
            or output.lower().strip().startswith("yes")
            or "answer: yes" in output.lower()
        ):
            labels.append("yes")
        else:
            labels.append("no")
    return labels


if __name__ == "__main__":
    args = create_arg_parser()
    input_file = args.input_file
    output_file = args.output_file
    batch_size = args.batch_size
    model_name = args.model_name
    selection = args.selection

    # Load dataset and generate prompts
    dataset = load_dataset("json", data_files=input_file)["train"]
    dataset = dataset.map(lambda example: generate_prompt(example, args.prompt_type))

    if selection > 0:
        # Select chunk of data
        start_index = (selection - 1) * 20000
        end_index = min(start_index + 20000, len(dataset))
        indices = list(range(start_index, end_index))
        dataset = dataset.select(indices)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remove_code=True)

    dataset = dataset.map(convert_to_tensors, batched=True)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model, bfloat16 to save time and memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    output_texts = []

    for batch in tqdm(dataloader, desc="Generating Texts"):
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        generated_texts = generate_text(input_ids, attention_mask)
        output_texts.extend(generated_texts)

    if args.test:
        df = dataset.to_pandas()[["DISCUSSION-ID", "PROMPT", "LABEL"]]
    else:
        df = dataset.to_pandas()[["DISCUSSION-ID", "PROMPT"]]
    df["output"] = output_texts
    pred_labels = get_labels(output_texts)
    df["PRED-LABEL"] = pred_labels

    df.to_json(output_file, orient="records", lines=True)
    print(f"Results saved as {output_file}")

    print("Classified as yes: ", pred_labels.count("yes"))

    if args.test:
        print("\n")
        true_labels = dataset["LABEL"]
        cr = classification_report(true_labels, pred_labels)
        print(cr)
        cr_output_path = output_file[:-6] + "_cr.txt"
        xl_output_path = output_file[:-6] + "_mismatch.xlsx"
        mismatch_df = df[df["LABEL"] != df["PRED-LABEL"]]
        mismatch_df.to_excel(xl_output_path)
        print(f"Mismatch overview saved as {xl_output_path}")
        with open(cr_output_path, "w") as f:
            f.write(cr)
        print(f"Classification report saved as {cr_output_path}")

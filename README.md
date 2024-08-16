# Constructing and Analyzing a Dataset of Fact-Checking Discussions from Wikipedia Talk Pages Using Phi-3 and BERTopic

## Abstract
In the digital age, the rapid spread of misinformation and fake news online poses significant challenges to the quality of public discourse. Wikipedia, one of the most widely used and trusted sources of information, operates on an open-edit model where content is contributed and reviewed by volunteer users. While this model allows for a vast and diverse repository of knowledge, it also raises concerns about the reliability and accuracy of its content. This thesis aims to explore how fact-checking is conducted within Wikipedia’s community by creating and analyzing the Wiki-Fact-check-Interaction dataset, a dataset ofWikipedia Talk Page discussions focused on fact-checking. The process involved extracting relevant discussions from English Wikipedia Talk Pages, experimenting with different prompting techniques using the Phi-3 language model to classify these discussions, and using BERTopic for clustering and topic modeling to uncover patterns in the data.

## Dataset
The resulting dataset can be found on [Kaggle](https://www.kaggle.com/datasets/mariekevdh/wiki-fact-check-interaction). Here you will also find a detailed description of the fields present in the dataset.

## How to Run the Code

### Install requirements
The requirements for all scripts except the BERTopic are listed in requirements.txt. The requirements for the notebook are listed in the notebook itself.

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

### Download base data
The dataset created in the thesis is based on the [Webis-WikiDiscussions-18](https://webis.de/data/webis-wikidiscussions-18.html) dataset. Make sure you have the files downloaded that can be found [here](https://zenodo.org/records/3339152). You will need PAGES.tsv, DISCUSSIONS.tsv and COMMENTS.tsv.

### Merge base data files
For ease of use in the rest of the experiments, the PAGES.tsv, DISCUSSIONS.tsv and COMMENTS.tsv files are first merged. This can be done by running the following script from the command line:
```bash
python merge_data.py -df [path_to_data_folder] -out [output_filename]
```

#### Parameters:
- **`-df`, `--data_folder`**:
  - **Type**: `str`
  - **Description**: Path to the folder that contains the Webis-WikiDiscussions-18 data files.
  - **Default**: `"data/"`

- **`-out`, `--output_file`**:
  - **Type**: `str`
  - **Description**: Name of the output file, without extension, where the results will be saved.
  - **Default**: `"grouped_data"`


### Cleaning and filtering data
To clean and filter the merged data, run the following script from the command line:
```bash
python clean_filter_data.py -in [input_file] -out [output_file] [options]
```
#### Parameters:
- **`-in`, `--input_file`**: 
  - **Type**: `str`
  - **Description**: Specifies the name of the input file. This parameter is required.

- **`-out`, `--output_file`**: 
  - **Type**: `str`
  - **Description**: Specifies the name of the output file, without the file extension. This parameter is required.

- **`-ds`, `--dev_size`**: 
  - **Type**: `int`
  - **Description**: Sets the number of samples to reserve for the development set. Defaults to `200`.

- **`-ts`, `--test_size`**: 
  - **Type**: `int`
  - **Description**: Determines the number of samples to save as a separate test set. Defaults to `100`.

- **`-inter`, `--interaction`**: 
  - **Type**: `bool`
  - **Description**: Controls whether to return only discussions that have interactive elements. Set to `true` by default.

- **`-s`, `--seed`**: 
  - **Type**: `int`
  - **Description**: Seed used for the random shuffling of the data when selecting development and test subsets. Defaults to `42`.

### Manual annotation
If a development and/or test set were created in the previous step, you can easily annotate these by running the following script:
```bash
python manual_annotation.py -in [input_file] -out [output_file] [options]
```
The folder `dev_test_data/` contains an annotated development set of 200 examples and an annotated test set of 100 examples.

#### Parameters
- **`-in`, `--input_file`**
  - **Type**: `str`
  - **Description**: Specifies the path to the input JSON file containing the discussion data (required).

- **`-out`, `--output_file`**
  - **Type**: `str`
  - **Description**: Specifies the path and name of the output file where the annotated data will be saved. The file format is JSON (required).

- **`-lc`, `--label_column`**
  - **Type**: `str`
  - **Description**: The column name in which the annotations will be saved, or an existing column with partial annotations.
  - **Default**: `"LABEL"`

### Classify fact-checking discussions
To execute the classifier on the cleaned and filtered data, use the following command from the command line. Note that you can process a specific subset of the data by using the selection option. Multiple output files can be processed simultaneously in following steps of the process. Also note that the implementation of the language model used in the current setup uses the flash attention mechanism. This requires a high-performance GPU, such as the NVIDIA A100, to function. :

```bash
python classify.py -in [input_file] -out [output_file] [options]
```

#### Parameters
- **`-in`, `--input_file`**: 
  - **Type**: `str`
  - **Description**: Name of the input file (required).

- **`-out`, `--output_file`**: 
  - **Type**: `str`
  - **Description**: Name of the output file (default: `"output.jsonl"`).

- **`-bs`, `--batch_size`**: 
  - **Type**: `int`
  - **Description**: Batch size for processing (default: `4`).

- **`-s`, `--selection`**: 
  - **Type**: `int`
  - **Description**: Chunk of data to process. `1` means the first 20000 rows, `2` means the second 20000 rows, etc. If `0`, all data will be processed (default: `0`).

- **`-m`, `--model_name`**: 
  - **Type**: `str`
  - **Description**: Name of the model to use (default: `"microsoft/Phi-3-mini-4k-instruct"`).

- **`-p`, `--prompt_type`**: 
  - **Type**: `str`
  - **Description**: Type of prompt to use for generating labels (default: `"label"`).

- **`-t`, `--test`**: 
  - **Type**: `bool`
  - **Description**: If set to `true`, a classification report will be printed, and a column `"LABEL"` is expected in the input data with the true labels (default: `false`).

### Process results
After get the generated output using classify.py, the file or files have to be processed to extract the summaries. If the data was processed in sections, they will be now be merged back into one file. To do this run the following script in the command line:
```bash
python process_results.py -r [result_file1 result_file2 ...] -m [main_data_file] -out [output_path]
```
#### Parameters:

- **`-r`, `--result_files`**
  - **Type**: `str`
  - **Description**: Accepts one or more result files separated by spaces. Each file should contain at least the columns 'output' and 'PRED-LABEL'.

- **`-m`, `--main_data`**
  - **Type**: `str`
  - **Description**: Specifies the file path for the main data file. This file must include columns PAGE-ID, PAGE-TITLE, DISCUSSION-TITLE, DISCUSSION-ID, and COMMENTS.
  - **Required**: Yes

- **`-out`, `--output_path`**
  - **Type**: `str`
  - **Description**: Specifies the path and name for the output file where the merged data will be saved, without including the file extension.
  - **Required**: Yes

### Get Wikipedia article content
To find content words that will be used in the clustering process, first the content of the Wikipedia articles that are associated with the Talk Pages has to be fetched. To do this run the following code:

```bash
python get_wiki_article_content.py -in [input_file] -out [output_file]
```

#### Parameters
- **`-in`, `--input_file`**  
  - **Type**: `str`  
  - **Description**: Specifies the path to the input JSON file containing page titles. This is a required parameter.  

- **`-out`, `--output_file`**  
  - **Type**: `str`  
  - **Description**: Specifies the name of the output file where the fetched content will be saved. The file will be saved in JSONL format.  
  - **Default**: `output.jsonl`  

### Get Wikipedia content words
To get the content words from the Wikipedia article content and mask these words in the Wikipedia Talk Page texts, run the following script on the command line:
```bash
python get_content_words.py -in [input_file] -a [article_file] -out [output_file] [options]
```

#### Parameters
- **`-in`, `--input_file`**
  - **Type**: `str`
  - **Description**: Specifies the path to the input JSON file containing comment data. This is a required parameter.

- **`-a`, `--article_file`**
  - **Type**: `str`
  - **Description**: Path to the JSON file containing the full articles. This is a required parameter.

- **`-out`, `--output_file`**
  - **Type**: `str`
  - **Description**: Path for the output file where the processed data will be saved. The output format is JSONL.
  - **Default**: `output.jsonl`

- **`-n`, `--top_n_percent`**
  - **Type**: `int`
  - **Description**: Specifies the top percentage of content words to extract based on TF-IDF scores.
  - **Default**: `5`

- **`-min`, `--min_n`**
  - **Type**: `int`
  - **Description**: The minimum number of content words to extract from each article.
  - **Default**: `30`

- **`-m`, `--mask`**
  - **Type**: `bool`
  - **Description**: If set to `true`, the script will add a new column `TEXT-MASKED` to the comments where the content words are masked.
  - **Default**: `true`

### Clustering and Topic Modeling
Lastly, the created dataset can be used for clustering and topic modeling with BERTopic. To do this, a Jupyter Notebook was created, called BERTopic.ipynb. The notebook allows for clustering, topic modeling, as well as fine-tuning of the created labels with a language model. Note that the implementation of the language model used in the current setup uses the flash attention mechanism. This requires a high-performance GPU, such as the NVIDIA A100, to function. 

The requirements necessary to run the code in this notebook are present in the 'setup' section. When running the notebook in Google Colab, it is advised to restart the session after installing the requirements and before running the rest of the cells.
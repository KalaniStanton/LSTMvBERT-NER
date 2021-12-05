# LSTMvBERT-NER
This repository contains multiple notebooks (created using Google Colab) that transform data from a doccano format for use in training a Bi-LSTM-CRF, fine-tuning a transformer using custom labels, and classifying using the fine-tuned bert-base-ner model.

The reason for splitting the methods into different notebooks is that the number of parameters required for all of these models causes the memory to exceed the amount of RAM allocated to a free Google Colab session. Additionally, splitting the files into their own notebooks makes it easier to identify and reference a specific method in this repository.

## Getting Started

It is highly reccomended that these notebooks be run in Google Colab as this was how they were originally constructed, but can be run on any cuda enabled device by simply modifying the path to the file located in the **Load Data** section of each file. 

## Directory Structure

```
|_README.md
|Data
| |_doccano_test.jsonl
|Notebooks
  |_BiLSTM-CRF_Notebook.py
  |_BERT-Finetune_Notebook.py
  |_BERT-BASE-NER_Notebook.py
```

## Notebooks

The three notebooks in this directory are for the different methods I used to perform entity extraction on text for PDF non-profit annual report documents. These data were extracted from these documents and converted to plain-text sentences, which were then labeled using a [docker image of the Doccano software](https://hub.docker.com/r/doccano/doccano). The following section will provide a brief summary of the methods used in each notebook and why they are classified as distinct from each other.

### Bi-LSTM-CRF

The Bi-LSTM-CRF notebook contains the code for building and training a Bidirectional LSTM network that uses a Viterbi Decoder to return the most likely sequence of tags based on the values trained in a Conditional Random Field matrix. In this CRF, the number of columns represents the tag space, and the number of rows represent the length of each input sequence. Unlike the other notebooks in this repository, this one trains the word embeddings, model, and weights in the conditional random field from scratch.

The notebook begins by importing the data set in a .jsonl format and transforms it into a sequential data structure that can be used by the model, storing the tags, vobabulary, and stores the data as a tuple containing the tokenized words and their labels corresponding at the corresponding index  `input[0] = (words, labels)`

### BERT-Finetune

Unlike the above, the BERT finetune uses the pre-trained word embeddings of the bert-base-cased and bert-based-uncased transformer models but then trains the model on the down-stream task of Named Entity Recognition. This means that this notebook contains two models, with the difference between them being the casing of the tokens that the embeddings were trained on. The model structure is much simpler than it is for the Bi-LSTM-CRF as it is simply the pre-trained transformer putputting a vector of logits (representing the tag space) and uses a greedy-search method of decoding (using the max logit values of the transformer outputs).

The notebook begins similarly to the Bi-LSTM-CRF as it compiles a list of tuples containing the words and tags, but the transformer architecture from the HuggingFace library requires that we transform it into a dataset object that can be used in a data-loader for training and inference. Additionally, the pre-trained tokenizer is applied to the dataset during this transformation phase, which produces the ids, attention maskes, and tags for the tokens in each input sequence. Ultimately we initialize two `CustomDataset` objects, one for training and one for testing, each of which are passed to their own respective dataloader objects for use in training and inference. For model evaluation, we compare the predicted tags with the true tags using the `classification_report_seqeval` function from the `seqeval` package.

This process is repeated for both the `bert-base-cased` and the `bert-base-uncased` finetuning process.

### BERT-BASE-NER

## Data Preprocessing

The dataset used in these notebooks was custom from applying OCR (via `textract`) to PDFs of three non-profit annual reports (Girls Who Code - 2020, Sierra Foundation - 2017, and Sesame Street Foundation - 2020). These were selected because I felt they represented distinct domains and presented information that I was interested in both syntactically coherent and incoherent formats. The documents were then converted to lists of plain-text by splitting on the occurrence of two new-line strings (‘\n\n’) and then were “sentencified” by splitting the resulting strings on any punctuation and removing any remaining, singular new-line (‘\n’) strings from the text. The documents ultimately returned 2,433 distinct sentences, which were loaded into a doccano docker container for labeling. I randomly selected 108 sentences to tag with entities and then use this tagged dataset across all of the models. Ideally, I would like to have more tagged sentences, but given the time constraints of the project I felt it was best to move forward and evaluate my methods on this set. 

The following code for the `rummage` function shows the method by which I would extract lists of text sequences from the PDF documents:

```python
import textract

def rummage(self):
    """Method for extracting sentence strings from text. 
    """
    # Conditionals detect document type and run a rudimentary parsing/cleaning procedure depending for pdfs and htmls
    text = textract.process(filePath)
    text = text.decode("utf-8")
    self.seedSource = str(input("Enter SOURCE (Title of Seed Source Document): ")) #Manual Input of seedSource title/name
    text_chunks = str(text).split('\r\n\r\n')
    clean_chunks = [''.join(filter(lambda x: x in set(string.printable), chunk)) for chunk in text_chunks]
    text_lists = [x.strip().split('\r\n') for x in clean_chunks]
    return text_lists
```

## Results and Evaluation

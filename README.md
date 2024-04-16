# Data Annotation using LLM

## Introduction

Data annotation is a crucial step in the machine learning pipeline, particularly in tasks like sentiment analysis, where labeled data is essential for training accurate models. In this repository, we explore two approaches for annotating data related to COVID-19 using Large Language Models (LLMs) and a bulk annotation tool.

## Approach 1: Data Annotation using LLM

In the first approach, we leverage the capabilities of Large Language Models (LLMs) to automate the data annotation process. We have developed a Python script that interacts with the GEMINI API, a generative AI service, to annotate tweets related to COVID-19 based on their sentiment.

### Installation

To run the data annotation script using LLM, follow these steps:

1. Install the `genai` library:
    ```
    pip install genai
    ```

2. Obtain a GEMINI API key from [GEMINI website](https://gemini.example.com) and configure it in your environment variables.

3. Ensure you have the required CSV file (`Corona_NLP_test.csv`) containing the tweet data.

### Workflow

[Workflow details as described in the previous section]

### Benefits

[Benefits as described in the previous section]

## Approach 2: Data Annotation using Bulk Tool

In the second approach, we employ a bulk annotation tool to annotate the tweet data. This approach focuses on scalability and efficiency through bulk processing, utilizing a pipeline consisting of a sentence encoder and UMAP for dimensionality reduction.

### Installation

To run the bulk annotation tool, follow these steps:

1. Install the required libraries:
    ```
    pip install pandas umap-learn embetter
    ```

2. Ensure you have the required CSV file (`Corona_NLP_test.csv`) containing the tweet data.

### Workflow

[Workflow details as described in the previous section]

### Benefits

[Benefits as described in the previous section]

## Conclusion

[Conclusion as described in the previous section]

Feel free to explore the code and documentation provided in this repository to understand the implementation details of each approach.

If you have any questions or suggestions, please don't hesitate to reach out!

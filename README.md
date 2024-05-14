# Book2Comic: Transforming Books into Comics

## Introduction

Welcome to Book2Comic, an interactive tool that transforms a text file of a book into a comic strip using cutting-edge natural language processing (NLP) techniques, large language models (LLMs), image generation models and convolutional neural network (CNN).

Book2Comic is designed to work through a user-friendly Streamlit interface, and deep learning generative models can be executed either locally or via the Hugging Face API for those who may not have powerful local hardware.

When it comes to the CNN (explained below), API execution was not a possibility, and therefore the alternative we propose is a whole notebook that recreates the full pipeline and can be executed in Google Colab. The drawback being the absence of a user interface, but it allows for a complete experience of all of Book2Comic features.

The code is written 100% in python, and the instructions for its installation and use are described below.

This project was undertaken by Miguel Ángel Dávila Romero and Tomás Alcántara Carrasco, as part of the Unstructured Data course within the Big Data Master's Degree program at Comillas ICAI University.

## Project Components

### Stage 1: Book Analysis
The first stage of the project involves a comprehensive analysis of the book to extract characters, their attributes, and the dialogues they engage in throughout the text. This process is carried out using the following python libraries:

- **BookNLP** [1]: Used for initial text processing to identify characters, locations, and events within the narrative.
- **SciPy and NLTK**: These libraries are employed for part-of-speech tagging and dialogue extraction. SciPy helps with data manipulation and statistical operations, while NLTK is used for linguistic processing.

During this stage, we perform the following tasks:
- **Character Extraction**: Identifying and cataloging all the characters in the book.
- **Attribute Analysis**: Determining the most representative attributes of each character using part-of-speech tagging.
- **Dialogue Extraction**: Collecting all dialogues and interactions involving the characters.

To refine the attributes of the characters, we use the **Mistral** language model, which provides natural language descriptions for the characters based on the attributes that were extracted before. This model can be executed locally or via the Hugging Face API, for resource limited devices.

### Stage 2: Scene Generation
In the scene generation stage, we use the **Mistral** model to process each paragraph of the text. The steps involved are:

1. **Text Processing**: For each paragraph, determine if it contains dialogue or descriptive text.
2. **Dialogue Handling**: If the paragraph contains dialogue, it is extracted directly.
3. **Description Generation**: For descriptive paragraphs, a large language model (LLM) generates a narrative scene description based on the extracted attributes and context. For dialogues, a close up shot of the speaker is used.

### Stage 3: Image Generation
The next stage involves generating images based on the previously created prompts. This is done using a fine-tuned **Stable Diffusion** modelm specifically trained for comic book looking images:

- **Prompt Utilization**: The prompts generated from the text analysis are used to create visual representations of the scenes.
- **Stable Diffusion Model**: An open-source image generation model that can be run locally or via the Hugging Face API to generate high-quality images based on the textual descriptions.

### Stage 4: Speech Bubble Addition
For adding speech bubbles to the comic panels, we utilize a CNN [2] which detects the faces in the scene:

- **ResNet Model**: A fine-tuned ResNet model specifically adjusted for handling drawings and cartoons.

Once the face position has been located, a bubble is placed in the right place (not covering the face), and the dialogue is written inside it.

This is an optional stage that can only be done either if the device executing the streamlit app is powerful enough and has CUDA installed, or via the notebook.

## How to Run the Project

in order to run Book2Comic locally, follow these steps:

1. **Install General Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Requirements for Local Face Detection (OPTIONAL)**:
   ```bash
   pip install -r requirements_face_detection_local.txt
   ```

3. **Install Requirements for Stable Diffusion (OPTIONAL)**:
   ```bash
   pip install -r requirements_stablediffusion_local.txt
   ```

4. **Install Requirements for Mistral (OPTIONAL)**:
   ```bash
   pip install -r requirements_mistral_local.txt
   ```

Alternatively, you can use an API key from Hugging Face to run most components remotely, which is especially useful if you do not have a powerful local machine. However, you will lose the character detection and bubble placement feature.

### Running the Streamlit Application

To launch the Streamlit application, use the following command:
```bash
streamlit run app.py
```

From that point on, just follow the steps and generate your comic!

## Usage

Book2Comic is very easy to use: just go to the URL where streamlit is running and follow the steps.

### Possible limitations

At the moment, Book2Comic is still a small project which has not been tested in a variety of books. That is why we recomend certain types of books if you want the tool to work correctly, since unfortunately we cannot assure that would be the case otherwise.

The kind of book we recommend is a small tale, where paragraphs are well separated. For example, we ran several tests on Sherlock Holmes' short stories, that can be found in this [link](https://sherlock-holm.es/ascii/).

### Screenshots

Here are some screenshots of the application in action:

1. **Main Interface**: The primary screen where you can upload a book and start the transformation process.
2. **Character Analysis**: A view showing extracted characters and their attributes.
3. **Generated Comics**: The final comic panels with images and dialogues.

## Additional Notes

We have implemented the use of Hugging Face APIs to ensure replicability of the project, making it accessible even to those without high-performance hardware. All parts of the project can be executed using these APIs, except for the speech bubble generation, as we have not yet found a suitable hosting solution for this model.

We hope you find Book2Comic a valuable tool for converting your favorite books into engaging comic strips. Enjoy transforming your stories!

## References

[1] [BookNLP, a natural language processing pipeline for books]( https://github.com/booknlp/booknlp?tab=readme-ov-file)

[2] [FaceDetector: A face detector model for both real-life and comic images based on RetinaFace model.](https://github.com/barisbatuhan/FaceDetector?tab=readme-ov-file)
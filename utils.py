import os
import base64
import json
import spacy
import requests
import io
import re
import pandas as pd
from booknlp.booknlp import BookNLP
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

# AUXILIARY FUNCTIONS

# Textual part

def clean_text(text):
    # Remove newline characters
    text = text.replace('\n', ' ')
    
    # Remove sequences of hyphens
    text = re.sub(r'-+', '', text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text

def call_model(prompt, api_key=None, use_local_model=False):
    use_local_model = False # Disabled, as for now
    if use_local_model:
        from llama_cpp import Llama
        llm = Llama(model_path='./llama.cpp/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf') 
        output = llm(prompt, max_tokens=300, echo=True, temperature=0.2)
        return clean_text(output["choices"][0]["text"])
    else:
        # Tu clave API de Hugging Face
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

        payload = {"inputs": prompt}

        response = requests.post(API_URL, headers=headers, json=payload)
        return clean_text(response.json()[0]["generated_text"])

def process_book(directory_name, input_file, book_id):

    if os.path.exists(f"booknlp_files/{directory_name}/{book_id}.book"):
        return proc(f"booknlp_files/{directory_name}/{book_id}.book")

    model_params={
        "pipeline":"entity,quote,supersense,event,coref", 
        "model":"big"
    }
    
    booknlp=BookNLP("en", model_params)

    # Output directory to store resulting files in
    output_directory= f"booknlp_files/{directory_name}/"

    booknlp.process(input_file, output_directory, book_id)

    return proc(f"booknlp_files/{directory_name}/{book_id}.book")

def proc(filename):
    with open(filename) as file:
        data=json.load(file)
    return data

def get_counter_from_dependency_list(dep_list):
    counter=Counter()
    for token in dep_list:
        term=token["w"]
        tokenGlobalIndex=token["i"]
        counter[term]+=1
    return counter

def create_character_data(data, printTop):
    character_data = {}
    for character in data["characters"]:

        agentList=character["agent"]
        patientList=character["patient"]
        possList=character["poss"]
        modList=character["mod"]

        character_id=character["id"]
        count=character["count"]

        referential_gender_distribution=referential_gender_prediction="unknown"

        if character["g"] is not None and character["g"] != "unknown":
            referential_gender_distribution=character["g"]["inference"]
            referential_gender=character["g"]["argmax"]

        mentions=character["mentions"]
        proper_mentions=mentions["proper"]
        max_proper_mention=""
        
        #Let's create some empty lists that we can append to.
        poss_items = []
        agent_items = []
        patient_items = []
        mod_items = []
    
        # just print out information about named characters
        if len(mentions["proper"]) > 0:
            max_proper_mention=mentions["proper"][0]["n"]
            for k, v in get_counter_from_dependency_list(possList).most_common(printTop):
                poss_items.append((v,k))
                
            for k, v in get_counter_from_dependency_list(agentList).most_common(printTop):
                agent_items.append((v,k))     

            for k, v in get_counter_from_dependency_list(patientList).most_common(printTop):
                patient_items.append((v,k))     

            for k, v in get_counter_from_dependency_list(modList).most_common(printTop):
                mod_items.append((v,k))  

            
            
            
            # print(character_id, count, max_proper_mention, referential_gender)
            character_data[character_id] = {"id": character_id,
                                  "count": count,
                                  "max_proper_mention": max_proper_mention,
                                  "referential_gender": referential_gender,
                                  "possList": poss_items,
                                  "agentList": agent_items,
                                  "patientList": patient_items,
                                  "modList": mod_items
                                 }
                                
    return character_data

def generate_attributes(text, character_list):

    # Load the English NLP model
    nlp = spacy.load("en_core_web_sm")

    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize a dictionary to store descriptions for each character
    character_descriptions = {c['names'][0]: {'all_names': c['names'], 'gender': c['gender'], 'attributes': []} for c in character_list}
    
    # Iterate through named entities in the document
    for ent in doc.ents:
        for character in character_list:
            character_names = character["names"]
            this_character_descriptions = []
            if ent.text in character_names:
                # Find the sentences containing any of the character names
                sentences = [sent.text for sent in doc.sents if any(name in sent.text for name in character_names)]
                # Extract descriptions from the context
                descriptions = []
                for sentence in sentences:
                    # Example: Extract adjectives describing the character
                    for token in nlp(sentence):
                        if token.pos_ == "ADJ":
                            descriptions.append(token.text)
                    # Add more sophisticated logic here for detailed extraction
                this_character_descriptions += descriptions

            for description in this_character_descriptions:
                if description not in character_descriptions[character_names[0]]['attributes']:
                    character_descriptions[character_names[0]]['attributes'].append(description)

    return character_descriptions

def generate_descriptions(characters, hb_api_key, use_local_model=False):

    descriptions = {}

    for character in characters:

        name = characters[character]['all_names'][0]
        gender = characters[character]['gender']
        attributes = characters[character]['attributes']

        print(f"Generating description for {name}")

        if attributes:

            prompt = f'''Given the following name of a character, their gender and a series of attributes (some of them physical, some of them not), please write a short, physical description of the character. You can use the attributes provided or come up with your own as long as they make sense.
            It is very important that you also specify the clothes that the character is wearing. Again, come up with the clothes according to the character's physical attributes, so that it makes sense for them to wear them.
            The description must not include the name of the character: it must be something like: "A tall man with big, blue ayes and browm hair and pale skin... wearing a black suit and a red tie."

            CHARACTER NAME: {name}
            GENDER: {gender}
            ATTRIBUTES: {",".join(attributes)}

            It is important for you to stick to the physical attributes of the character, since this description will be used in order to generate a portrait of the character: their physical appearance, not their personality or background.

            Please, reply only with the description sentence.
            '''
        
        else:

            prompt = f'''Given the following name of a character and their gender, please write a short, physical description of the character. Make up the description based on the name and gender of the character.
            It is very important that you also specify the clothes that the character is wearing. You can come up with the clothes according to the character's physical attributes, so that it makes sense for them to wear them.
            The description must not include the name of the character: it must be something like: "A tall man with big, blue ayes and browm hair and pale skin... wearing a black suit and a red tie."

            CHARACTER NAME: {name}
            GENDER: {gender}

            It is important for you to stick to the physical attributes of the character, since this description will be used in order to generate a portrait of the character: their physical appearance, not their personality or background.

            Please, reply only with the description sentence.'''

        description = call_model(prompt, hb_api_key, use_local_model)

        if "Please, reply only with the description sentence." in description:
            description = description.split("Please, reply only with the description sentence.")[1].strip()
        else:
            description = description.strip()
        if "GENDER:" in description:
            description = description.split("GENDER:")[1].split(' ')[1].strip()

        descriptions[name] = description

    return descriptions

def get_character_ids(entities_path, characters, cont=0):

    df_entities = pd.read_csv(entities_path, delimiter="\t")
    
    n_characters = len(characters)

    for index, entity in df_entities.iterrows():
        name = entity['text']
        coref = entity['COREF']
        for character in characters:
            if 'coref' not in character:
                if name in character['names']:
                    character['coref'] = coref
                    cont += 1
                    break
        if cont == n_characters:
            break
    
    return characters

def construct_quote_to_character(directory_name, book_id, characters):

    df_quotes = pd.read_csv(f"booknlp_files/{directory_name}/{book_id}.quotes", delimiter="\t")

    quotes = {}

    for index, quote in df_quotes.iterrows():
        text = quote['quote']
        char_id = quote['char_id']
        for character in characters:
            if char_id == character['coref']:
                quotes[text] = (char_id, character['names'][0])
                break
        else:
            quotes[text] = (char_id, None)

    return quotes

def split_into_paragraphs(input_file):
    with open(input_file) as file:
        text = file.read()
    # Use a regular expression to split the text wherever there are two or more newline characters
    paragraphs = re.split(r'\n{2,}', text)
    # Remove any residual leading or trailing whitespace from each paragraph
    return [re.sub(r'\n\s+', ' ', paragraph).strip() for paragraph in paragraphs]

def get_characters_in_scene(paragraph, characters, descriptions, quotes):
    # Initialize an empty list to store the characters found in the scene
    characters_in_scene = []
    aux = []
    is_quote = False
    
    # Iterate over the characters to check if they are mentioned in the paragraph
    for character in characters:
        # Check if the character's name or any of their aliases are mentioned in the paragraph
        for alias in character['names']:
            if alias in paragraph:
                characters_in_scene.append({'name': alias, 'description': descriptions[character['names'][0]]})
                aux.append(alias)
                break
        
    # Iterate over the quotes to check if the character is speaking in the paragraph
    for quote in quotes:
        if quote in paragraph:
            if quotes[quote][1] not in aux:
                characters_in_scene.append({'name': quotes[quote], 'description': descriptions[quotes[quote]]})
                is_quote = True
                return characters_in_scene, is_quote

    if '[NARRATOR]' in [c['names'][0] for c in characters] and '[NARRATOR]' not in aux:
        characters_in_scene.append({'name': '[NARRATOR]', 'description': descriptions['[NARRATOR]']})
    
    # Return the list of characters found in the scene
    return characters_in_scene, is_quote

def generate_scene_from_paragraph(paragraph, characters, descriptions, quotes, hb_api_key, use_local_model=False, previous_scene=None):

    characters_in_scene, is_quote = get_characters_in_scene(paragraph, characters, descriptions, quotes)

    print("GENERATING SCENE FOR PARAGRAPH:")
    print(paragraph)
    print("CHARACTERS IN SCENE:")
    print([c['name'] for c in characters_in_scene])

    characters_str = ""

    for character in characters_in_scene:
        characters_str += f"{character['name']}"
        if character['description']:
            characters_str += f": {character['description']}"
        characters_str += "\n"

    if previous_scene:

        prompt = f'''I am going to give you a paragraph extracted from a tale, and I want you to extract a prompt useful for a text to image model (DallE, Stable Diffusion, Midjourney) that represents a scene.
    Attached to the paragraph, you also have a description of the characters that are mentioned in it, so that you use it in the prompt if you think it is necessary.
    Do not include the name of the characters in the prompt: only their description is necessary for the model to generate the image.
    Keep the prompt as short and descriptive as possible, ensuring the result will be a great image that encapsules the scene that the paragraph is describing.
    IMPORTANT: If the paragraph is a dialogue, just stick to a close up of the speaker: no text is required for the scene, only the description for the image.

    Please, reply with the prompt only, without any additional text.

    PARAGRAPH:

    {paragraph}

    CHARACTERS:
    {characters_str}

    PREVIOUS SCENE:
    {previous_scene}

    IMAGE PROMPT:'''
    
    else:

        prompt = f'''I am going to give you a paragraph extracted from a tale, and I want you to extract a prompt useful for a text to image model (DallE, Stable Diffusion, Midjourney) that represents a scene.
    Attached to the paragraph, you also have a description of the characters that are mentioned in it, so that you use it in the prompt if you think it is necessary.
    Do not include the name of the characters in the prompt: only their description is necessary for the model to generate the image.
    Keep the prompt as short and descriptive as possible, ensuring the result will be a great image that encapsules the scene that the paragraph is describing.
    IMPORTANT: If the paragraph is a dialogue, just stick to a close up of the speaker: no text is required for the scene, only the description for the image.

    Please, reply with the prompt only, without any additional text.

    PARAGRAPH:

    {paragraph}

    CHARACTERS:
    {characters_str}

    IMAGE PROMPT:'''
        
    scene = call_model(prompt, hb_api_key, use_local_model)

    if "IMAGE PROMPT:" in scene:
        scene = scene.split("IMAGE PROMPT:")[1].strip()
    else:
        scene = scene.strip()

    return scene

def generate_text_from_paragraph(paragraph, hb_api_key, use_local_model=False):

    prompt = f'''I am going to give you a paragraph extracted from a tale. I have an image that represents the scene, and what I want you to do is give me the piece of text that will be attached to it, kind of like a comic book.
Give me the little text that would suit the panel. Think of it as a comic book panel, where the text is a short sentence that complements the image, but doesn't repeat what is already shown in it.

Please, reply with the text only, without any additional text.

PARAGRAPH:

{paragraph}

TEXT:'''

    text = call_model(prompt, hb_api_key, use_local_model)

    if "TEXT:" in text:
        text = text.split("TEXT:")[1].strip()
    else:
        text = text.strip()

    return text

def generate_textual_scene(paragraph, characters, descriptions, quotes, hb_api_key, use_local_model=False, previous_scene=None):
    return {'prompt': generate_scene_from_paragraph(paragraph, characters, descriptions, quotes, hb_api_key, use_local_model, previous_scene), 'text': generate_text_from_paragraph(paragraph, hb_api_key, use_local_model)}

# Image part

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def create_book_cover(title: str, author: str, file_path: str):
    # Create a blank A4 image with a black background
    img = Image.new('RGB', (2480, 3508), color='black')
    draw = ImageDraw.Draw(img)

    # Set the title and author fonts
    title_font = ImageFont.truetype("static/Georgia.ttf", 120)
    author_font = ImageFont.truetype("static/Georgia.ttf", 100)
    c2b_font = ImageFont.truetype("static/Georgia.ttf", 80)

    # Calculate text width and height to center the text
    title_width, title_height = textsize(title, font=title_font)
    author_width, author_height = textsize(author, font=author_font)
    c2b_width, c2b_height = textsize("Adapted by Book2Comic", font=c2b_font)

    # Calculate positions
    title_x = (img.width - title_width) / 2
    title_y = (img.height / 2) - title_height
    author_x = (img.width - author_width) / 2
    author_y = (img.height / 2) + (author_height / 2)
    c2b_x = (img.width - c2b_width) / 2
    c2b_y = (img.height / 2) + (c2b_height / 2)

    # Draw text on the image
    draw.text((title_x, title_y), title, font=title_font, fill="white")
    draw.text((author_x, author_y), author, font=author_font, fill="white")
    draw.text((c2b_x, c2b_y), "Adapted by Book2Comic", font=c2b_font, fill="white")

    # Save the image
    img.save(file_path, 'PNG')

def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current_line = words[0]

    for word in words[1:]:
        test_line = f"{current_line} {word}"
        test_width, _ = textsize(test_line, font)
        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    
    lines.append(current_line)  # Add the last line
    return lines

def create_comic_grid(images_paths, dialogues, save_path='comic.png', grid_size=(3,2), margin=20, min_text_height=80):
    num_filas, num_columnas = grid_size
    images = [Image.open(path) for path in images_paths]
    ancho_img, alto_img = images[0].size
    
    try:
        font = ImageFont.truetype("static/Georgia.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    text_heights = [min_text_height] * len(images)
    wrapped_text = []

    for index, text in enumerate(dialogues):
        lines = wrap_text(text, font, ancho_img - margin)
        wrapped_text.append("\n".join(lines))
        line_height = textsize("Test", font)[1] + 5
        total_height = line_height * len(lines)
        text_heights[index] = max(total_height, min_text_height)
    
    comic_height = margin
    for i in range(num_filas):
        max_text_height = max(text_heights[i * num_columnas:(i + 1) * num_columnas])
        comic_height += alto_img + max_text_height + margin

    comic_width = num_columnas * ancho_img + (num_columnas + 1) * margin
    comic = Image.new('RGB', (comic_width, comic_height), 'white')
    
    current_top = margin
    for fila in range(num_filas):
        max_text_height = max(text_heights[fila * num_columnas:(fila + 1) * num_columnas])
        for columna in range(num_columnas):
            index = fila * num_columnas + columna
            image = images[index]
            left = columna * ancho_img + (columna + 1) * margin
            comic.paste(image, (left, current_top))
            
            draw = ImageDraw.Draw(comic)
            text_top = current_top + alto_img + 5
            draw.multiline_text((left, text_top), wrapped_text[index], font=font, fill="black")
        
        current_top += alto_img + max_text_height + margin
    
    comic.save(save_path)
    print(f"The page was saved in {save_path}")

def generate_image(prompt, output_file, api_token):

    API_URL = "https://api-inference.huggingface.co/models/blink7630/graphic-novel-illustration"
    headers = {"Authorization": f"Bearer {api_token}"}

    payload = {'inputs': prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200 and response.headers['Content-Type'].startswith('image'):
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        image.save(output_file)
        print(f"Image saved in '{output_file}'.")
    else:
        print("Error:", response.status_code)
        print("Respuesta:", response.text)

# UTILITY FUNCTIONS

# Textual part

def extract_characters(directory_name, input_file, book_id):

    data = process_book(directory_name, input_file, book_id)
    character_data = create_character_data(data, 1)
    character_list = {a['max_proper_mention']: a['referential_gender'] for a in character_data.values()}

    with open(f"booknlp_files/{directory_name}/{book_id}.book.html") as file:
        data = file.read()
    
    character_other_names = []
    for a in data.split("<h2>Named characters</h2>")[1].split("<p>")[0].split("<br />"):
        aux = a.replace("\n", "")
        if aux != "":
            if "(" not in aux:
                main_name = aux.split(" ")[1]
            else:
                main_name = " ".join(aux.split(" (")[0].split(" ")[1:])
            character_other_names.append([main_name])
    
    named_characters = []

    cont = 0
    for c in character_other_names:
        if c[0] == "[NARRATOR]":
            named_characters.append({'names': c, 'gender': 'unknown', 'coref': 0})
            cont = 1
        elif c[0] in character_list:
            named_characters.append({'names': c, 'gender': character_list[c[0]]})
        else:
            named_characters.append({'names': c, 'gender': 'unknown'})

    named_characters = get_character_ids(f"booknlp_files/{directory_name}/{book_id}.entities", named_characters, cont)
    
    return named_characters

def extract_descriptions(input_file, named_characters, hb_api_key, use_local_model=False):
        
    with open(input_file) as file:
        book_text = file.read()
    
    print("Generating attributes...")
    characters = generate_attributes(book_text, named_characters)

    print("Generating descriptions...")
    descriptions = generate_descriptions(characters, hb_api_key, use_local_model)
    
    return descriptions

def generate_textual_scenes(paragraphs, characters, descriptions, quotes, hb_api_key, use_local_model=False, previous_scene=None):
    scenes = []
    for paragraph in paragraphs:
        scene = generate_textual_scene(paragraph, characters, descriptions, quotes, hb_api_key, use_local_model, previous_scene)
        previous_scene = scene['prompt']
        scenes.append(scene)
    return scenes

# Image part

def generate_comic_page(scenes, first_scene, book_id, api_token):

    images_paths = []
    dialogues = []

    # Create the output dir if it doesn't exist
    os.makedirs(f"output_dir/{book_id}/img/scenes", exist_ok=True)
    os.makedirs(f"output_dir/{book_id}/img/pages", exist_ok=True)

    for i, scene in enumerate(scenes):
        image_path = f"output_dir/{book_id}/img/scenes/scene{i+first_scene}.png"
        generate_image(scene['prompt'], image_path, api_token)
        images_paths.append(image_path)
        dialogues.append(scene['text'])
        print(f"Scene {i+first_scene} generated")

    output_file = f"output_dir/{book_id}/img/pages/page{int(first_scene/6)}.png"
    
    create_comic_grid(images_paths, dialogues, output_file, grid_size=(3,2), margin=20, min_text_height=80)
    
    print("Page generated")

    return output_file

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href
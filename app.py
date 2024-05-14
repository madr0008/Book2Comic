import streamlit as st
import os
from utils import split_into_paragraphs, extract_characters, construct_quote_to_character, extract_descriptions, generate_textual_scenes, generate_comic_page, get_binary_file_downloader_html, get_character_ids

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False,2:False,3:False}

if 'multiselect_change' not in st.session_state:
    st.session_state.multiselect_change = False

if 'character_names' not in st.session_state:
    st.session_state.character_names = []

if 'character_names_aux' not in st.session_state:
    st.session_state.character_names_aux = []

if 'character_list' not in st.session_state:
    st.session_state.character_list = []

if 'description_dict' not in st.session_state:
    st.session_state.description_dict = {}

if 'new_descriptions_set' not in st.session_state:
    st.session_state.new_descriptions_set = False

if 'current_paragraph' not in st.session_state:
    st.session_state.current_paragraph = 0

if 'textual_scenes' not in st.session_state:
    st.session_state.textual_scenes = []

if 'next_page' not in st.session_state:
    st.session_state.next_page = False

if 'hf_api_key' not in st.session_state:
    st.session_state.hf_api_key = ""

if 'local_dependencies' not in st.session_state:
    st.session_state.local_dependencies = False

if 'descriptions_extracted' not in st.session_state:
    st.session_state.descriptions_extracted = False

def set_character_names(character_names):
    st.session_state.character_names = character_names

def set_character_names_aux(character_names_aux):
    st.session_state.character_names_aux = character_names_aux

def set_character_list(character_list):
    st.session_state.character_list = character_list

def set_description_dict(description_dict):
    st.session_state.description_dict = description_dict

def multiselect_change():
    st.session_state.multiselect_change = True
    set_character_names(st.session_state['Characters'])

def description_change(character_name):
    st.session_state.description_dict[character_name] = st.session_state[character_name]

def set_next_page(a):
    st.session_state.next_page = a

def clicked(button):
    st.session_state.clicked[button] = True
    if button == 2:
        multiselect_change()
    if button == 3:
        set_next_page(True)


favicon_path = "static/book2comic_logo.png" # Path to the favicon 
st.set_page_config(page_title="Book2Comic", page_icon=favicon_path, initial_sidebar_state="auto")
style = "style='text-align: center;'"  # Define the style for the HTML elements
# Define the HTML markup for the title with favicon
title_with_favicon = f"""
    <head>
        <title>Book2Comic</title>
        <link rel="shortcut icon" href="{favicon_path}">
    </head>
    <body>
        <h1 style='text-align: center;'>Book2Comic</h1>
    </body>
"""

# Render the title with favicon
st.markdown(title_with_favicon, unsafe_allow_html=True)
st.write(f"<p {style}>Your tool for synthetic comics generation</p>", unsafe_allow_html=True)

# Radio button to check if the user has local dependencies or prefers to use api calls
st.write(f"<h2 {style}>Choose your option</h2>", unsafe_allow_html=True)
option = st.radio("Choose your option", ("I have installed local dependencies for using models locally", "I prefer to use Hugging Face API calls"))

if option == "I have installed local dependencies for using models locally":
    st.session_state.local_dependencies = True
else:
    st.session_state.local_dependencies = False
    # Input Hugging Face API key
    st.session_state.hf_api_key = st.text_input("Input your Hugging Face API key")

if st.session_state.local_dependencies or st.session_state.hf_api_key != "":

    # Subir txt
    st.write(f"<h2 {style}>Upload your file</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        uploaded_file = st.file_uploader("Upload a file...", type=['txt'])

    if uploaded_file is not None:
        
        # Get uploaded_file name
        uploaded_file_name = uploaded_file.name
        input_file = f"uploaded_files/{uploaded_file_name}"
        st.session_state.directory_name = input_file.split('/')[-1].split('.')[0]

        # Save it with the same name under the path uploaded_files
        with open(f"uploaded_files/{uploaded_file_name}", "wb") as f:
            f.write(uploaded_file.getvalue())

        #Input a text area for introducing the name of the book
        st.write(f"<h2 {style}>Introduce the name of the book</h2>", unsafe_allow_html=True)
        st.session_state.book_name = st.text_input("Book name")
        st.session_state.book_files_name = st.session_state.book_name.lower().strip().replace(" ", "_")
        st.session_state.book_author = st.text_input("Book author")
        if 'paragraphs' not in st.session_state:
            st.session_state.paragraphs = split_into_paragraphs(input_file)
        if 'n_paragraphs' not in st.session_state:
            st.session_state.n_paragraphs = len(st.session_state.paragraphs)

        st.button('Extract characters', on_click=clicked, args=[1])

        if st.session_state.clicked[1]:

            if not st.session_state.multiselect_change:

                # Extract characters
                character_list = extract_characters(st.session_state.directory_name, input_file, st.session_state.book_files_name)
                character_names = [c['names'][0] for c in character_list]
                character_names_aux = [c['names'][0] for c in character_list]
                set_character_list(character_list)
                set_character_names(character_names)
                set_character_names_aux(character_names_aux)

            # Show characters in checkboxes in order to select those that the user wants
            st.write(f"<h2 {style}>Select the characters</h2>", unsafe_allow_html=True)
            selected_characters = st.multiselect("Characters", st.session_state.character_names_aux, st.session_state.character_names, on_change=multiselect_change, key='Characters')
            st.write(f"<h3 {style}>Add more characters</h3>", unsafe_allow_html=True)
            st.write("You can input each character between bracket and separate them by ';'.")
            st.write("Inside each bracket, you can input several ways of calling a character, separated by commas.")
            st.write("After the bracket, you must specify their gender after '_'.")
            st.write("For example: ")
            st.write("[Character1,AnotherWayOfCallingCharacter1]_male;[Character2]_female, ...")
            new_characters = st.text_input("Character's names")

            st.button('Extract descriptions', on_click=clicked, args=[2])

            if st.session_state.clicked[2]:

                if not st.session_state.clicked[3]:

                    # Update character_dict
                    new_character_list = [c for c in st.session_state.character_list if c['names'][0] in st.session_state['Characters']]
                    new_character_list_aux = []
                    if new_characters:
                        new_characters = new_characters.split(';')
                        for character in new_characters:
                            names = character.split('_')[0]
                            gender = character.split('_')[-1]
                            names = names.split('[')[1].split(']')[0].split(',')
                            new_character_list_aux.append({'names': names, 'gender': gender})
                        new_character_list_aux = get_character_ids(f"booknlp_files/{st.session_state.directory_name}/{st.session_state.book_files_name}.entities", new_character_list_aux)
                        new_character_list += new_character_list_aux

                    set_character_list(new_character_list)

                    # Extract quotes
                    if 'quotes' not in st.session_state:
                        st.session_state.quotes = construct_quote_to_character(st.session_state.directory_name, st.session_state.book_files_name, st.session_state.character_list)

                    # Extract descriptions
                    if not st.session_state.descriptions_extracted:
                        description_dict = extract_descriptions(input_file, st.session_state.character_list, st.session_state.hf_api_key, st.session_state.local_dependencies)
                        st.session_state.descriptions_extracted = True
                        set_description_dict(description_dict)

                # Show character (key) and description (value) in table, allowing to edit the descriptions
                st.write(f"<h2 {style}>Character descriptions</h2>", unsafe_allow_html=True)
                new_description_dict = {}
                for character in st.session_state.character_list:
                    character_name = character['names'][0]
                    new_description_dict[character_name] = st.text_area(character_name, st.session_state.description_dict[character_name])
                
                st.button('Generate first page', on_click=clicked, args=[3])

                if st.session_state.clicked[3]:

                    if not st.session_state.new_descriptions_set:
                        set_description_dict(new_description_dict)
                        st.session_state.new_descriptions_set = True

                    if st.session_state.next_page:

                        st.session_state.next_page = False

                        # Generate scenes
                        if st.session_state.current_paragraph < st.session_state.n_paragraphs:
                            n_sum = 6
                            if st.session_state.current_paragraph + n_sum > st.session_state.n_paragraphs:
                                n_sum = st.session_state.n_paragraphs - st.session_state.current_paragraph
                            paragraphs = st.session_state.paragraphs[st.session_state.current_paragraph:st.session_state.current_paragraph+n_sum]
                            if st.session_state.current_paragraph == 0:
                                scenes = generate_textual_scenes(paragraphs, st.session_state.character_list, st.session_state.description_dict, st.session_state.quotes, st.session_state.hf_api_key, st.session_state.local_dependencies)
                            else:
                                scenes = generate_textual_scenes(paragraphs, st.session_state.character_list, st.session_state.description_dict, st.session_state.quotes, st.hf_api_key, st.session_state.textual_scenes[-1], st.session_state.local_dependencies)

                        # Show scenes in table with characters, background_description, scene_description and dialogue, allowing to edit any of them
                        st.write(f"<h2 {style}>Comic scenes for page {int(st.session_state.current_paragraph/6)+1}</h2>", unsafe_allow_html=True)
                        st.write(f"<h3 {style}>Textual representation</h3>", unsafe_allow_html=True)
                        col1, col2 = st.columns([1,1])
                        cont = 1
                        for scene in scenes:
                            scene_str = f"Scene {st.session_state.current_paragraph + cont}"
                            with col1:
                                scene['prompt'] = st.text_area(scene_str + "'s image description", scene['prompt'], key=scene_str + "_prompt")
                            with col2:
                                scene['text'] = st.text_area(scene_str + "'s text", scene['text'], key=scene_str + "_text")
                            cont += 1
                        
                        st.session_state.textual_scenes += scenes
                        st.session_state.current_textual_scenes = scenes
                        st.session_state.previous_paragraph = st.session_state.current_paragraph
                        st.session_state.current_paragraph += n_sum

                    if st.button('Generate images', on_click=set_next_page, args=[False]):
                        # Generate comic
                        output_file = generate_comic_page(st.session_state.current_textual_scenes, st.session_state.previous_paragraph, st.session_state.book_files_name, st.session_state.hf_api_key, st.session_state.local_dependencies)
                        st.write(f"<h2 {style}>Comic page generated</h2>", unsafe_allow_html=True)
                        st.image(output_file, use_column_width=True)

                    st.button("Generate next page", on_click=set_next_page, args=[True])
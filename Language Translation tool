import os
from googletrans import Translator

# Set up the translator
translator = Translator()

def translate_text(text, source_language, target_language):
    """
    Translate text from one language to another using Google Translate API
    """
    try:
        result = translator.translate(text, src=source_language, dest=target_language)
        return result.text
    except Exception as e:
        return str(e)

def main():
    # Get the text to translate from the user
    text = input("Enter the text to translate: ")

    # Get the source and target languages from the user
    source_language = input("Enter the source language (e.g. en, es, fr): ")
    target_language = input("Enter the target language (e.g. en, es, fr): ")

    # Translate the text
    translated_text = translate_text(text, source_language, target_language)

    # Print the translated text
    print("Translated text:", translated_text)

if __name__ == "__main__":
    main()

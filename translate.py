from gtts import gTTS
import io
from deep_translator import GoogleTranslator
import textwrap

# Language map
language_map = {
    'English': 'en-us',
    'Spanish': 'es',
    'French': 'fr'
}

def translate(text, lang, way):
    # Split text into chunks of 500 characters
    chunks = textwrap.wrap(text, 500)
    translated_chunks = []

    if way:
        for chunk in chunks:
            translator = GoogleTranslator(source='en', target=language_map[lang])  # Initialize translator for each chunk
            translated_chunks.append(translator.translate(chunk))
    else:
        for chunk in chunks:
            translator = GoogleTranslator(source=language_map[lang], target="en")  # Initialize translator for each chunk
            translated_chunks.append(translator.translate(chunk))

    # Join all translated chunks
    return ' '.join(translated_chunks)

def generate_audio(text, lang):
    if not text:
        raise ValueError("No text to speak.")
    languages = {"English": "en", "French": "fr", "Spanish": "es"}
    lang_code = languages.get(lang, "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return audio_io

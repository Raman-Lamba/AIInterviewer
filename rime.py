import requests
import os

def rime_tts(text, output_filename, speaker="doe_john", modelId="arcana", api_key=None):
    """
    Synthesize speech from text using the Rime API and save as MP3.
    :param text: The text to synthesize.
    :param output_filename: The filename to save the MP3 audio.
    :param speaker: The speaker voice to use (default: 'doe_john').
    :param modelId: The model to use (default: 'arcana').
    :param api_key: The RIME API key (default: from env RIME_API_KEY).
    """
    url = "https://users.rime.ai/v1/rime-tts"
    if api_key is None:
        api_key = os.getenv("RIME_API_KEY")
    if not api_key:
        raise ValueError("RIME_API_KEY not set in environment or passed as argument.")
    payload = {
        "speaker": speaker,
        "text": text,
        "modelId": modelId,
        "repetition_penalty": 1.5,
        "temperature": 0.5,
        "top_p": 0.5,
        "samplingRate": 24000,
        "max_tokens": 1200
    }
    headers = {
        "Accept": "audio/mp3",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers, stream=True)
    if response.status_code == 200:
        with open(output_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    else:
        raise RuntimeError(f"Rime API error: {response.status_code} {response.text}")

# End of module. No print statements needed unless for debugging.
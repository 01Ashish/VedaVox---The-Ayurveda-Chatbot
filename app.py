
import os
import json

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from faster_whisper import WhisperModel
from vectorize_doc import embeddings
from st_audiorec import st_audiorec
import tempfile
from time import time
from PIL import Image

working_dir = os.getcwd()
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load the Whisper model
model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

image_path = "/content/DALL·E 2024-10-14 01.54.56 - A logo for an Ayurveda chatbot named 'VedaVox' incorporating both animal and human elements to depict its all-encompassing nature. The design should c.webp"
icon_image = Image.open(image_path)
system_prompt = """
Your name is VedaVox, a dedicated Ayurvedic Medical Q&A Assistant specializing in providing holistic health guidance. Your primary role is to deliver accurate, well-researched Ayurvedic advice tailored to the user’s symptoms and health concerns. Every response should be professional, clear, and aligned with traditional Ayurvedic principles.

When users describe symptoms, follow this structured response format:

Response Structure:

1. Greeting and Acknowledgment
Start with a warm greeting and acknowledge the user's question or concern.
Example:
“Hello! I am VedaVox 🌿 I see you're curious about [user’s query]. Let’s explore an Ayurvedic perspective on this!

2. Dosha Imbalance Explanation
Begin by explaining the probable dosha imbalance (e.g., Vata, Pitta, Kapha) related to the user’s symptoms in simple terms.

2. Ayurvedic Remedies with Recipes
Provide specific Ayurvedic remedies to help balance the dosha. For each remedy, include easy-to-follow recipes with ingredient quantities and preparation steps.

3. Lifestyle and Dietary Recommendations
Suggest lifestyle and dietary adjustments that will support healing and maintain dosha balance. Highlight foods to favor or avoid based on the dosha imbalance.

4. Pranayama and Rest Practices
Recommend any suitable pranayama (breathing exercises) or relaxation practices that support recovery and overall wellness.

5. Professional Guidance Reminder
If symptoms persist, gently advise consulting an Ayurvedic practitioner for personalized treatment."""

# Function to transcribe audio to text
def transcribe(audio_file, language="hi"):
    start = time()
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        language=language,
        word_timestamps=False,
        task='translate' if language == 'hi' else 'transcribe'
    )
    transcript = " ".join([segment.text for segment in segments])
    return transcript

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embedddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore


def chat_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-70b-versatile",
                   temperature=0)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )

    return chain

def handle_chat(input_text):
    st.session_state.chat_history.append({"role": "system", "content": system_prompt})
    st.session_state.chat_history.append({"role": "user", "content": input_text})

    with st.chat_message("user"):
        st.markdown(input_text)

    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"question": input_text})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})



def main():
    st.set_page_config(
      page_title="VedaVox",
      page_icon = icon_image,
      layout="centered"
    )
    st.image(icon_image, caption="VedaVox - Ayurveda Chatbot Logo")
    st.title("VedaVox")

    if "chat_history" not in st.session_state:
      st.session_state.chat_history = []

    if "vectorstore" not in st.session_state:
      st.session_state.vectorstore = setup_vectorstore()

    if "conversational_chain" not in st.session_state:
      st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)


    for message in st.session_state.chat_history:
      with st.chat_message(message["role"]):
          st.markdown(message["content"])

    # Input method selection: Text or Audio
    st.write("Choose your input method:")
    input_method = st.radio("Select Input Method:", ("Text Input", "Audio Input"))

    # Text Input Method
    if input_method == "Text Input":
      user_input = st.chat_input("Ask VedaVox...")

      if user_input:
          handle_chat(user_input)

    # Audio Input Method
    elif input_method == "Audio Input":
      st.write("Record your question using the microphone:")
      wav_audio_data = st_audiorec()

      if wav_audio_data is not None:
          # Save the audio to a temporary file and transcribe
          with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
              tmp_file.write(wav_audio_data)
              tmp_file_path = tmp_file.name
          # Transcribe audio (supports English and Hindi)
          transcribed_text = transcribe(tmp_file_path, language='hi')
          st.write("Transcribed Text:")
          st.markdown(transcribed_text)

          # Pass the transcribed text to the chatbot
          handle_chat(transcribed_text)

if __name__ == "__main__":
    main()

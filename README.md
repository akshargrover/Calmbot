# CalmBot 🧘

<p>
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/LangChain-FFD700?logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjRkZGIiB2aWV3Qm94PSIwIDAgMzAgMzAiIHdpZHRoPSIzMCIgaGVpZ2h0PSIzMCI+PHJlY3Qgd2lkdGg9IjMwIiBoZWlnaHQ9IjMwIiByeD0iNSIvPjwvc3ZnPg==" alt="LangChain" />
  <img src="https://img.shields.io/badge/Google%20Gemini-4285F4?logo=google&logoColor=white" alt="Google Gemini" />
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas" />
</p>

CalmBot is a compassionate mental health chatbot that uses Retrieval-Augmented Generation (RAG) to provide supportive, empathetic, and informative responses based on curated mental health documents. It leverages Google Gemini and LangChain for natural language understanding and document retrieval.

## Features
- Answers mental health questions using a curated set of PDF documents
- Retrieval-Augmented Generation (RAG) for context-aware responses
- Empathetic, supportive, and non-diagnostic tone
- Quick prompt buttons for common concerns (anxiety, sleep, focus, sadness)
- Option to show document sources for transparency
- Chat history is saved to a CSV file

## Requirements
- Python 3.10
- Google API Key (for Gemini)
- LangChain, Streamlit, and other dependencies (see `requirements.txt`)

## Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Calmbot
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Create a `.env` file in the project root with the following:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     LANGCHAIN_API_KEY=your_langchain_api_key_here
     ```
4. **Add your mental health documents:**
   - Place PDF files in the `mental_health_docs/` directory. Example files are already included.

## Running CalmBot
Start the Streamlit app:
```bash
streamlit run chatbot.py
```

## Usage
- Enter your mental health questions in the chat box.
- Use the quick prompt buttons for common issues.
- Optionally, enable "Show Sources" to see which documents were referenced.
- Chat history is displayed (newest first) and saved to `chat_log.csv`.

## Project Structure
```
Calmbot/
  chatbot.py            # Main Streamlit app
  requirements.txt      # Python dependencies
  chat_log.csv          # Chat history log
  mental_health_docs/   # PDF documents for RAG
  chroma_db/            # Vector store (auto-generated)
  documents/            # (Unused, for future expansion)
  README.md             # This file
```

## Notes
- CalmBot is for informational and supportive purposes only. It does not provide medical or diagnostic advice.
- All responses are generated using a combination of document context and general language model understanding.

## License
See [LICENSE](LICENSE) for details.
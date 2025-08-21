# README: LangChain × Ollama Streamlit Chat App

This project demonstrates a simple and optimized Streamlit application that integrates LangChain with Ollama for conversational AI. The app allows users to interact with various Ollama models using a chat interface, with advanced configuration options and session memory.

## Project Structure

- `app.py`: Basic Streamlit app using LangChain and Ollama for Q&A.
- `optimized_app.py`: Advanced Streamlit app with responsive UI, model selection, session memory, and LangSmith tracing.
- `requirements.txt`: Python dependencies for LangChain, Ollama, Streamlit, and related tools.
- `.env`: Environment variables for LangSmith API and project name.

## How It Works

1. **Environment Setup**: Loads environment variables and configures LangSmith tracing for monitoring and debugging.
2. **Model Integration**: Uses LangChain's Ollama integration to connect to local or remote Ollama servers and select models.
3. **Prompt Engineering**: Employs chat prompt templates for structured conversation, including system and user messages.
4. **Streamlit UI**: Provides a user-friendly chat interface with options to configure model, temperature, context window, and more.
5. **Session Memory**: Maintains chat history for context-aware responses.
6. **Error Handling**: Displays helpful error messages if the Ollama server or model is unavailable.

## Usage

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Start Ollama server (if not running):
   ```powershell
   ollama serve
   ollama pull gemma3:1b
   ```
3. Run the app:
   ```powershell
   streamlit run optimized_app.py
   ```
   or
   ```powershell
   streamlit run app.py
   ```

## Notebook Analysis

If you add a Jupyter notebook, analyze each cell and add comments explaining its purpose. This helps users understand the workflow and logic behind each step.

## Key Features

- **Model Selection**: Easily switch between Ollama models.
- **Advanced Controls**: Adjust temperature, top-p, context window, and max tokens.
- **LangSmith Tracing**: Optional tracing for debugging and analytics.
- **Responsive UI**: Modern, user-friendly chat interface.

## Requirements

- Python 3.8+
- Ollama installed and running
- Streamlit
- LangChain and related packages

## License

This project is for educational and demonstration purposes.

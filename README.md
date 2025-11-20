# Document Summarizer

A web application for summarizing text documents using AI-powered natural language processing. This project uses Hugging Face's BART Large CNN model to generate concise summaries from text input or uploaded documents.

## 🚀 Live Demo

**🔗 Live Application**: [https://doc-summarizer.onrender.com](https://doc-summarizer.onrender.com) *(Update this after deployment)*

**📦 Repository**: [https://github.com/vashishthtushar/doc-summarizer](https://github.com/vashishthtushar/doc-summarizer)

## Features

- **Multiple Summary Styles**: Choose from brief, detailed, or bullet-point summaries
- **Text Input**: Paste text directly into the web interface
- **File Upload**: Support for `.txt` and `.md` file uploads
- **Real-time Processing**: Fast summarization using Hugging Face API
- **Clean Web Interface**: Simple and intuitive user interface
- **Error Handling**: Graceful fallback when API is unavailable

## Technologies Used

- **Backend**: Python, Flask
- **AI/ML**: Hugging Face Transformers API (BART Large CNN model)
- **Frontend**: HTML, CSS, JavaScript
- **Libraries**: 
  - Flask (web framework)
  - Requests (HTTP client)
  - python-dotenv (environment variable management)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Hugging Face API key (free account at [huggingface.co](https://huggingface.co))

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd doc-summarizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file**
   - Copy the example file: `cp .env.example .env` (or `copy .env.example .env` on Windows)
   - Open `.env` file and add your Hugging Face API key:
   ```env
   HF_API_KEY=your_huggingface_api_key_here
   ```
   
   To get your Hugging Face API key:
   - Sign up at [huggingface.co](https://huggingface.co/join)
   - Go to Settings → Access Tokens
   - Create a new token and copy it

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   Open your browser and navigate to: `http://127.0.0.1:5000`

## Usage

### Web Interface

1. Open the application in your browser
2. Choose your input method:
   - **Text Input**: Paste your text in the text area
   - **File Upload**: Click "Choose File" and select a `.txt` or `.md` file
3. Select summary style:
   - **Brief**: Short 2-4 sentence summary
   - **Detailed**: Comprehensive summary with more details
   - **Bullets**: Key points in bullet format
4. Click "Summarize" button
5. View the generated summary below

### API Endpoints

#### Health Check
```
GET /api/health
```
Returns the status of the summarizer service and API connectivity.

#### Summarize
```
POST /api/summarize
Content-Type: application/x-www-form-urlencoded

Parameters:
- text (optional): Text to summarize
- file (optional): File upload (.txt or .md)
- style (optional): Summary style (brief, detailed, bullets)
```

Example using curl:
```bash
curl -X POST http://127.0.0.1:5000/api/summarize \
  -F "text=Your text here" \
  -F "style=brief"
```

## Project Structure

```
doc-summarizer/
│
├── app.py              # Main Flask application
├── summarizer.py       # Core summarization logic
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in repo)
├── .gitignore         # Git ignore file
│
├── templates/         # HTML templates
│   └── index.html    # Main web interface
│
├── static/           # Static files
│   └── styles.css    # CSS styling
│
└── uploads/          # Uploaded files directory
    └── .gitkeep
```

## Configuration

You can customize the application by modifying environment variables in `.env`:

- `HF_API_KEY`: Your Hugging Face API key (required)
- `HF_MODEL`: Model to use (default: `facebook/bart-large-cnn`)
- `CHUNK_SIZE`: Text chunking size for long documents (default: 3000)
- `HF_MAX_RETRIES`: Maximum retry attempts for API calls (default: 3)
- `HF_TEMPERATURE`: Model temperature for generation (default: 0.1)

## Approach and Design Decisions

### Architecture Overview

The application follows a clean separation of concerns with a modular design:

1. **Flask Backend (`app.py`)**: Handles HTTP requests, file uploads, and serves the web interface
2. **Summarization Engine (`summarizer.py`)**: Contains the core logic for text processing and API interactions
3. **Frontend (`templates/index.html`)**: Simple, responsive web interface built with vanilla JavaScript

### Key Design Decisions

**1. API-Based Approach**
- Chose Hugging Face Inference API over local model deployment for:
  - Faster development and setup
  - No need for GPU resources
  - Free tier availability for educational use
  - Easy model switching without code changes

**2. BART Large CNN Model Selection**
- Selected `facebook/bart-large-cnn` specifically because:
  - Trained for summarization tasks (not general language generation)
  - Optimized for CNN/DailyMail datasets (news-style summarization)
  - Produces abstractive summaries (not just extractive)
  - Good balance between quality and response time

**3. Text Chunking Strategy**
- Implemented paragraph-aware chunking to:
  - Handle documents larger than model context limits
  - Preserve semantic coherence by keeping paragraphs together
  - Split only when absolutely necessary
  - Combine chunk summaries for long documents

**4. Style-Based Summarization**
- Three distinct styles with different parameters:
  - **Brief**: Short max_length (50) for concise summaries
  - **Detailed**: Longer max_length (120) for comprehensive coverage
  - **Bullets**: Medium length (80) with prompt engineering for list format

**5. Error Handling and Resilience**
- Multiple layers of error handling:
  - API failures fall back to mock summaries (demo functionality)
  - Retry logic with exponential backoff for transient errors
  - Input validation at multiple levels
  - Graceful degradation when services are unavailable

**6. Security Considerations**
- Environment variables for sensitive data (API keys)
- Input sanitization for file uploads
- Secure filename handling with Werkzeug
- File type and size restrictions

**7. Code Organization**
- Separation of concerns: routing logic separate from summarization logic
- Configurable parameters via environment variables
- Reusable `Summarizer` class that can be imported independently
- Clear function naming and documentation

## How It Works

1. **Text Processing**: The application receives text input either from direct input or file upload
2. **Chunking**: Long texts are split into manageable chunks using paragraph-aware splitting
3. **API Call**: Each chunk is sent to Hugging Face's BART Large CNN model via their inference API
4. **Summary Generation**: The model generates a summary based on the selected style and parameters
5. **Echo Detection**: Output is checked to ensure it's not just echoing the input
6. **Chunk Combination**: If multiple chunks exist, summaries are synthesized into a final coherent summary
7. **Response**: The summary is returned and displayed to the user

The BART Large CNN model is specifically trained for summarization tasks and produces high-quality, coherent summaries through abstractive summarization techniques.

## Error Handling

The application includes robust error handling:
- If the API is unavailable, it returns a fallback summary
- Invalid file types are rejected with appropriate error messages
- Input validation ensures minimum text length requirements
- Connection timeouts are handled gracefully

## Limitations

- Maximum file size: 5 MB
- Supported file formats: `.txt` and `.md` only
- Requires internet connection for API calls
- API rate limits may apply based on your Hugging Face account

## Future Improvements

- Support for more file formats (PDF, DOCX)
- Batch processing for multiple files
- Summary length customization
- Export summaries to different formats
- User authentication and history
- Local model support for offline use

## License

This project is for educational purposes.

## Author

Created as a college project demonstrating the use of AI/ML APIs in web applications.

## Acknowledgments

- Hugging Face for providing free access to their transformer models
- Flask community for excellent documentation
- BART model creators for the powerful summarization capabilities


# PDF Question Answering Tool using Langchain and OpenAI

## Overview

This Python script provides a robust solution for extracting and answering questions from PDF documents using advanced natural language processing techniques. The tool leverages:
- LangChain for document processing
- Hugging Face for embeddings
- OpenAI's GPT models for question answering

## Features

- PDF document loading
- Intelligent text splitting
- Vector store creation
- Semantic similarity-based question answering

## Prerequisites

- Python 3.8+
- OpenAI API Key
- Hugging Face account (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mahato99/basic-rag.git
```

2. Create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Place your PDF file in the project directory (default: `ril.pdf`)
2. Modify the `pdf_path` in `main()` if needed
3. Run the script:
```bash
python rag.py
```

## Customization

- Change embedding model in `create_vector_store()`
- Modify GPT model in `create_qa_chain()`
- Adjust text splitting parameters in `split_documents()`



## Troubleshooting

- Ensure all API keys are correctly set
- Check internet connection
- Verify PDF file integrity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License
 MIT

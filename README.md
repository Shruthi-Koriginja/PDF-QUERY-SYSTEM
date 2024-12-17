# PDF-QUERY-SYSTEM
A Python project to query PDF files using OpenAI

# PDF Query System with OpenAI

## Project Overview
This project allows users to extract information from a PDF file and query its content. It uses OpenAI's GPT model to generate context-aware responses.

## Features
- Extract text from a PDF file.
- Process and split text into smaller chunks for analysis.
- Generate embeddings using OpenAI's text-embedding-ada-002 model.
- Retrieve the most relevant text chunks using FAISS similarity search.
- Generate a response to user queries using OpenAI's GPT model.

## How to Run the Code
1. *Install Dependencies:*
   Run the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt

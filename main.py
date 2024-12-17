# Install Required Libraries
!pip install openai pdfplumber faiss-cpu

# Import Libraries
import openai
import pdfplumber
import faiss
import numpy as np

# Step 1: Set up OpenAI API Key
openai.api_key = "YOUR_API_KEY"  # Replace with your OpenAI API key

# Step 2: Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 3: Function to Split Text into Chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Step 4: Generate Embeddings Using OpenAI
def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

# Step 5: Set Up FAISS Index
def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Step 6: Search Relevant Chunks
def search_query(index, query, text_chunks):
    query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")['data'][0]['embedding']
    query_embedding = np.array([query_embedding])
    distances, indices = index.search(query_embedding, k=3)
    return [text_chunks[i] for i in indices[0]]

# Step 7: Generate Response Using OpenAI GPT
def generate_response(relevant_chunks, query):
    prompt = "Here are some relevant passages:\n\n" + "\n\n".join(relevant_chunks) + f"\n\nAnswer the question: {query}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that answers questions based on context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response['choices'][0]['message']['content']

# Step 8: Main Script
if _name_ == "_main_":
    # Replace with your PDF path
    pdf_path = "path_to_your_pdf.pdf"  # Example: "sample.pdf"

    # Extract and Process PDF Text
    text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text_into_chunks(text)
    print("Extracted and Chunked Text:", text_chunks)

    # Generate Embeddings and Create FAISS Index
    embeddings = generate_embeddings(text_chunks)
    index = create_faiss_index(embeddings)

    # Take Query from User
    query = input("Enter your query: ")
    relevant_chunks = search_query(index, query, text_chunks)
    print("Relevant Chunks:", relevant_chunks)

    # Generate Response
    response = generate_response(relevant_chunks, query)
    print("Response:", response)
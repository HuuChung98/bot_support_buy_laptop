# Laptop Recommendation Chatbot

This project implements an AI-powered chatbot that provides laptop recommendations using Azure OpenAI and Pinecone for vector search capabilities. The chatbot combines RAG (Retrieval Augmented Generation) and function calling to provide accurate and contextual responses.

## Features

- Laptop recommendation based on user requirements
- System status checking functionality
- Vector search using Pinecone
- RAG implementation with LangChain
- Function calling with Azure OpenAI

## Prerequisites

Before running the application, make sure you have:

1. Python 3.8 or higher installed
2. Azure OpenAI API access
3. Pinecone API access

## Required Environment Variables

Create a `.env` file in the project root with the following variables:

```env
AZURE_OPENAI_EMBEDDING_API_KEY=your_embedding_api_key
AZURE_OPENAI_EMBEDDING_ENDPOINT=your_embedding_endpoint
AZURE_OPENAI_EMBEDDING_MODEL=your_embedding_model_name

AZURE_OPENAI_LLM_API_KEY=your_llm_api_key
AZURE_OPENAI_LLM_ENDPOINT=your_llm_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

PINECONE_API_KEY=your_pinecone_api_key
```

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install langchain-pinecone langchain-openai pinecone-client python-dotenv openai
```

## Project Structure

- `bot_support_by_laptop.py`: Main application file
  - Data preparation and vector storage
  - Azure OpenAI and Pinecone setup
  - Function calling implementation
  - RAG implementation using LangChain

## Components

1. **Laptop Data**: Pre-defined laptop information with descriptions and tags
2. **Embeddings**: Uses Azure OpenAI for generating embeddings
3. **Vector Storage**: Pinecone for storing and retrieving vector embeddings
4. **Chat Model**: Azure OpenAI for generating responses
5. **Function Calling**: Implements specific functions for laptop recommendations and system status checks

## Running the Application

1. Ensure all environment variables are set in the `.env` file
2. Run the main script:

```bash
python bot_support_by_laptop.py
```

The script will:
1. Initialize the vector store with laptop data
2. Process a set of predefined queries
3. Display both RAG-based and function-calling-based responses

## Example Queries

The system comes with predefined test queries:
- Lightweight laptop recommendation for business trips
- Gaming laptop with high-end graphics
- Budget laptop for students

## Functions Available

1. `recommend_laptop`: Suggests laptops based on:
   - Usage purpose
   - Budget range
   - Preferred tags

2. `get_laptop_details`: Retrieves detailed information about a specific laptop

3. `check_system_status`: Checks the status of IT devices

## Architecture

The system uses a dual-approach for generating responses:
1. RAG (Retrieval Augmented Generation) using LangChain
2. Function calling through Azure OpenAI

This combination provides both context-aware responses and structured data handling capabilities.
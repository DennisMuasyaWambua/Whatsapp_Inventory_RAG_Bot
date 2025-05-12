# from sqlalchemy import create_engine, inspect
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import ollama
# from typing import Dict, Tuple, List, Any
# import os
# from django.conf import settings

# def create_vector_store_from_db(
#     db_url: str,
#     embedding_model_name: str = 'all-MiniLM-L6-v2'
# ) -> FAISS:
#     """
#     Create a FAISS vector store from database text data.
    
#     Args:
#         db_url (str): SQLAlchemy-compatible database URL
#         embedding_model_name (str): Name of the HuggingFace embedding model
        
#     Returns:
#         FAISS: Vector store for semantic search
#     """
#     # Setup embedding function for LangChain
#     embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
#     # Connect to database and extract text data
#     engine = create_engine(db_url)
#     inspector = inspect(engine)
#     all_docs = []
    
#     with engine.connect() as conn:
#         tables = inspector.get_table_names()
        
#         for table in tables:
#             try:
#                 print(f"Processing table: {table}")
#                 df = pd.read_sql(f"SELECT * FROM {table}", conn)
                
#                 # Auto-detect text columns
#                 text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
#                 if not text_columns:
#                     print(f"Skipping table '{table}' (no text columns)")
#                     continue
                
#                 # Format each row as a document
#                 for _, row in df.iterrows():
#                     # Create metadata to track source
#                     metadata = {
#                         "table": table,
#                         "id": str(row.get("id", "unknown"))
#                     }
                    
#                     # Create document text including column names
#                     text_parts = []
#                     for col in text_columns:
#                         if pd.notna(row[col]) and row[col]:
#                             text_parts.append(f"{col}: {row[col]}")
                    
#                     doc_text = "\n".join(text_parts)
#                     if doc_text.strip():
#                         all_docs.append({"content": doc_text, "metadata": metadata})
            
#             except Exception as e:
#                 print(f"Error processing table '{table}': {e}")
#                 continue
    
#     print(f"Collected {len(all_docs)} documents from database")
    
#     # Split documents if they're too long
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100
#     )
    
#     # Create documents for vector store
#     documents = []
#     for doc in all_docs:
#         chunks = text_splitter.create_documents(
#             texts=[doc["content"]], 
#             metadatas=[doc["metadata"]]
#         )
#         documents.extend(chunks)
    
#     print(f"Created {len(documents)} chunks after splitting")
    
#     # Create and return the vector store
#     vector_store = FAISS.from_documents(documents, embeddings)
#     return vector_store


# # Remove redundant imports since we already have the correct import above:
# # from langchain_community.vectorstores import FAISS
# # from langchain_huggingface import HuggingFaceEmbeddings

# def load_existing_vector_store(path)-> FAISS:
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # replace with your actual model
#     return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)




# def chat_with_database(message):
#     """
#     Interactive chat with database using direct Ollama client.
    
#     Args:
#         db_url (str): Database connection URL
#     """
#     print("Creating vector store from database (this may take a while)...")

#     print(settings.DB_URL)
    
#     db_url = settings.DB_URL

#     BASE_DIR = settings.BASE_DIR  # This should point to the directory containing manage.py

#     vectorOutput = os.path.join(BASE_DIR, 'vector_output')

#     if os.path.exists(vectorOutput) and os.path.isdir(vectorOutput):
#         print("Folder exists")
#         vector_store = load_existing_vector_store(f'{vectorOutput}')  
#     else:
#         print("Folder does not exist")
#         vector_store = create_vector_store_from_db(settings.DB_URL)
    
#     # Create retriever from vector store
#     retriever = vector_store.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 5}  # Return top 5 most relevant chunks
#     )
    
#     # Setup Ollama client directly
#     ollama_client = ollama.Client()
    
#     # Interactive chat loop
#     while True:
#         print("\n" + "-"*50)
#         question = input(message)
#         print("-"*50 + "\n")
        
#         if question.lower() in ["q", "quit", "exit"]:
#             break
            
#         # Get relevant documents from vector store
#         docs = retriever.get_relevant_documents(question)
        
#         # Format context from relevant documents
#         context = "\n\n".join([doc.page_content for doc in docs])
        
#         # Build prompt
#         prompt = f"""
#         You are an expert in answering questions about an ecommerce store database.
#         Use only the information in the provided context to answer the question.
#         If the answer cannot be found in the context, say "I don't have enough information to answer that."
        
#         Context:
#         {context}
        
#         Question: {question}
        
#         Answer:
#         """
        
#         # Send to Ollama directly
#         try:
#             response = ollama_client.chat(
#                 model="llama2",
#                 messages=[
#                     {
#                         "role": "system", 
#                         "content": "You are an expert in answering questions about database content."
#                     },
#                     {
#                         "role": "user", 
#                         "content": prompt
#                     }
#                 ]
#             )
            
#             answer = response['message']['content']
            
#             # Display answer
#             print("Answer:")
#             print(answer)
            
#             # Show sources
#             print("\nSources:")
#             for i, doc in enumerate(docs[:3]):
#                 print(f"Source {i+1} (from table '{doc.metadata.get('table')}'):")
#                 print(f"  {doc.page_content[:150]}...")
                
#         except Exception as e:
#             print(f"Error: {e}")
from sqlalchemy import create_engine, inspect
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from typing import Dict, Tuple, List, Any


def vectorize_entire_database(
    db_url: str,
    embedding_model: str = 'all-MiniLM-L6-v2',
    max_rows: int = None,
    store_vectors: bool = False,
    output_path: str = 'vector_output.npz'
):
    """
    Vectorizes text data across all tables in a database.

    Args:
        db_url (str): SQLAlchemy-compatible database URL.
        embedding_model (str): SentenceTransformer model name.
        max_rows (int): Optional limit on rows per table.
        store_vectors (bool): Whether to save embeddings to disk.
        output_path (str): File path for saving embeddings if store_vectors is True.

    Returns:
        dict: Table-wise {table_name: (DataFrame, embeddings)} dictionary.
    """
    engine = create_engine(db_url)
    inspector = inspect(engine)
    model = SentenceTransformer(embedding_model)
    results = {}

    with engine.connect() as conn:
        tables = inspector.get_table_names()

        for table in tables:
            try:
                print(f"Processing table: {table}")
                query = f"SELECT * FROM {table}"
                if max_rows:
                    query += f" LIMIT {max_rows}"
                df = pd.read_sql(query, conn)

                # Auto-detect text columns
                text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
                if not text_columns:
                    print(f"Skipping table '{table}' (no text columns)")
                    continue

                # Combine text columns into one
                df['__combined_text__'] = df[text_columns].astype(str).agg(' '.join, axis=1)

                # Generate embeddings
                embeddings = model.encode(df['__combined_text__'].tolist(), show_progress_bar=True)

                results[table] = (df.drop(columns=['__combined_text__']), embeddings)

            except Exception as e:
                print(f"Error processing table '{table}': {e}")
                continue

    # Optional: Save embeddings to disk
    if store_vectors:
        np.savez_compressed(output_path, **{
            f"{table}_embeddings": embeddings
            for table, (_, embeddings) in results.items()
        })

    return results


def create_vector_store_from_db(
    db_url: str,
    embedding_model_name: str = 'all-MiniLM-L6-v2'
) -> FAISS:
    """
    Create a FAISS vector store from database text data.
    
    Args:
        db_url (str): SQLAlchemy-compatible database URL
        embedding_model_name (str): Name of the HuggingFace embedding model
        
    Returns:
        FAISS: Vector store for semantic search
    """
    # Setup embedding function for LangChain
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Connect to database and extract text data
    engine = create_engine(db_url)
    inspector = inspect(engine)
    all_docs = []
    
    with engine.connect() as conn:
        tables = inspector.get_table_names()
        
        for table in tables:
            try:
                print(f"Processing table: {table}")
                df = pd.read_sql(f"SELECT * FROM {table}", conn)
                
                # Auto-detect text columns
                text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
                if not text_columns:
                    print(f"Skipping table '{table}' (no text columns)")
                    continue
                
                # Format each row as a document
                for _, row in df.iterrows():
                    # Create metadata to track source
                    metadata = {
                        "table": table,
                        "id": str(row.get("id", "unknown"))
                    }
                    
                    # Create document text including column names
                    text_parts = []
                    for col in text_columns:
                        if pd.notna(row[col]) and row[col]:
                            text_parts.append(f"{col}: {row[col]}")
                    
                    doc_text = "\n".join(text_parts)
                    if doc_text.strip():
                        all_docs.append({"content": doc_text, "metadata": metadata})
            
            except Exception as e:
                print(f"Error processing table '{table}': {e}")
                continue
    
    print(f"Collected {len(all_docs)} documents from database")
    
    # Split documents if they're too long
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    # Create documents for vector store
    documents = []
    for doc in all_docs:
        chunks = text_splitter.create_documents(
            texts=[doc["content"]], 
            metadatas=[doc["metadata"]]
        )
        documents.extend(chunks)
    
    print(f"Created {len(documents)} chunks after splitting")
    
    # Create and return the vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index_store")
    return vector_store


def chat_with_database(db_url: str, query: str = None):
    """
    Process a database query and return a formatted response for WhatsApp.
    
    Args:
        db_url (str): Database connection URL
        query (str, optional): The user's query. If not provided, a generic response is returned.
        
    Returns:
        str: Formatted response text for WhatsApp
    """
    import logging
    import os
    from django.conf import settings
    
    try:
        if not db_url:
            return "Database connection URL is not configured. Please set the DB_URL environment variable."
            
        # Log the query for debugging
        if query:
            logging.info(f"Processing query: {query}")
        else:
            logging.info("No query provided, returning generic response")
            return "Hello! I'm your inventory assistant. Ask me anything about our products, stock, or prices."
        
        # Check if vector store exists, if not create it
        BASE_DIR = settings.BASE_DIR
        vector_store_path = os.path.join(BASE_DIR, 'faiss_index_store')
        
        # If running for the first time, we'll need to create and save the vector store
        if not os.path.exists(vector_store_path):
            try:
                logging.info("Vector store not found. Attempting to create one...")
                # This could take some time for large databases
                vector_store = create_vector_store_from_db(db_url)
                logging.info("Vector store created successfully")
            except Exception as e:
                logging.error(f"Failed to create vector store: {str(e)}")
                return "I'm setting up my database connection. Please try again in a few minutes."
        else:
            try:
                # Load existing vector store
                logging.info("Loading existing vector store...")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.load_local(vector_store_path, embeddings,allow_dangerous_deserialization=True)
                logging.info("Vector store loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load vector store: {str(e)}")
                return "I'm having trouble accessing my database. Please try again later."
        
        # Process the query if we have a valid vector store
        if query and vector_store:
            try:
                # Create retriever from vector store
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}  # Return top 5 most relevant chunks
                )
                
                # Get relevant documents
                docs = retriever.get_relevant_documents(query)
                
                if not docs:
                    return "I couldn't find any information related to your question in our database."
                
                # Format context from relevant documents
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Build prompt for LLM
                prompt = f"""
                You are a friendly and knowledgeable ecommerce assistant trained to help customers with product and sales-related questions.

                Your objectives:
                1. Help the customer find products using only the provided context.
                2. Suggest similar or related items based on what's available in the context.
                3. Recommend relevant upsells or popular complementary products.
                4. DO NOT reveal or refer to any customer data, personal history, or private information—even if it exists in the database.

                Rules:
                - Use ONLY the context to answer.
                - If the answer is not in the context, reply:  
                "I don't have enough information to answer that."
                - Responses must be clear, concise, and written in a friendly, helpful tone.
                - Format the reply for WhatsApp:  
                Short sentences, bullet points (if needed), and easy to read on a mobile device.

                IMPORTANT FORMATTING INSTRUCTIONS:
                - When listing multiple products or categories, format them in a user-friendly numbered list
                - Extract only the product names/categories and present them neatly
                - Example format: "We have 3 types of cups: 1. Measuring Cups (KSh 450), 2. Disposable Cups (KSh 120), 3. Coffee Mugs (KSh 350)"
                - Include price information when available, using the format: "Product Name (KSh Price)"
                - Do NOT include raw data like paths, slugs, or metadata in your response
                - When a user asks about how many of a product type you have, count the unique categories and list them by name only
                - When a user asks about prices, clearly state the price next to each product name

                Think like a helpful sales rep: be polite, warm, and offer useful suggestions without overloading the customer.

                Context:  
                {context}

                Question:  
                {query}

                Answer:

                """
                
                # Check if Ollama is available
                try:
                    from langchain_ollama.llms import OllamaLLM
                    llm = OllamaLLM(model="llama2")
                    response = llm.invoke(prompt)
                    logging.info("Used Ollama LLM for response generation")
                except:
                    # If Ollama isn't available, provide a user-friendly response
                    logging.info("Ollama not available, providing user-friendly response")
                    response = f"I found some related products for you:\n\n"
                    
                    # Extract product names and prices
                    products = []
                    for doc in docs:
                        content = doc.page_content
                        product_info = {"name": "", "price": ""}
                        
                        # Try to extract name and price from content
                        for line in content.split('\n'):
                            line_lower = line.lower()
                            if line_lower.startswith('name:'):
                                product_info["name"] = line.split(':', 1)[1].strip()
                            elif any(price_field in line_lower for price_field in ['price:', 'regular_price:', 'sale_price:', 'cost:']):
                                try:
                                    price_parts = line.split(':', 1)
                                    if len(price_parts) > 1:
                                        price_value = price_parts[1].strip()
                                        # Clean up price value
                                        if price_value and price_value not in ["None", "null"]:
                                            product_info["price"] = price_value
                                except:
                                    pass
                        
                        # Add to products list if we have a name
                        if product_info["name"] and not any(p["name"] == product_info["name"] for p in products):
                            products.append(product_info)
                    
                    # Format as a numbered list with prices
                    if products:
                        response = f"We have {len(products)} types of products that match your query:\n\n"
                        for i, product in enumerate(products, 1):
                            if product["price"]:
                                response += f"{i}. {product['name']} (KSh {product['price']})\n"
                            else:
                                response += f"{i}. {product['name']}\n"
                        response += "\nHow can I help you with these products today?"
                    else:
                        # Fallback if we couldn't extract product names
                        response = "I found some products that might interest you, but I'm having trouble providing specific details. Could you please ask in a different way?"
                
                return response
                
            except Exception as e:
                logging.error(f"Error processing query with vector store: {str(e)}")
                return "I had trouble processing your question. Could you try asking in a different way?"
        
        # Fallback response
        return "Thank you for your query. I'm still learning about our inventory. Please try asking a specific question about our products or stock."
        
    except Exception as e:
        logging.error(f"Error in chat_with_database: {str(e)}", exc_info=True)
        return "Sorry, I encountered an error while processing your request."


# if __name__ == "__main__":
#     DB_URL = "postgresql://postgres:Muasya254;@localhost:5432/shop2shop"
#     chat_with_database(DB_URL)
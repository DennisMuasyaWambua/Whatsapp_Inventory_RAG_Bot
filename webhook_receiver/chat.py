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
#     embedding_model_name: str = 'paraphrase-MiniLM-L3-v2'
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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_limiter import MemoryLimiter, memory_monitor


def vectorize_entire_database(
    db_url: str,
    embedding_model: str = 'paraphrase-MiniLM-L3-v2',
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


@memory_monitor(max_memory_mb=800)
def create_vector_store_from_db(
    db_url: str,
    embedding_model_name: str = 'paraphrase-MiniLM-L3-v2'
) -> FAISS:
    """
    Create a FAISS vector store from database text data with memory optimization.
    
    Args:
        db_url (str): SQLAlchemy-compatible database URL
        embedding_model_name (str): Name of the HuggingFace embedding model
        
    Returns:
        FAISS: Vector store for semantic search
    """
    import gc
    import torch
    import psutil
    import os
    
    # Setup embedding function for LangChain
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},  # Force CPU for consistency
        encode_kwargs={
            'truncation': True,
            'padding': True,
            'max_length': 256,  # Limit token length for speed
            'batch_size': 1     # Small batch size for memory efficiency
        }
    )
    
    # Connect to database and extract text data
    engine = create_engine(db_url)
    inspector = inspect(engine)
    all_docs = []
    
    # Memory monitoring
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    with engine.connect() as conn:
        tables = inspector.get_table_names()
        
        for table_idx, table in enumerate(tables):
            try:
                print(f"Processing table {table_idx+1}/{len(tables)}: {table}")
                
                # Process table in smaller chunks to prevent memory overflow
                chunk_size = 1000  # Process 1000 rows at a time
                offset = 0
                
                # First, get total row count for the table
                count_query = f"SELECT COUNT(*) as total_rows FROM {table}"
                total_rows = pd.read_sql(count_query, conn).iloc[0]['total_rows']
                
                # Calculate 0.1% of the total rows, minimum 1, maximum 100
                rows_to_process = max(1, min(100, int(total_rows * 0.001)))
                print(f"Table {table} has {total_rows} rows, processing {rows_to_process} rows (0.1% max 100)")
                
                while True:
                    # Get chunk of data, but limit to 1% of total rows
                    remaining_rows = rows_to_process - offset
                    if remaining_rows <= 0:
                        break
                    
                    current_chunk_size = min(chunk_size, remaining_rows)
                    chunk_query = f"SELECT * FROM {table} LIMIT {current_chunk_size} OFFSET {offset}"
                    df_chunk = pd.read_sql(chunk_query, conn)
                    
                    if df_chunk.empty:
                        break
                    
                    # Auto-detect text columns
                    text_columns = df_chunk.select_dtypes(include=['object', 'string']).columns.tolist()
                    if not text_columns:
                        print(f"Skipping table '{table}' (no text columns)")
                        break
                    
                    # Format each row as a document
                    for _, row in df_chunk.iterrows():
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
                    
                    # Clear chunk from memory
                    del df_chunk
                    gc.collect()
                    
                    # Check memory usage
                    current_memory = process.memory_info().rss / 1024 / 1024
                    if current_memory > initial_memory + 300:  # If we've used more than 300MB extra
                        print(f"Memory usage high ({current_memory:.2f} MB), processing documents in smaller batches")
                        break
                    
                    offset += chunk_size
                    
                    # Limit total documents to prevent memory overflow
                    if len(all_docs) > 1000:  # Limit to 1k documents
                        print(f"Reached document limit ({len(all_docs)}), stopping to prevent memory issues")
                        break
                
                if len(all_docs) > 1000:
                    break
            
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
    
    # Create vector store with ultra-small batches to prevent OOM
    batch_size = 1  # Reduced to 1 to prevent memory overflow
    vector_store = None
    
    import numpy as np
    from langchain_community.docstore.in_memory import InMemoryDocstore
    
    print(f"Computing embeddings for {len(documents)} documents in micro-batches of {batch_size}")
    
    all_embeddings = []
    all_docs = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(documents) + batch_size - 1)//batch_size
        
        # Monitor memory before each batch
        current_memory = process.memory_info().rss / 1024 / 1024
        print(f"Batch {batch_num}/{total_batches} - Memory: {current_memory:.2f} MB")
        
        # Skip batch if memory is too high
        if current_memory > initial_memory + 500:  # 500MB limit
            print(f"Memory limit reached ({current_memory:.2f} MB), stopping vectorization")
            break
        
        try:
            # Extract texts from batch
            batch_texts = [doc.page_content for doc in batch]
            
            # Compute embeddings for this batch
            batch_embeddings = embeddings.embed_documents(batch_texts)
            
            # Store embeddings and documents
            all_embeddings.extend(batch_embeddings)
            all_docs.extend(batch)
            
            # Aggressive memory cleanup
            del batch_texts, batch_embeddings, batch
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection after each batch
            gc.collect()
            
            # Additional cleanup every 10 batches
            if batch_num % 10 == 0:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux memory trim
                print(f"Deep memory cleanup after batch {batch_num}")
            
        except Exception as e:
            print(f"Error computing embeddings for batch {batch_num}: {e}")
            # Continue with smaller batches on error
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"Reducing batch size to {batch_size} and retrying")
                continue
            else:
                raise e
    
    print(f"Creating FAISS index from {len(all_embeddings)} pre-computed embeddings")
    
    # Create FAISS index from pre-computed embeddings
    try:
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # Create FAISS index
        import faiss
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        
        # Create docstore
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(all_docs)})
        index_to_docstore_id = {i: str(i) for i in range(len(all_docs))}
        
        # Create FAISS vector store
        from langchain_community.vectorstores.faiss import FAISS
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        # Clear large arrays
        del all_embeddings, embeddings_array, all_docs
        gc.collect()
        
        print("FAISS vector store created successfully")
        
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        raise e
    
    if vector_store is None:
        raise ValueError("No documents to vectorize")
    
    vector_store.save_local("faiss_index_store")
    return vector_store


def process_order(products, order_text):
    """
    Extract order details from user message and generate order confirmation.
    
    Args:
        products (list): List of available products with names and prices
        order_text (str): The user's order message
        
    Returns:
        str: Order confirmation or follow-up questions
    """
    import re
    
    # Look for product mentions with quantities
    ordered_items = []
    total_price = 0
    
    # Extract potential product mentions (basic implementation)
    for product in products:
        product_name = product["name"].lower()
        if product_name in order_text.lower():
            # Look for quantity pattern (e.g., "2 cups" or "cups x2" or "cups (2)")
            quantity_patterns = [
                rf"(\d+)\s+{re.escape(product_name)}",  # "2 cups"
                rf"{re.escape(product_name)}\s+x\s*(\d+)",  # "cups x2"
                rf"{re.escape(product_name)}\s*\((\d+)\)",  # "cups (2)"
            ]
            
            quantity = 1  # Default quantity
            for pattern in quantity_patterns:
                matches = re.search(pattern, order_text.lower())
                if matches:
                    quantity = int(matches.group(1))
                    break
            
            # Calculate price if available
            item_price = 0
            if product["price"]:
                try:
                    item_price = float(product["price"]) * quantity
                    total_price += item_price
                except ValueError:
                    # If price is not a valid number
                    pass
            
            ordered_items.append({
                "name": product["name"],
                "quantity": quantity,
                "price": product["price"],
                "total": item_price
            })
    
    # Check if we have shipping address
    address_match = re.search(r"(deliver|ship|send).+to\s+(.+?)(?:\.|\n|$)", order_text, re.IGNORECASE)
    delivery_address = address_match.group(2).strip() if address_match else None
    
    # Check for payment method
    payment_methods = ["m-pesa", "mpesa", "cash on delivery", "cod", "bank transfer"]
    payment_method = None
    for method in payment_methods:
        if method in order_text.lower():
            payment_method = method
            break
    
    # Generate response based on extracted information
    if not ordered_items:
        return "I'd be happy to help you place an order! Could you please specify which products you'd like to purchase and the quantity of each?"
    
    response = "Here's what I understand from your order:\n\n"
    for item in ordered_items:
        if item["price"]:
            response += f"• {item['quantity']}x {item['name']} - KSh {item['price']} each (Total: KSh {item['total']})\n"
        else:
            response += f"• {item['quantity']}x {item['name']} - Price to be confirmed\n"
    
    response += f"\nOrder Total: KSh {total_price:.2f}"
    
    # Check if we need more information
    missing_info = []
    if not delivery_address:
        missing_info.append("delivery address")
    if not payment_method:
        missing_info.append("preferred payment method (M-Pesa, Cash on Delivery, or Bank Transfer)")
    
    if missing_info:
        response += f"\n\nTo complete your order, I'll need your {' and '.join(missing_info)}."
    else:
        response += f"\nDelivery to: {delivery_address}\nPayment method: {payment_method.upper()}\n\nThank you for your order! A shop representative will contact you shortly to confirm and finalize your purchase."
    
    return response

@memory_monitor(max_memory_mb=600)
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
                embeddings = HuggingFaceEmbeddings(
                    model_name="paraphrase-MiniLM-L3-v2",
                    model_kwargs={'device': 'cpu'},  # Force CPU for consistency
                    encode_kwargs={
                        'truncation': True,
                        'padding': True,
                        'max_length': 256,  # Limit token length for speed
                        'batch_size': 1     # Small batch size for memory efficiency
                    }
                )
                vector_store = FAISS.load_local(vector_store_path, embeddings,allow_dangerous_deserialization=True)
                logging.info("Vector store loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load vector store: {str(e)}")
                return "I'm having trouble accessing my database. Please try again later."
        
        # Process the query if we have a valid vector store
        if query and vector_store:
            try:
                # Create retriever from vector store with optimized search parameters for speed
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 5,   # Reduced to 5 for faster processing
                        "fetch_k": 10  # Reduced candidate pool for speed
                    }
                )
                
                # Get relevant documents
                docs = retriever.invoke(query)
                
                if not docs:
                    return "I couldn't find any information related to your question in our database. Could you try asking about specific products, prices, or stock levels?"
                
                # Format context from relevant documents
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Debug: Log the retrieved context
                logging.info(f"Retrieved context length: {len(context)} characters")
                logging.info(f"Number of documents retrieved: {len(docs)}")
                logging.info(f"First 200 chars of context: {context[:200]}")
                
                # Build prompt for LLM
                prompt = f"""
You are a professional and friendly customer support assistant for an inventory/e-commerce system.

Instructions:
1. Use ONLY the provided context to answer questions
2. If the context doesn't contain the answer, say "I don't have that specific information available"
3. Be specific and helpful - include product names, prices, and quantities when available
4. Format responses clearly for WhatsApp (short paragraphs, bullet points when helpful)
5. If multiple products match, list them with their details

Context from database:
{context}

Customer question: {query}

Response:"""
                
                # Use lighter Llama models for response generation or fallback to structured response
                try:
                    # Configure Ollama for Railway deployment
                    import os
                    ollama_host = os.getenv('OLLAMA_HOST', 'localhost:11434')
                    ollama_base_url = f"http://{ollama_host}"
                    
                    # Prioritize fastest models first: 1B model for speed
                    models_to_try = ["llama3.2:1b", "llama3.2:1b-instruct"]  # Try both variants
                    response = None
                    
                    for model_name in models_to_try:
                        try:
                            llm = OllamaLLM(
                                model=model_name,
                                base_url=ollama_base_url,  # Use Railway's Ollama instance
                                temperature=0.3,  # Lower temperature for faster, more focused responses
                                top_k=10,         # Limit vocabulary for speed
                                top_p=0.8,        # Focus on most likely tokens
                                num_predict=200,  # Limit response length for speed
                                repeat_penalty=1.1,
                                timeout=30        # Add timeout for Railway deployment
                            )
                            response = llm.invoke(prompt)
                            logging.info(f"Used Ollama {model_name} for response generation at {ollama_base_url}")
                            break
                        except Exception as model_error:
                            logging.warning(f"Model {model_name} failed at {ollama_base_url}: {str(model_error)}")
                            continue
                    
                    if not response:
                        raise Exception("Lightweight Llama model failed")
                except Exception as e:
                    # If Ollama isn't available, provide a better structured response
                    logging.info(f"Ollama not available ({str(e)}), providing structured response from retrieved context")
                    
                    # Extract product information from retrieved documents
                    products = []
                    all_info = []
                    
                    for doc in docs:
                        content = doc.page_content
                        product_info = {"name": "", "price": "", "description": "", "stock": "", "other_details": []}
                        
                        # Parse content line by line to extract structured information
                        lines = content.split('\n')
                        for line in lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                key = key.strip().lower()
                                value = value.strip()
                                
                                if not value or value.lower() in ['none', 'null', '']:
                                    continue
                                    
                                if key in ['name', 'product_name', 'title']:
                                    product_info["name"] = value
                                elif key in ['price', 'regular_price', 'sale_price', 'cost']:
                                    product_info["price"] = value
                                elif key in ['description', 'desc', 'product_description']:
                                    product_info["description"] = value
                                elif key in ['stock', 'quantity', 'stock_quantity', 'inventory']:
                                    product_info["stock"] = value
                                else:
                                    # Collect other relevant details
                                    product_info["other_details"].append(f"{key.title()}: {value}")
                        
                        # Add to products list if we have meaningful information
                        if product_info["name"] or any([product_info["price"], product_info["description"], product_info["stock"]]):
                            # Avoid duplicates
                            if not any(p.get("name") == product_info["name"] for p in products if p.get("name")):
                                products.append(product_info)
                        
                        # Also collect all content for general information
                        if content.strip():
                            all_info.append(content.strip())
                    
                    # Generate response based on extracted information
                    if products:
                        # Check if this is an order request
                        order_keywords = ["order", "buy", "purchase", "checkout", "get", "want", "deliver", "cart"]
                        is_order_request = any(keyword in query.lower() for keyword in order_keywords)
                        
                        if is_order_request:
                            # Process as an order request
                            response = process_order(products, query)
                        else:
                            # Format products information
                            if len(products) == 1:
                                p = products[0]
                                response = f"Here's information about {p['name'] or 'this product'}:\n\n"
                                if p["price"]:
                                    response += f"💰 Price: KSh {p['price']}\n"
                                if p["stock"]:
                                    response += f"📦 Stock: {p['stock']}\n"
                                if p["description"]:
                                    response += f"📝 {p['description']}\n"
                                if p["other_details"]:
                                    response += "\n" + "\n".join(p["other_details"][:3])  # Limit to 3 details
                                response += "\n\nWould you like to order this product or need more information?"
                            else:
                                response = f"Found {len(products)} products matching your query:\n\n"
                                for i, p in enumerate(products[:5], 1):  # Limit to 5 products
                                    name = p["name"] or f"Product {i}"
                                    price_info = f" - KSh {p['price']}" if p["price"] else ""
                                    stock_info = f" (Stock: {p['stock']})" if p["stock"] else ""
                                    response += f"{i}. {name}{price_info}{stock_info}\n"
                                
                                if len(products) > 5:
                                    response += f"\n...and {len(products) - 5} more products.\n"
                                response += "\nAsk about a specific product for more details!"
                    else:
                        # No structured products found, but we have content
                        if all_info:
                            # Provide the most relevant information from context
                            combined_info = "\n".join(all_info[:2])  # Use first 2 chunks
                            if len(combined_info) > 300:
                                combined_info = combined_info[:297] + "..."
                            response = f"Here's what I found:\n\n{combined_info}\n\nNeed more specific information? Just ask!"
                        else:
                            response = "I found some database records related to your query, but couldn't extract specific details. Could you be more specific about what you're looking for?"
                
                return response
                
            except Exception as e:
                logging.error(f"Error processing query with vector store: {str(e)}")
                return "I had trouble processing your question. Could you try asking in a different way?"
        
        # More specific fallback response
        return "I'm ready to help you with product inquiries! Try asking me about specific products, prices, stock levels, or placing an order."
        
    except Exception as e:
        logging.error(f"Error in chat_with_database: {str(e)}", exc_info=True)
        return "Sorry, I encountered an error while processing your request."


# if __name__ == "__main__":
#     DB_URL = "postgresql://postgres:Muasya254;@localhost:5432/shop2shop"
#     chat_with_database(DB_URL)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs, language="en", chunk_size=800, chunk_overlap=300):
    """
    Split documents into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    Optimized for both English and Chinese (Simplified/Traditional).

    Args:
        docs (list): List of document dictionaries with 'content' and 'metadata'.
        language (str): 'en' or 'zh'.
        chunk_size (int): Target size of each chunk.
        chunk_overlap (int): Number of characters to overlap.

    Returns:
        list: List of chunk dictionaries with 'page_content' and 'metadata'.
    """
    chunks = []

    # Normalize language input
    target_language = str(language).lower().strip()

    # Define separators based on language
    # LangChain tries to split by the first separator, if the chunk is still too big,
    # it moves to the next separator, and so on.
    if target_language == "zh":
        separators = [
            "\n\n",  # Paragraphs (Strongest)
            "\n",  # Line breaks
            "。",  # Period
            "！",  # Exclamation mark
            "？",  # Question mark
            "；",  # Semicolon (Important for lists)
            "：",  # Colon (Important for structured data like "Name: Value")
            "、",  # Enumeration comma (Important for "1、First item")
            "，",  # Comma (Weakest sentence break)
            " ",  # Spaces (Rare in Chinese but possible)
            "",  # Character by character (Last resort)
        ]
    else:
        # Default to English standard separators
        separators = ["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]

    # Initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        keep_separator=True,  # Keep punctuation at the end of the sentence
        strip_whitespace=True,  # Clean up extra whitespace
    )

    for doc in docs:
        # Validate content
        text = doc.get("content")
        if not isinstance(text, str) or not text.strip():
            continue

        # Check document language matches target language
        # (Defaulting to 'en' if not specified in doc)
        doc_lang = doc.get("language", "en").lower().strip()
        if doc_lang != target_language:
            continue

        # Prepare metadata (remove content to avoid duplication in memory)
        base_metadata = doc.copy()
        base_metadata.pop("content", None)

        # Create chunks using LangChain
        # create_documents accepts list of texts and list of metadatas
        lc_docs = text_splitter.create_documents(
            texts=[text], metadatas=[base_metadata]
        )

        # Convert back to the project's standard dictionary format
        for i, lc_doc in enumerate(lc_docs):
            chunk_metadata = lc_doc.metadata.copy()
            # Add chunk index for reference
            chunk_metadata["chunk_index"] = i

            chunks.append(
                {"page_content": lc_doc.page_content, "metadata": chunk_metadata}
            )

    return chunks
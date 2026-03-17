def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150):
    """
    Splits text into overlapping chunks
    """

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks
import re

def split_text_semantically(text: str, max_words: int = 100) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        if not sentence.strip():
            continue
            
        words = sentence.split()
        if current_len + len(words) > max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = words
            current_len = len(words)
        else:
            current_chunk.extend(words)
            current_len += len(words)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    print(f"Article split into {len(chunks)} semantic chunks.")
    return chunks
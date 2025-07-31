import re 
from typing import List, Dict 

def remove_common_headers_footers(pages : List[str]) -> List[str]:
    """
    Removes repeated lines (headers and footers) that appears on most pages
    """
    line_counts = {}
    for page in pages:
        for line in page.splitlines():
            line_counts[line.strip()] = line_counts.get(line.strip(),0) + 1 
            
        
    # lines that apperas more than 60% of the pages - headers/footers
    threshold = int(len(pages) * 0.6)
    noisy_lines = {line for line, count in line_counts.items() if count > threshold}
    
    cleaned = []
    for page in pages: 
        lines = [
            line for line in page.splitlines()
            if line.strip() not in noisy_lines and not line.lower().startswith("page")
        ]
        cleaned.append("\n".join(lines))
        
    return cleaned 

def normalize_whitespace(text:str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()  # Remove leading and trailing whitespace

    
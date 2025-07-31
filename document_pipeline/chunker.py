import tiktoken
import re
import spacy
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to load spacy model, fallback to basic chunking if not available
try:
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except OSError:
    logger.warning("spaCy model not found. Using basic sentence splitting.")
    HAS_SPACY = False
    nlp = None

encoding = tiktoken.get_encoding("cl100k_base")

@dataclass
class ChunkMetadata:
    """Enhanced metadata for each chunk."""
    semantic_score: float = 0.0
    entity_count: int = 0
    sentence_count: int = 0
    keywords: List[str] = None
    section_type: str = "body"
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

class EnhancedChunker:
    """Enhanced chunking with semantic awareness and improved context preservation."""
    
    def __init__(self, chunk_size: int = 600, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.sentence_enders = re.compile(r'[.!?]+\s+')
        self.section_headers = re.compile(r'^(#{1,6}\s+|[A-Z][A-Z\s]+:|\d+\.\s+[A-Z])')
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(encoding.encode(text))
    
    def enhanced_recursive_split(self, text: str, document_metadata: Dict = None) -> List[Dict]:
        """
        Enhanced chunking with semantic awareness and improved overlap strategy.
        """
        if not text.strip():
            return []
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Detect document structure
        sections = self._detect_sections(text)
        
        all_chunks = []
        global_position = 0
        
        for section_type, section_text in sections:
            section_chunks = self._chunk_section(
                section_text, 
                section_type, 
                global_position
            )
            all_chunks.extend(section_chunks)
            global_position += len(section_text)
        
        # Apply cross-section overlap for better context
        all_chunks = self._apply_cross_section_overlap(all_chunks)
        
        return all_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better chunking."""
        # Normalize whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common formatting issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words
        text = re.sub(r'(\w)([.!?])([A-Z])', r'\1\2 \3', text)  # Space after sentence
        
        return text.strip()
    
    def _detect_sections(self, text: str) -> List[Tuple[str, str]]:
        """Detect different sections in the document."""
        sections = []
        lines = text.split('\n')
        current_section = []
        current_type = "body"
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_section:
                    current_section.append('')
                continue
            
            # Detect section headers
            new_type = self._classify_line(line)
            
            if new_type != current_type and current_section:
                # Save current section
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    sections.append((current_type, section_text))
                current_section = []
            
            current_type = new_type
            current_section.append(line)
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append((current_type, section_text))
        
        return sections if sections else [("body", text)]
    
    def _classify_line(self, line: str) -> str:
        """Classify a line into section types."""
        line_upper = line.upper()
        
        # Headers and titles
        if self.section_headers.match(line):
            return "header"
        
        # Common insurance/legal section patterns
        if any(keyword in line_upper for keyword in [
            'DEFINITIONS', 'COVERAGE', 'EXCLUSIONS', 'CONDITIONS', 
            'BENEFITS', 'LIMITATIONS', 'CLAIMS', 'PREMIUM'
        ]):
            return "important"
        
        # Lists and numbered items
        if re.match(r'^\s*[\da-z]\.\s+|^\s*[-•]\s+', line):
            return "list"
        
        # Tables (simple detection)
        if '|' in line or '\t' in line:
            return "table"
        
        return "body"
    
    def _chunk_section(self, text: str, section_type: str, global_position: int) -> List[Dict]:
        """Chunk a specific section with type-aware strategies."""
        if section_type == "table":
            return self._chunk_table(text, global_position)
        elif section_type == "list":
            return self._chunk_list(text, global_position)
        else:
            return self._chunk_text_semantically(text, section_type, global_position)
    
    def _chunk_table(self, text: str, global_position: int) -> List[Dict]:
        """Special handling for table content."""
        lines = text.split('\n')
        table_chunks = []
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            
            if current_tokens + line_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                table_chunks.append(self._create_chunk_dict(
                    chunk_text, len(table_chunks), global_position, "table"
                ))
                
                # Start new chunk with header if available
                if table_chunks and '|' in current_chunk[0]:  # Keep header
                    current_chunk = [current_chunk[0], line]
                    current_tokens = self.count_tokens(current_chunk[0]) + line_tokens
                else:
                    current_chunk = [line]
                    current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            table_chunks.append(self._create_chunk_dict(
                chunk_text, len(table_chunks), global_position, "table"
            ))
        
        return table_chunks
    
    def _chunk_list(self, text: str, global_position: int) -> List[Dict]:
        """Special handling for list content."""
        items = re.split(r'\n(?=\s*[\da-z]\.\s+|^\s*[-•]\s+)', text)
        list_chunks = []
        current_chunk = []
        current_tokens = 0
        
        for item in items:
            item_tokens = self.count_tokens(item)
            
            if current_tokens + item_tokens > self.chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                list_chunks.append(self._create_chunk_dict(
                    chunk_text, len(list_chunks), global_position, "list"
                ))
                current_chunk = [item]
                current_tokens = item_tokens
            else:
                current_chunk.append(item)
                current_tokens += item_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            list_chunks.append(self._create_chunk_dict(
                chunk_text, len(list_chunks), global_position, "list"
            ))
        
        return list_chunks
    
    def _chunk_text_semantically(self, text: str, section_type: str, global_position: int) -> List[Dict]:
        """Chunk text with semantic awareness."""
        if HAS_SPACY and len(text) < 1000000:  # spaCy has limits
            return self._semantic_chunking(text, section_type, global_position)
        else:
            return self._sentence_aware_chunking(text, section_type, global_position)
    
    def _semantic_chunking(self, text: str, section_type: str, global_position: int) -> List[Dict]:
        """Use spaCy for semantic chunking."""
        doc = nlp(text)
        sentences = list(doc.sents)
        
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        
        for sent in sentences:
            sent_text = sent.text.strip()
            sent_tokens = self.count_tokens(sent_text)
            
            if current_tokens + sent_tokens > self.chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk_text = ' '.join(s.text for s in current_chunk_sentences)
                chunk_metadata = self._extract_semantic_metadata(current_chunk_sentences)
                
                chunk_dict = self._create_chunk_dict(
                    chunk_text, len(chunks), global_position, section_type, chunk_metadata
                )
                chunks.append(chunk_dict)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, self.overlap
                )
                current_chunk_sentences = overlap_sentences + [sent]
                current_tokens = sum(self.count_tokens(s.text) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sent)
                current_tokens += sent_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(s.text for s in current_chunk_sentences)
            chunk_metadata = self._extract_semantic_metadata(current_chunk_sentences)
            
            chunk_dict = self._create_chunk_dict(
                chunk_text, len(chunks), global_position, section_type, chunk_metadata
            )
            chunks.append(chunk_dict)
        
        return chunks
    
    def _sentence_aware_chunking(self, text: str, section_type: str, global_position: int) -> List[Dict]:
        """Fallback to sentence-aware chunking without spaCy."""
        sentences = self.sentence_enders.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(self._create_chunk_dict(
                    chunk_text, len(chunks), global_position, section_type
                ))
                
                # Add overlap
                overlap_size = min(len(current_chunk), 2)  # Keep last 2 sentences
                current_chunk = current_chunk[-overlap_size:] + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(self._create_chunk_dict(
                chunk_text, len(chunks), global_position, section_type
            ))
        
        return chunks
    
    def _extract_semantic_metadata(self, sentences) -> ChunkMetadata:
        """Extract semantic metadata from spaCy sentences."""
        if not HAS_SPACY:
            return ChunkMetadata()
        
        entities = []
        keywords = []
        
        for sent in sentences:
            # Extract named entities
            entities.extend([ent.text for ent in sent.ents])
            
            # Extract keywords (nouns, proper nouns, adjectives)
            for token in sent:
                if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
        
        # Calculate semantic score based on entity density and keyword richness
        total_words = sum(len(sent.text.split()) for sent in sentences)
        entity_density = len(entities) / max(total_words, 1)
        keyword_diversity = len(set(keywords)) / max(len(keywords), 1)
        semantic_score = (entity_density + keyword_diversity) / 2
        
        return ChunkMetadata(
            semantic_score=semantic_score,
            entity_count=len(entities),
            sentence_count=len(sentences),
            keywords=list(set(keywords))
        )
    
    def _get_overlap_sentences(self, sentences, overlap_tokens: int):
        """Get sentences for overlap based on token count."""
        if not sentences:
            return []
        
        overlap_sentences = []
        current_tokens = 0
        
        for sent in reversed(sentences):
            sent_tokens = self.count_tokens(sent.text)
            if current_tokens + sent_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sent)
                current_tokens += sent_tokens
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk_dict(self, text: str, chunk_index: int, global_position: int, 
                          section_type: str, metadata: ChunkMetadata = None) -> Dict:
        """Create a standardized chunk dictionary."""
        if metadata is None:
            metadata = ChunkMetadata()
        
        return {
            "chunk_id": f"chunk_{chunk_index:04d}",
            "text": text,
            "token_count": self.count_tokens(text),
            "char_range": (global_position, global_position + len(text)),
            "section_type": section_type,
            "semantic_score": metadata.semantic_score,
            "entity_count": metadata.entity_count,
            "sentence_count": metadata.sentence_count,
            "keywords": metadata.keywords
        }
    
    def _apply_cross_section_overlap(self, chunks: List[Dict]) -> List[Dict]:
        """Apply overlap between sections for better context."""
        if len(chunks) <= 1:
            return chunks
        
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk.copy()
            
            # Add context from previous chunk if different section
            if (i > 0 and 
                chunks[i-1].get('section_type') != chunk.get('section_type')):
                prev_text = chunks[i-1]['text']
                # Add last sentence from previous chunk
                sentences = self.sentence_enders.split(prev_text)
                if sentences and len(sentences) > 1:
                    context = sentences[-1].strip()
                    if context:
                        enhanced_chunk['text'] = f"[Previous context: {context}] {chunk['text']}"
                        enhanced_chunk['token_count'] = self.count_tokens(enhanced_chunk['text'])
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks

# Maintain backward compatibility
def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def recursive_split(text: str, chunk_size: int = 600, overlap: int = 200) -> List[Dict]:
    """
    Backward compatible function with enhanced chunking.
    """
    chunker = EnhancedChunker(chunk_size, overlap)
    return chunker.enhanced_recursive_split(text)
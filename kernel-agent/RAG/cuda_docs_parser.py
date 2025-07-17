"""Crawls complete CUDA documentation, parses and cleans content by subject section. (Includes code examples, tables, warnings in the documentation)"""
import json
import re
from bs4 import BeautifulSoup, NavigableString
from typing import Dict, List, Any, Optional
import requests

class CUDADocsParser:
    def __init__(self):
        self.documents = []
        
    def fetch_and_parse_url(self, url: str) -> List[Dict[str, Any]]:
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Ensure proper encoding detection
            if response.encoding is None or response.encoding == 'ISO-8859-1':
                response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
            return self.parse_document(soup)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return []
    
    def parse_document(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse the CUDA documentation into individual section documents"""
        self.documents = []
        
        main_content = (
            soup.find('div', {'role': 'main'}) or 
            soup.find('div', class_='document') or
            soup.find('div', {'itemprop': 'articleBody'}) or
            soup.find('div', class_='rst-content')
        )
        
        if not main_content:
            print("Could not find main content area")
            return []
        
        sections = main_content.find_all('section', recursive=True)
        
        for section in sections:
            document = self.parse_section_to_document(section)
            if document:
                self.documents.append(document)
        
        return self.documents
    
    def parse_section_to_document(self, section_element) -> Optional[Dict[str, Any]]:
        section_id = section_element.get('id', '')
        if not section_id:
            return None
        
        section_number, title = self.extract_section_info(section_element)
        
        text_content = self.extract_section_text(section_element)
        
        if not text_content.strip():
            return None
        
        return {
            "section_id": section_id,
            "section_number": section_number,
            "title": title,
            "text": text_content.strip()
        }
    
    def extract_section_info(self, section_element) -> tuple[str, str]:
        header_selectors = [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p.rubric-h1', 'p.rubric-h2', 'p.rubric-h3',
            'p.rubric', '.rubric'
        ]
        
        header_tag = None
        for selector in header_selectors:
            header_tag = section_element.find(selector)
            if header_tag:
                break
        
        if not header_tag:
            section_id = section_element.get('id', '')
            title = section_id.replace('-', ' ').title()
            return "", title
        
        header_text = self.extract_plain_text(header_tag).strip()
        
        section_number = ""
        title = header_text
        
        section_number_span = header_tag.find('span', class_='section-number')
        if section_number_span:
            section_number = section_number_span.get_text().strip()
            title = header_text.replace(section_number, '').strip()
        else:
            number_match = re.match(r'^(\d+(?:\.\d+)*)\.\s*(.*)$', header_text)
            if number_match:
                section_number = number_match.group(1)
                title = number_match.group(2)
        
        return section_number, title
    
    def extract_section_text(self, section_element) -> str:
        text_parts = []
        
        # excluding nested sections
        for child in section_element.children:
            if hasattr(child, 'name') and child.name:
                if child.name == 'section':
                    continue
                
                if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] or 'rubric' in child.get('class', []):
                    continue
                
                element_text = self.process_content_element(child)
                if element_text:
                    text_parts.append(element_text)
        
        return '\n\n'.join(text_parts)
    
    def process_content_element(self, element) -> str:
        """Process different types of content elements"""
        if not hasattr(element, 'name') or not element.name:
            return ""
        
        if element.name == 'p':
            if 'rubric' in element.get('class', []):
                return ""
            return self.extract_plain_text(element)
        
        elif element.name == 'div' and any('highlight' in cls for cls in element.get('class', [])):
            return self.extract_code_block(element)
        
        elif element.name == 'pre':
            code_content = element.get_text().strip()
            return f'"""{code_content}"""' if code_content else ""
        
        elif element.name == 'table' or (element.name == 'div' and 'table' in str(element.get('class', []))):
            return self.extract_table(element)
        
        elif element.name in ['ul', 'ol']:
            return self.extract_list(element)
        
        elif element.name == 'dl':
            return self.extract_definition_list(element)
        
        elif element.name == 'div' and any(cls in element.get('class', []) for cls in ['admonition', 'note', 'warning', 'tip', 'important']):
            return self.extract_admonition(element)
        
        elif element.name == 'blockquote':
            content = self.extract_plain_text(element)
            return f'"""{content}"""' if content else ""
        
        elif element.name == 'div':
            child_texts = []
            for child in element.children:
                if hasattr(child, 'name') and child.name and child.name != 'section':
                    child_text = self.process_content_element(child)
                    if child_text:
                        child_texts.append(child_text)
            return '\n'.join(child_texts)
        
        else:
            return self.extract_plain_text(element)
    
    def clean_unicode_text(self, text: str) -> str:
        """Clean up problematic Unicode characters and symbols"""
        if not text:
            return ""
        
        # Common problematic character replacements
        replacements = {
            '□': '',  # Unicode replacement character box (often from headerlinks)
            'ï': '',  # Often garbled link symbols
            'â': '',  # Often garbled warning symbols
            'ï¿½': '',  # Unicode replacement character
            'â€™': "'",  # Right single quotation mark
            'â€œ': '"',  # Left double quotation mark  
            'â€': '"',   # Right double quotation mark
            'â€"': '-',  # Em dash
            'â€"': '--', # En dash
            'â€¢': '•',  # Bullet point
            'Â': '',     # Often unwanted non-breaking space artifacts
            '\ufeff': '', # Byte order mark
            '\u200b': '', # Zero width space
            '\u200c': '', # Zero width non-joiner
            '\u200d': '', # Zero width joiner
            '\u2060': '', # Word joiner
            '¶': '',      # Paragraph/permalink symbol
            '#': '',      # Sometimes used in headerlinks
        }
        
        # Apply replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove other control characters and problematic symbols
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def clean_table_cell(self, text: str) -> str:
        if not text:
            return ""
        
        # First clean unicode issues
        text = self.clean_unicode_text(text)
        
        if '`' in text:
            parts = []
            current_part = ""
            in_code = False
            
            i = 0
            while i < len(text):
                if text[i] == '`':
                    if in_code:
                        if current_part and current_part not in parts:
                            parts.append(f"`{current_part}`")
                        current_part = ""
                        in_code = False
                    else:
                        if current_part.strip():
                            parts.append(current_part.strip())
                        current_part = ""
                        in_code = True
                else:
                    current_part += text[i]
                i += 1
            
            if current_part:
                if in_code:
                    parts.append(f"`{current_part}`")
                else:
                    parts.append(current_part.strip())
            
            cleaned_text = " ".join(parts)
        else:
            cleaned_text = text
        
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.replace('\\r\\n', ' ').replace('\\n', ' ').replace('\r', ' ').replace('\n', ' ')
        
        return cleaned_text.strip()
    
    def extract_plain_text(self, element) -> str:
        """Extract plain text, preserving inline code formatting with proper Unicode handling"""
        if not element:
            return ""
        
        text_parts = []
        
        # Process direct children instead of all descendants to avoid duplication
        for child in element.children:
            if isinstance(child, NavigableString):
                # Skip text that's inside headerlink anchors
                parent = child.parent
                if parent and hasattr(parent, 'get') and 'headerlink' in parent.get('class', []):
                    continue
                text_str = str(child)
                text_parts.append(text_str)
            elif hasattr(child, 'name'):
                if child.name == 'a' and 'headerlink' in child.get('class', []):
                    # Skip headerlink anchors entirely
                    continue
                elif child.name == 'code':
                    # Handle code elements specially to avoid duplication
                    code_text = self.extract_code_content(child)
                    text_parts.append(f"`{code_text}`")
                else:
                    # Recursively process other elements
                    child_text = self.extract_plain_text(child)
                    text_parts.append(child_text)
        
        if not text_parts:
            # Fallback: get text but remove headerlink content
            temp_element = element.__copy__()
            # Remove all headerlink elements
            for link in temp_element.find_all('a', class_='headerlink'):
                link.decompose()
            text_parts.append(temp_element.get_text())
        
        result = ''.join(text_parts)
        
        # Clean Unicode issues
        result = self.clean_unicode_text(result)
        
        return result
    
    def extract_code_content(self, code_element) -> str:
        """Extract content from code elements, handling nested spans properly"""
        if not code_element:
            return ""
        
        # For code elements with nested spans, concatenate the span contents
        spans = code_element.find_all('span', class_='pre')
        if spans:
            # Join all the span contents
            code_parts = []
            for span in spans:
                span_text = span.get_text()
                if span_text:
                    code_parts.append(span_text)
            return ''.join(code_parts)
        else:
            # No nested spans, just get the text
            return code_element.get_text()
    
    def extract_code_block(self, element) -> str:
        pre_element = element.find('pre')
        if not pre_element:
            return ""
        
        code_content = pre_element.get_text().strip()
        return f'"""{code_content}"""' if code_content else ""
    
    def extract_table(self, element) -> str:
        """Extract table as a list of dictionaries with proper headers

        Format [{"column_1_name": "row_1_column1_content", "column_2_name": "row_1_column2_content"},
        {"column_1_name": "row_2_column1_content", "column_2_name": "row_2_column2_content"}...]
        
        """
        if element.name != 'table':
            table = element.find('table')
        else:
            table = element
        
        if not table:
            return ""
        
        all_rows = table.find_all('tr')
        if not all_rows:
            return ""
        
        # table headers are first row
        headers = []
        first_row = all_rows[0]
        for cell in first_row.find_all(['th', 'td']):
            header_text = self.extract_plain_text(cell).strip()
            headers.append(header_text)
        
        if not headers:
            return ""
        
        data_rows = []
        for tr in all_rows[1:]:
            cells = tr.find_all(['td', 'th'])
            if not cells:
                continue
                
            row_data = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    cell_content = self.extract_plain_text(cell).strip()
                    cell_content = self.clean_table_cell(cell_content)
                    row_data[headers[i]] = cell_content
            
            if row_data:
                data_rows.append(row_data)
        
        if data_rows:
            return str(data_rows)
        return ""
    
    def extract_list(self, element) -> str:
        items = []
        for li in element.find_all('li', recursive=False):
            item_text = self.extract_plain_text(li)
            if item_text:
                items.append(f"- {item_text}")
        
        return '\n'.join(items)
    
    def extract_definition_list(self, element) -> str:
        items = []
        current_term = None
        
        for child in element.children:
            if hasattr(child, 'name'):
                if child.name == 'dt':
                    current_term = self.extract_plain_text(child)
                elif child.name == 'dd' and current_term:
                    definition = self.extract_plain_text(child)
                    items.append(f"{current_term}: {definition}")
                    current_term = None
        
        return '\n'.join(items)
    
    def extract_admonition(self, element) -> str:
        admonition_type = "note"
        classes = element.get('class', [])
        for cls in classes:
            if cls in ['note', 'warning', 'tip', 'important', 'caution']:
                admonition_type = cls
                break
        
        content_parts = []
        for p in element.find_all('p'):
            if 'admonition-title' not in p.get('class', []):
                text = self.extract_plain_text(p)
                if text:
                    content_parts.append(text)
        
        content = ' '.join(content_parts)
        return f'"""{admonition_type.upper()}: {content}"""' if content else ""
    
    def save_to_json(self, documents: List[Dict[str, Any]], filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

def parse_cuda_docs(url: str = "https://docs.nvidia.com/cuda/cuda-c-programming-guide/", 
                   output_file: str = "cuda_docs_parsed.json"):
    parser = CUDADocsParser()
    
    print(f"Fetching and parsing CUDA documentation from: {url}")
    documents = parser.fetch_and_parse_url(url)
    
    if documents:
        parser.save_to_json(documents, output_file)
        print(f"Parsing complete. {len(documents)} documents saved to {output_file}")
            
        return documents
    else:
        print("Failed to parse documentation")
        return []

def parse_html_content(html_content: str, output_file: str = "cuda_section_parsed.json"):
    parser = CUDADocsParser()
    soup = BeautifulSoup(html_content, 'html.parser')
    
    documents = parser.parse_document(soup)
    
    if documents:
        parser.save_to_json(documents, output_file)
        print(f"HTML content parsed. {len(documents)} documents saved to {output_file}")
        return documents
    else:
        print("Failed to parse HTML content")
        return []

if __name__ == "__main__":
    parse_cuda_docs()
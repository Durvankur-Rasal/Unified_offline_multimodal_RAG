import os
import glob
import click

# Import our custom modules from the src/ directory
from src.document_loaders import get_loader
from src.semantic_processor import SemanticProcessor

SOURCE_DIR = "source_documents"
INDEX_DIR = "faissindex"

def ensure_directories():
    """Ensure necessary directories exist before starting."""
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

@click.command()
def main():
    """
    Multimodal Data Ingestion Pipeline.
    Reads files from source_documents/, extracts text, and builds a FAISS index.
    """
    ensure_directories()
    
    print(f"Starting ingestion from: {SOURCE_DIR}/")
    
    # Grab all files in the source directory
    all_files = glob.glob(os.path.join(SOURCE_DIR, "*.*"))
    
    if not all_files:
        print(f"No files found in {SOURCE_DIR}/. Please add some PDFs, DOCX, images, or audio files.")
        return

    text_corpus = []
    error_log = []

    # 1. Extraction Phase (Algorithm 1: Multimodal Ingestion) 
    print("\n--- Phase 1: File Extraction ---")
    for file_path in all_files:
        filename = os.path.basename(file_path)
        print(f"Processing: [{filename}]")
        
        try:
            # Dynamically get the correct loader based on the file extension
            loader = get_loader(file_path)
            extracted_text_list = loader.load(file_path)
            
            # Combine all pages/paragraphs from the file into one string
            full_text = "\n".join(extracted_text_list)
            
            if full_text.strip():
                # Store text along with its metadata (source filename)
                text_corpus.append({"text": full_text, "source": filename})
                print(f" -> Successfully extracted text from {filename}")
            else:
                print(f" -> Warning: No readable text found in {filename}")
                
        except ValueError as ve:
            # Handle unsupported file types gracefully
            print(f" -> Skipped {filename}: {ve}")
            error_log.append(f"Unsupported type: {filename}")
        except Exception as e:
            # Handle processing errors (e.g., corrupted files)
            print(f" -> Error processing {filename}: {e}")
            error_log.append(f"Failed {filename}: {e}")

    # 2. Semantic Processing Phase (Algorithm 2: Chunking & Indexing) 
    print("\n--- Phase 2: Semantic Indexing ---")
    if text_corpus:
        try:
            processor = SemanticProcessor(index_dir=INDEX_DIR)
            processor.chunk_and_embed(text_corpus)
        except Exception as e:
            print(f"Error during semantic processing and indexing: {e}")
    else:
        print("No readable text was extracted. Skipping indexing phase.")

    # 3. Final Summary
    print("\n--- Ingestion Summary ---")
    print(f"Total files detected: {len(all_files)}")
    print(f"Files successfully extracted: {len(text_corpus)}")
    
    if error_log:
        print("\nErrors encountered:")
        for err in error_log:
            print(f"- {err}")
    else:
        print("\nAll files processed successfully! You are ready to run the query pipeline.")

if __name__ == "__main__":
    main()
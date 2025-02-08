from PIL import Image
import pytesseract
from gtts import gTTS, gTTSError
import os
from typing import Union, List
from pathlib import Path
import argparse
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

class ContentToSpeech:
    def __init__(self, output_dir: str = 'output/'):
        """
        Initialize the converter with an output directory for audio files
        
        Args:
            output_dir (str): Directory to save output audio files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_files_from_directory(self, directory: Union[str, Path], 
                               file_types: List[str] = None) -> List[Path]:
        """
        Get all files of specified types from a directory
        
        Args:
            directory (Union[str, Path]): Directory to scan
            file_types (List[str]): List of file extensions to include (e.g., ['.jpg', '.png'])
                                   If None, all files will be included
        
        Returns:
            List[Path]: List of file paths
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        if file_types:
            file_types = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                         for ext in file_types]
            files = []
            for ext in file_types:
                files.extend(directory.glob(f'*{ext}'))
        else:
            files = list(directory.glob('*'))
        
        return sorted(files)

    def extract_text_from_images(self, image_paths: List[Path], lang: str = 'ukr') -> str:
        """
        Extract text from a list of image paths.
        
        Args:
            image_paths (List[Path]): List of paths to image files
            lang (str): Language code for OCR (default: 'ukr' for Ukrainian)
            
        Returns:
            str: Concatenated text from all images
        """
        full_text = ""
        
        for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
            try:
                img = Image.open(image_path).convert('RGB')
                text = pytesseract.image_to_string(img, lang=lang)
                full_text += text + "\n"
            except (IOError, SyntaxError, TypeError) as e:
                print(f"\nError processing image {image_path}: {e}")
        
        return full_text.strip()
    
    def read_text_files(self, text_paths: List[Path], encoding: str = 'utf-8') -> str:
        """
        Read and concatenate text from multiple text files.
        
        Args:
            text_paths (List[Path]): List of paths to text files
            encoding (str): File encoding to use
            
        Returns:
            str: Concatenated text from all files
        """
        full_text = ""
        
        for text_path in tqdm(text_paths, desc="Processing text files", unit="file"):
            try:
                with open(text_path, 'r', encoding=encoding) as f:
                    full_text += f.read() + "\n"
            except Exception as e:
                print(f"\nError processing text file {text_path}: {e}")
        
        return full_text.strip()

    @retry(
        stop=stop_after_attempt(3),  # Reduce number of attempts
        wait=wait_exponential(multiplier=5, min=30, max=300),
        retry=lambda retry_state: retry_state.outcome.failed and
              isinstance(retry_state.outcome.exception(), gTTSError)
    )
    def text_to_speech_with_retry(self, text: str, output_path: Path, lang: str = 'uk'):
        """
        Convert text to speech with retry logic for rate limiting
        
        Args:
            text (str): Text to convert
            output_path (Path): Path to save the audio file
            lang (str): Language code for TTS
        """
        try:
            tts = gTTS(text, lang=lang)
            tts.save(str(output_path))
            # Add a longer delay after successful conversion
            time.sleep(5)
        except Exception as e:
            print(f"\nAttempting retry due to error: {str(e)}")
            raise

    def split_text_into_chunks(self, text: str, chunk_size: int = 3000) -> List[str]:
        """
        Split text into larger chunks to reduce API calls.
        We increase the chunk size to 3000 characters (from 1000)
        and split on sentences instead of words for better audio quality.
        
        Args:
            text (str): Text to split
            chunk_size (int): Maximum size of each chunk
            
        Returns:
            List[str]: List of text chunks
        """
        # First split by paragraphs
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            # Split paragraph into sentences (basic splitting)
            sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
            
            for sentence in sentences:
                sentence_size = len(sentence) + 1  # +1 for space/newline
                
                if current_size + sentence_size > chunk_size and current_chunk:
                    # Join the current chunk and add it to chunks
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_size
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def text_to_speech(self, text: str, output_filename: str, lang: str = 'uk') -> Path:
        """
        Convert text to speech and save as audio file.
        Modified to handle chunks more efficiently.
        
        Args:
            text (str): Text to convert to speech
            output_filename (str): Name of the output audio file
            lang (str): Language code for TTS (default: uk for Ukrainian)
            
        Returns:
            Path: Path to the saved audio file
        """
        if not text:
            raise ValueError("Input text is empty")
        
        # Split text into chunks and show total number of chunks
        chunks = self.split_text_into_chunks(text)
        print(f"\nProcessing {len(chunks)} chunks. This may take a while...")
        
        # temporary directory for chunks
        temp_dir = self.output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Process each chunk with progress bar
        chunk_files = []
        with tqdm(total=len(chunks), desc="Converting to speech", unit="chunk") as pbar:
            for i, chunk in enumerate(chunks):
                chunk_path = temp_dir / f"chunk_{i}.mp3"
                
                try:
                    # Show current chunk size
                    print(f"\nProcessing chunk {i+1}/{len(chunks)} (size: {len(chunk)} characters)")
                    
                    self.text_to_speech_with_retry(chunk, chunk_path, lang)
                    chunk_files.append(chunk_path)
                    pbar.update(1)
                    
                    # Add an additional delay between chunks
                    time.sleep(10)
                    
                except Exception as e:
                    print(f"\nError processing chunk {i+1}: {e}")
                    continue
        
        if not chunk_files:
            raise ValueError("No chunks were successfully processed")
        
        # Combine all chunks into final output
        output_path = self.output_dir / output_filename
        
        if len(chunk_files) == 1:
            # If only one chunk, just rename it
            chunk_files[0].rename(output_path)
        else:
            # Combine multiple chunks using ffmpeg
            try:
                import ffmpeg
                
                # Create file list for ffmpeg
                with open(temp_dir / "files.txt", "w") as f:
                    for chunk_file in chunk_files:
                        f.write(f"file '{chunk_file}'\n")
                
                # Combine files
                stream = ffmpeg.input(temp_dir / "files.txt", format='concat', safe=0)
                stream = ffmpeg.output(stream, str(output_path), c='copy')
                ffmpeg.run(stream, overwrite_output=True)
                
            except ImportError:
                print("ffmpeg-python not installed. Installing it would enable better audio concatenation")
                # Fallback to basic concatenation
                with open(output_path, 'wb') as outfile:
                    for chunk_file in chunk_files:
                        with open(chunk_file, 'rb') as infile:
                            outfile.write(infile.read())
        
        # Cleanup temporary files
        for chunk_file in chunk_files:
            try:
                chunk_file.unlink()
            except:
                pass
        try:
            (temp_dir / "files.txt").unlink()
            temp_dir.rmdir()
        except:
            pass
        
        return output_path
    
    def process_directory(self, 
                         input_dir: Union[str, Path],
                         output_filename: str,
                         content_type: str = 'text',
                         file_types: List[str] = None,
                         ocr_lang: str = 'ukr',
                         tts_lang: str = 'uk',
                         encoding: str = 'utf-8') -> Path:
        """
        Process all files in a directory and convert to speech.
        
        Args:
            input_dir (Union[str, Path]): Input directory containing files to process
            output_filename (str): Name of the output audio file
            content_type (str): Type of content - 'text' or 'images'
            file_types (List[str]): List of file extensions to process
            ocr_lang (str): Language code for OCR
            tts_lang (str): Language code for TTS
            encoding (str): Encoding for text files
            
        Returns:
            Path: Path to the saved audio file
        """
        input_dir = Path(input_dir)
        
        if file_types is None:
            if content_type == 'text':
                file_types = ['.txt', '.doc', '.docx']
            elif content_type == 'images':
                file_types = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        files = self.get_files_from_directory(input_dir, file_types)
        
        if not files:
            raise ValueError(f"No matching files found in directory: {input_dir}")
        
        if content_type == 'text':
            full_text = self.read_text_files(files, encoding)
        elif content_type == 'images':
            full_text = self.extract_text_from_images(files, ocr_lang)
        else:
            raise ValueError(f"Invalid content type: {content_type}")
        
        return self.text_to_speech(full_text, output_filename, tts_lang)

def main():
    parser = argparse.ArgumentParser(description='Convert text or images to speech')
    
    parser.add_argument('input_dir', help='Directory containing input files')
    parser.add_argument('output_filename', help='Output audio filename (e.g., output.mp3)')
    
    type_group = parser.add_mutually_exclusive_group(required=True)
    type_group.add_argument('--images', action='store_true', help='Process images')
    type_group.add_argument('--text', action='store_true', help='Process text files')
    
    parser.add_argument('--output-dir', default='output/', help='Output directory for audio files')
    parser.add_argument('--ocr-lang', default='ukr', help='OCR language code (default: ukr)')
    parser.add_argument('--tts-lang', default='uk', help='TTS language code (default: uk)')
    parser.add_argument('--encoding', default='utf-8', help='Text file encoding (default: utf-8)')
    parser.add_argument('--file-types', nargs='+', help='Specific file extensions to process (e.g., .jpg .png)')
    
    args = parser.parse_args()
    
    try:
        converter = ContentToSpeech(output_dir=args.output_dir)
        
        content_type = 'images' if args.images else 'text'
        
        output_path = converter.process_directory(
            input_dir=args.input_dir,
            output_filename=args.output_filename,
            content_type=content_type,
            file_types=args.file_types,
            ocr_lang=args.ocr_lang,
            tts_lang=args.tts_lang,
            encoding=args.encoding
        )
        
        print(f"\nSuccess! Audio saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)

if __name__ == "__main__":
    main()
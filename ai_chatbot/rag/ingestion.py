# rag_core/rag/ingestion.py

from .pdf_loader import load_pdf
from .chunking import chunk_text
from .embedding import embed_texts
from .vector_store import upsert_chunks, clear_namespace, delete_document_vectors
import threading
import logging
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
from io import BytesIO
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

logger = logging.getLogger(__name__)

vision_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)

def _image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def describe_slide(image):
    image_b64 = _image_to_base64(image)
    
    prompt = (
        "Analyze this image from a presentation slide in detail. "
        "1. Extract ALL visible text word-for-word (OCR), especially years, titles, legislation acts, and key terms. "
        "2. If there is a chart or graph, explain the X and Y axes, legends, and data points clearly. "
        "3. If there is a table, extract and summarize the rows and columns. "
        "4. Describe what the image represents overall."
    )
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    )
    
    response = vision_llm.invoke([message])
    return response.content

def _process_slide_deck_task(file_path, namespace, file_type, original_filename):
    try:
        logger.info(f"Starting async ingestion for {file_path} in namespace {namespace}")
        # Delete only the old version of this specific document if it existed
        delete_document_vectors(namespace, original_filename)
        
        chunks = []
        metadatas = []

        is_pptx = original_filename.lower().endswith(('.pptx', '.ppt'))

        if is_pptx:
            logger.info("Processing as PPTX using python-pptx")
            try:
                prs = Presentation(file_path)
                
                # Helper function to recursively extract from shapes
                def extract_from_shape(shape, text_list, img_descriptions, slide_idx):
                    if hasattr(shape, "text") and shape.text.strip():
                        text_list.append(shape.text.strip())
                    
                    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                        for child_shape in shape.shapes:
                            extract_from_shape(child_shape, text_list, img_descriptions, slide_idx)
                    elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        try:
                            image_stream = BytesIO(shape.image.blob)
                            img = Image.open(image_stream)
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            img_desc = describe_slide(img)
                            img_descriptions.append(img_desc)
                        except Exception as e:
                            logger.error(f"Error parsing picture in slide {slide_idx}: {e}")

                for i, slide in enumerate(prs.slides):
                    slide_text_list = []
                    image_descriptions = []
                    
                    for shape in slide.shapes:
                        extract_from_shape(shape, slide_text_list, image_descriptions, i)
                    
                    slide_text = "\n".join(slide_text_list)
                    
                    combined_description = f"Slide {i+1} Text:\n{slide_text}\n"
                    if image_descriptions:
                        combined_description += "Slide Images Descriptions:\n" + "\n".join(image_descriptions)
                        
                    chunks.append(combined_description)
                    metadatas.append({
                        "text": combined_description,
                        "is_slide": True,
                        "slide_index": i + 1,
                        "file_type": file_type
                    })
            except Exception as e:
                logger.error(f"Error processing PPTX file: {e}")
                return
        else:
            logger.info("Processing as PDF using PyMuPDF (fitz)")
            doc = fitz.open(file_path)
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=150)
                mode = "RGBA" if pix.alpha else "RGB"
                image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    
                description = describe_slide(image)
                chunks.append(description)
                metadatas.append({
                    "text": description,
                    "is_slide": True,
                    "slide_index": i + 1,
                    "file_type": file_type
                })
            
        if not chunks:
            logger.warning("No chunks were generated from the slide deck.")
            return

        embeddings = embed_texts(chunks)
        upsert_chunks(chunks, embeddings, namespace, original_filename, custom_metadata=metadatas)
        
        logger.info(f"Successfully processed {len(chunks)} slides for namespace {namespace}")
        
    except Exception as e:
        logger.error(f"Failed to process document asynchronously: {e}")

def process_document_async(file_path, file_type, namespace, original_filename):
    thread = threading.Thread(
        target=_process_slide_deck_task, 
        args=(file_path, namespace, file_type, original_filename)
    )
    thread.start()
    return {"message": "Processing started in the background."}

def process_pdf(file, session_id, original_filename):
    # Delete only the old version of this specific document if it existed
    delete_document_vectors(session_id, original_filename)

    # 1. Load
    text = load_pdf(file)

    # 2. Chunk
    chunks = chunk_text(text)

    # 3. Embed
    embeddings = embed_texts(chunks)

    # 4. Store (namespace = session)
    upsert_chunks(chunks, embeddings, namespace=session_id, filename=original_filename)

    return len(chunks)
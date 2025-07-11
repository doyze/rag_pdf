import base64
import gradio as gr
import os
import shutil
import pymupdf as fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import qdrant_client
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

import torch
import ollama
import uuid
import shortuuid
import logging
import re
from openai import OpenAI
import google.generativeai as genai
import google.api_core.exceptions
from dotenv import load_dotenv

from typing import List, Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Image folder
TEMP_IMG="./data/images"
TEMP_VECTOR="./data/chromadb"

summarize = ""

def get_available_models():
    """
    Dynamically fetches available models from Ollama and Google Gemini.
    """
    models = []
    # Fetch Ollama models
    try:
        ollama_models = ollama.list()
        # Use .get() for safer access to the 'name' key
        models.extend([model.get('name') for model in ollama_models.get('models', []) if model.get('name')])
    except Exception as e:
        logger.warning(f"Could not fetch Ollama models: {e}")

    # Fetch Gemini models
    try:
        if GEMINI_API_KEY:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    models.append(m.name.replace("models/", "")) # Clean up the name
    except Exception as e:
        logger.warning(f"Could not fetch Gemini models: {e}")
    
    # Fallback to a default list if fetching fails
    if not models:
        logger.warning("Could not dynamically fetch models, using fallback list.")
        models = ["pdf-gemma", "pdf-qwen", "pdf-llama", "gemini-1.5-flash", "gemini-1.5-pro-latest"]
        
    return sorted(list(set(models))) # Return sorted unique list

# Load environment variables from .env file
load_dotenv()

# Google Gemini API Key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not found in .env file. Gemini models will not be available.")

AVAILABLE_MODELS = get_available_models()

# LM Studio Client
lm_studio_client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")


# ตั้งค่า device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# โหลดโมเดล embedding
# SentenceTransformer สำหรับข้อความหลายภาษา (เน้นภาษาไทย)
sentence_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")
collection_name = "pdf_data"

# Create collection if it doesn't exist
try:
    if not qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=sentence_model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
        )
except Exception as e:
    logging.error(f"Error creating collection: {e}")

# Create directory for storing images
os.makedirs(TEMP_IMG, exist_ok=True)

sum_tokenizer = MT5Tokenizer.from_pretrained('StelleX/mt5-base-thaisum-text-summarization')
sum_model = MT5ForConditionalGeneration.from_pretrained('StelleX/mt5-base-thaisum-text-summarization').to(device)

def summarize_content(content: str) -> str:
    """
        สรุปเนื้อหา 
    """
    logging.info("%%%%%%%%%%%%%% SUMMARY %%%%%%%%%%%%%%%%%%%%%")    
       
    input_ = sum_tokenizer(content, truncation=True, max_length=1024, return_tensors="pt")
    with torch.no_grad():
        preds = sum_model.generate(
            input_['input_ids'].to(device),
            num_beams=15,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
            max_length=250
        )

    summary = sum_tokenizer.decode(preds[0], skip_special_tokens=True)

    logging.info(f" summary: {summary}.")
    logging.info("%%%%%%%%%%%%%% SUMMARY %%%%%%%%%%%%%%%%%%%%%")
    return summary

# แยกเนื้อหา, รูป ออกจาก PDF
def ocr_with_typhoon(image_path: str) -> str:
    """
    ใช้โมเดล typhoon-ocr-7b ผ่าน LM Studio เพื่อดึงข้อความจากรูปภาพ
    """
    try:
        logging.info(f"Performing OCR on {image_path}")
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')

        response = lm_studio_client.chat.completions.create(
            model="local-model/typhoon-ocr-7b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text from this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
            timeout=60.0,
        )
        ocr_text = response.choices[0].message.content
        logging.info(f"OCR result: {ocr_text[:100]}...")
        return ocr_text
    except Exception as e:
        logger.error(f"Error during OCR with Typhoon: {e}", exc_info=True)
        return ""

def extract_pdf_content(pdf_path: str) -> List[Dict]:
    """
    แยกข้อความและรูปภาพจาก PDF โดยใช้ PyMuPDF และ Typhoon OCR
    """
    try:
        doc = fitz.open(pdf_path)
        content_chunks = []
        all_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text using PyMuPDF
            text = page.get_text("text").strip()
            if not text:
                text = f"ไม่มีข้อความในหน้า {page_num + 1}"
            
            logging.info("################# Text data ##################")
            chunk_data = {"text": f"ข้อมูลจากหน้า {page_num + 1} : {text}", "images": []}
            
            # Extract images
            image_list = page.get_images(full=True)
            logging.info("################# images list ##################")
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    img_id = f"pic_{page_num + 1}_{img_index + 1}"
                    img_path = f"{TEMP_IMG}/{img_id}.{image_ext}"
                    image.save(img_path, format=image_ext.upper())

                    # Perform OCR with Typhoon
                    ocr_text = ocr_with_typhoon(img_path)
                    all_text.append(f"{ocr_text}\n\n\n")

                    img_desc = f"รูปภาพ จากหน้า {page_num + 1} ของ รูปที่ {img_index + 1}, OCR Text: {ocr_text[:80]}..."
                    chunk_data["text"] += f"\n[ภาพ: {img_id}.{image_ext} - {ocr_text}]"
                    chunk_data["images"].append({
                        "data": image,
                        "path": img_path,
                        "description": img_desc
                    })
                except Exception as e:
                    logger.warning(f"ไม่สามารถประมวลผลรูปภาพที่หน้า {page_num + 1}, รูปที่ {img_index + 1}: {e}")
            
            all_text.append(f"{text}\n\n\n")
            if chunk_data["text"]:
                content_chunks.append(chunk_data)
        
        if not any(chunk["images"] for chunk in content_chunks):
            logger.warning("ไม่พบรูปภาพใน PDF: %s", pdf_path)
        
        doc.close()
        content_text = "".join(all_text)
        # ตัดคำภาษาไทย
        thaitoken_text = preprocess_thai_text(content_text) if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in content_text) else content_text
        print("################################")
        print(f"{thaitoken_text}")
        print("################################")
        global summarize
        summarize = summarize_content(thaitoken_text)
        return content_chunks
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดในการแยก PDF: %s", str(e))
        raise

# ตัดคำภาษาไทย 
def preprocess_thai_text(text: str) -> str:
    """
    ตัดคำภาษาไทยด้วย pythainlp เพื่อเตรียมข้อความ

    Args:
        text (str): ข้อความภาษาไทย

    Returns:
        str: ข้อความที่ตัดคำแล้ว
    """
    return " ".join(word_tokenize(text, engine="newmm"))


def embed_text(text: str) -> np.ndarray:
    """
    สร้าง embedding สำหรับข้อความโดยใช้ SentenceTransformer 

    Args:
        text (str): ข้อความที่ต้องการสร้าง embedding        

    Returns:
        np.ndarray: Embedding vector ที่รวมจากหลายโมเดล
    """
    logging.info("-------------- start embed text  -------------------")
    
    # ตัดคำภาษาไทย
    processed_text = preprocess_thai_text(text) if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else text
    
    # สร้าง embedding ด้วย SentenceTransformer
    sentence_embedding = sentence_model.encode(processed_text, normalize_embeddings=True, device=device)    
        
    return sentence_embedding

def store_in_qdrant(content_chunks: List[Dict], pdf_name: str):
    """
    เก็บข้อมูลข้อความและรูปภาพใน Qdrant พร้อม embedding
    """
    logging.info("##### Start store in qdrant #########")
    points = []
    for chunk in content_chunks:
        text = chunk["text"]
        images = chunk["images"]
        text_embedding = embed_text(text)
        
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=text_embedding.tolist(),
                payload={"type": "text", "source": pdf_name, "text": text}
            )
        )

        for img in images:
            img_path = img["path"]
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=text_embedding.tolist(),
                    payload={"type": "image", "source": pdf_name, "image_path": img_path, "text": text}
                )
            )
    
    qdrant_client.upsert(collection_name=collection_name, points=points, wait=True)

def process_pdf_upload(pdf_file):
    """
    จัดการการอัปโหลดและประมวลผล PDF
    """
    try:
        if pdf_file is None:
            return "กรุณาอัปโหลดไฟล์ PDF ก่อน", "", None
        
        pdf_path = pdf_file.name
        pdf_name = os.path.basename(pdf_path)
        logging.info(f"#### Start processing {pdf_name} ####")
        
        content_chunks = extract_pdf_content(pdf_path)
        if not content_chunks:
            return f"ไม่สามารถแยกเนื้อหาจาก {pdf_name} ได้", "", None

        logging.info("#### Storing content in vector db ####")
        store_in_qdrant(content_chunks, pdf_name)
        
        logging.info(f"#### Finished processing {pdf_name} ####")
        
        indexed_files_text = "\n".join(get_indexed_files())
        return f"ประมวลผลและจัดเก็บ {pdf_name} สำเร็จ", indexed_files_text, None
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการประมวลผล PDF: {e}", exc_info=True)
        return f"เกิดข้อผิดพลาดในการประมวลผล PDF: {str(e)}", "", None

def clear_vector_db():
    try:
        if qdrant_client.collection_exists(collection_name=collection_name):
            qdrant_client.delete_collection(collection_name=collection_name)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=sentence_model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
        )
    except Exception as e:
        logging.error(f"Error clearing Qdrant collection: {e}")
        raise e

def clear_all_data():
    """
    ล้างข้อมูลใน vector database และไฟล์ในโฟลเดอร์ images
    """
    try:
        clear_vector_db()
        if os.path.exists(TEMP_IMG):
            shutil.rmtree(TEMP_IMG)
            os.makedirs(TEMP_IMG, exist_ok=True)
        
        return "ล้างข้อมูลใน vector database และโฟลเดอร์ images สำเร็จ", None, ""
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการล้างข้อมูล: {str(e)}", None, ""

def get_indexed_files():
    """
    ดึงรายชื่อไฟล์ที่ถูกจัดเก็บใน Qdrant
    """
    try:
        # This is a simplified way to get distinct sources.
        # For very large collections, a more efficient approach might be needed.
        response, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=10000,  # Adjust limit as needed
            with_payload=["source"],
            with_vectors=False
        )
        sources = {point.payload['source'] for point in response}
        return list(sources)
    except Exception as e:
        logger.error(f"Error fetching indexed files: {e}")
        return []


def query_rag(question: str,  chat_llm: str = "pdf-qwen"):
    """
    ค้นหาในระบบ RAG และสร้างคำตอบแบบ streaming โดยใช้ Ollama
    """
    logging.info(f"####  RAG get Question #### ")
    question_embedding = embed_text(question)
    
    results=[]
    # เช็คข้อมูลจาก เอกสาร ดึงมา 3 รายการ   ##  Retrival
    max_result = 3
    if "กี่" in question:
        max_result = 5
    
    if "ทั้งหมด" in question:
        max_result = 10

    if "บ้าง" in question:
        max_result = 5

    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=question_embedding.tolist(),
        limit=max_result
    )
    logging.info(f"##### results from vector: { results }")
    context_texts = []
    image_paths = []

    for result in results:
        doc = result.payload['text']
        metadata = result.payload
        context_texts.append(doc)
        logging.info(doc)
        logging.info(f"metadata: {metadata}")
        # Regex pattern สำหรับค้นหา [img: ชื่อไฟล์.jpeg]
        pattern = r"pic_(\d+)_(\d+)\.jpeg"

        # ค้นหาทุกรูป แบบที่ตรงกับ ส่งเข้ามา
        imgs = re.findall(pattern, doc)
        print("----------IIIII------------")
        print(imgs)
        print("----------IIIII------------")
        if imgs:
            image_paths.append(imgs)
            logging.info(f"img: {imgs}")

        print("---------------------------")
        if metadata:
            if metadata["type"] == "image":
                logging.info(f"image_path : { metadata['image_path']}")
                image_paths.append(metadata['image_path'])
    


    context = "\n".join(context_texts)
    ##  Augmented
    logging.info("############## Begin Augmented prompt #################")
    prompt = f"""จากบริบทต่อไปนี้ ตอบคำถาม: {question}

    บริบท: 
        {summarize}

        {context}

    ให้คำตอบที่ชัดเจนและกระชับเป็นภาษาไทย หากบริบทมีชื่อไฟล์รูปภาพ ให้ระบุชื่อไฟล์ในวงเล็บเหลี่ยมแบบนี้: [ภาพ: ชื่อไฟล์] """ 
    
    logging.info(f"promt: {prompt}")
    logging.info("##############  End Augmented prompt #################")

    logging.info("+++++++++++++  Send prompt To LLM  ++++++++++++++++++")
    ## Generation  เพื่อการตอบ chat
    if chat_llm.startswith("gemini"):
        gemini_model = genai.GenerativeModel(chat_llm)
        stream = gemini_model.generate_content(prompt, stream=True)
    else:
        stream = ollama.chat(
            model=chat_llm,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
    
    return stream

def user(user_message: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    จัดการ input ของผู้ใช้และเพิ่มลงในประวัติการแชท
    """
    return "", history + [{"role": "user", "content": user_message}]

def chatbot_interface(history: List[Dict], llm_model: str):
    """
    อินเทอร์เฟซแชทบอทแบบ streaming
    """
    user_message = history[-1]["content"]
    
    stream= query_rag(user_message, chat_llm=llm_model)

    history.append({"role": "assistant", "content": ""})
    full_answer=""
    """
    ส่วนของการ ตอบคำถาม
    """
    try:
        for chunk in stream:
            if llm_model.startswith("gemini"):
                content = chunk.text
            else:
                content = chunk["message"]["content"]
            
            full_answer += content
            history[-1]["content"] += content
            yield history
    except google.api_core.exceptions.ResourceExhausted as e:
        logger.error(f"Gemini API quota exceeded: {e}", exc_info=True)
        error_message = "ขออภัย, คุณใช้โควต้าฟรีของ Gemini API เกินกำหนดแล้ว กรุณาตรวจสอบแผนการใช้งานของคุณ"
        history[-1]["content"] = error_message
        yield history
    except google.api_core.exceptions.ServiceUnavailable as e:
        logger.error(f"Gemini API service unavailable: {e}", exc_info=True)
        error_message = "ขออภัย, โมเดล Gemini กำลังทำงานหนักในขณะนี้ กรุณาลองใหม่อีกครั้งในภายหลัง"
        history[-1]["content"] = error_message
        yield history
    except Exception as e:
        logger.error(f"Error during response generation: {e}", exc_info=True)
        error_message = f"ขออภัย, เกิดข้อผิดพลาดในการสร้างคำตอบ: {e}"
        history[-1]["content"] = error_message
        yield history
    

    """
    ส่วนของการดึงรูปภาพ ที่เกี่ยวข้องมาแสดง โดยดึงจาก คำตอบด้านบน 
    """

    # ใช้ regex เพื่อดึงชื่อไฟล์ที่อยู่ใน [ภาพ: ...] 
    print(full_answer)
    pattern1 = r"\[(?:ภาพ:\s*)?(pic_\w+[-_]?\w*\.(?:jpe?g|png))\]"
    pattern2 = r"(pic_\w+[-_]?\w*\.(?:jpe?g|png))"
    # ค้นหาทุกรูป แบบที่ตรงกับ ส่งเข้ามา
    
    print("----------PPPP------------")       
    image_list = re.findall(pattern1, full_answer)
    print(image_list)
    if (len(image_list)==0):
        image_list = re.findall(pattern2, full_answer)
    print("----------xxxx------------")  
    # ดึงเฉพาะรูปที่ไม่ซ้ำกัน
    image_list_uniq = list(dict.fromkeys(image_list))  
    if image_list_uniq:
        history[-1]["content"] += "\n\n**รูปภาพที่เกี่ยวข้อง:**"
        yield history
        # ดึงรูปมาแสดง
        for img in image_list_uniq:
            img_path = f"{TEMP_IMG}/{img}"
            logger.info(f"Displaying image: {img_path}")
            if os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode()
                    
                    mime_type = f"image/{os.path.splitext(img)[1][1:]}"
                    image_response = f"![{img}](data:{mime_type};base64,{encoded_string})"
                    
                    history.append(
                        {
                            "role": "assistant",
                            "content": image_response,
                        }
                    )
                    yield history
                except Exception as e:
                    logger.error(f"Error displaying image {img_path}: {e}")



# Gradio interface
with gr.Blocks() as demo:
    logo="https://camo.githubusercontent.com/9433204b08afdc976c2e4f5a4ba0d81f8877b585cc11206e2969326d25c41657/68747470733a2f2f63646e2e6a7364656c6976722e6e65742f67682f6e61726f6e67736b6d6c2f68746d6c352d6c6561726e406c61746573742f6173736574732f696d67732f546c697665636f64654c6f676f2d3435302e77656270"
    gr.Markdown(f"""<h3 style='display: flex; align-items: center; gap: 15px; padding: 10px; margin: 0;'>
        <img alt='T-LIVE-CODE' src='{logo}' style='height: 100px;' >
        <span style='font-size: 1.5em;'>แชทบอท PDF: RAG</span></h3>""")

    with gr.Tab("แอดมิน - อัปโหลด PDF"):
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="อัปโหลดไฟล์ PDF")
                upload_button = gr.Button("ประมวลผล PDF")
                clear_button = gr.Button("ล้างข้อมูล")
                upload_output = gr.Textbox(label="สถานะการอัปโหลด")
            with gr.Column(scale=1):
                indexed_files_display = gr.Textbox(
                    label="ไฟล์ในฐานข้อมูล",
                    lines=10,
                    interactive=False,
                    value="\n".join(get_indexed_files())
                )

        upload_button.click(
            fn=process_pdf_upload,
            inputs=pdf_input,
            outputs=[upload_output, indexed_files_display, pdf_input]
        )
        clear_button.click(
            fn=clear_all_data,
            inputs=None,
            outputs=[upload_output, pdf_input, indexed_files_display],
            queue=False
        )
    
    with gr.Tab("แชท"):
        # Choice เลือก Model
        model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value="gemini-1.5-flash",
            label="เลือก LLM Model"
        )
        selected_model = gr.State(value="pdf-gemma")  # เก็บไว้ใน state
        model_selector.change(fn=lambda x: x, inputs=model_selector, outputs=selected_model)
        # Chat Bot
        chatbot = gr.Chatbot(type="messages")
        with gr.Row():
            msg = gr.Textbox(label="ถามคำถามเกี่ยวกับ PDF", elem_id="chat_msg", scale=10)
        
        # Submit function 
        msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False
        ).then(
            fn=chatbot_interface,
            inputs=[chatbot, selected_model],
            outputs=chatbot
        )

if __name__ == "__main__":
    # The application will no longer clear data on startup.
    # Data will only be cleared when the "ล้างข้อมูล" button is pressed.
    demo.launch()

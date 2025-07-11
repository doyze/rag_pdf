# แชทบอท PDF ด้วย RAG (Retrieval-Augmented Generation)

โปรเจกต์นี้คือระบบแชทบอทที่ให้ผู้ใช้สามารถอัปโหลดไฟล์ PDF และสนทนาโต้ตอบเกี่ยวกับเนื้อหาในเอกสารนั้นได้ โดยใช้เทคนิค Retrieval-Augmented Generation (RAG) เพื่อให้คำตอบมีความถูกต้องและอ้างอิงจากข้อมูลใน PDF โดยตรง

## คุณสมบัติหลัก

- **รองรับภาษาไทย:** สามารถประมวลผลและตัดคำภาษาไทยในเอกสาร PDF ได้
- **ดึงข้อมูลจาก PDF:** แยกข้อความและรูปภาพออกจากไฟล์ PDF
- **สรุปเนื้อหา:** สร้างบทสรุปย่อของเอกสาร PDF โดยอัตโนมัติ
- **ถาม-ตอบ:** ตอบคำถามจากเนื้อหาในเอกสาร โดยใช้ LLM ผ่าน Ollama
- **แสดงผลรูปภาพ:** หากคำตอบเกี่ยวข้องกับรูปภาพในเอกสาร จะแสดงรูปภาพนั้นประกอบในหน้าแชท
- **ส่วนติดต่อผู้ใช้ (UI):** ใช้งานง่ายผ่านเว็บเบราว์เซอร์ด้วย Gradio แบ่งเป็นส่วนสำหรับอัปโหลดเอกสารและส่วนสำหรับสนทนา

## สถาปัตยกรรม

ระบบทำงานโดยมีขั้นตอนดังนี้:

1.  **อัปโหลดและประมวลผล PDF:**
    - ผู้ใช้ทำการอัปโหลดไฟล์ PDF ผ่านหน้าเว็บ
    - ระบบใช้ `PyMuPDF` เพื่อแยกข้อความและรูปภาพออกจากแต่ละหน้า
    - ข้อความภาษาไทยจะถูกตัดคำด้วย `pythainlp`
    - เนื้อหาทั้งหมดจะถูกสรุปย่อด้วยโมเดล `mt5-base-thaisum`

2.  **สร้างและจัดเก็บ Embeddings:**
    - ข้อความที่แยกออกมาจะถูกแปลงเป็น Vector Embeddings ด้วยโมเดล `multilingual-e5-large`
    - Embeddings พร้อมกับข้อความต้นฉบับและตำแหน่งของรูปภาพจะถูกจัดเก็บลงใน `Qdrant` ซึ่งเป็น Vector Database

3.  **การค้นหาและสร้างคำตอบ (RAG):**
    - เมื่อผู้ใช้พิมพ์คำถามเข้ามา ระบบจะสร้าง Embedding สำหรับคำถามนั้น
    - นำ Embedding ของคำถามไปค้นหาข้อมูล (ข้อความและรูปภาพ) ที่เกี่ยวข้องที่สุดจาก Qdrant (Retrieval)
    - นำข้อมูลที่ค้นหาได้ (Context) มารวมกับคำถามและบทสรุปของเอกสาร เพื่อสร้างเป็น Prompt (Augmented)
    - ส่ง Prompt ไปให้ Large Language Model (LLM) ที่ทำงานบน `Ollama` เพื่อสร้างคำตอบ (Generation)
    - คำตอบและรูปภาพที่เกี่ยวข้องจะถูกส่งกลับไปแสดงผลที่หน้าแชท

## สิ่งที่ต้องมี (Prerequisites)

- **Python 3.8+**
- **Ollama:** สำหรับรัน Large Language Models (LLMs) สามารถดาวน์โหลดได้ที่ [ollama.com](https://ollama.com/)
- **Docker:** สำหรับรัน Qdrant
- **Qdrant:** สำหรับ Vector Database เมื่อติดตั้งและรันผ่าน Docker แล้วจะสามารถเข้าถึง Dashboard ได้ที่ `http://localhost:6333/`

## การติดตั้ง

1.  **Clone a project:**
    ```bash
    git clone https://github.com/T-LINE-CODE-NOW/RAG_PDF_THAI.git
    cd RAG_PDF_THAI
    ```

2.  **รัน Qdrant ผ่าน Docker:**
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

3.  **สร้าง LLM Models:**
    - ตรวจสอบให้แน่ใจว่า Ollama Service กำลังทำงานอยู่
    - รันสคริปต์ `model-create.bat` เพื่อดาวน์โหลดโมเดลพื้นฐานและสร้างโมเดลสำหรับโปรเจกต์นี้ (`pdf-qwen`, `pdf-llama`, `pdf-gemma`)
    ```bash
    model-create.bat
    ```

4.  **สร้าง Virtual Environment (แนะนำ):**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # บน Windows
    # source venv/bin/activate  # บน macOS/Linux
    ```

5.  **ติดตั้ง Dependencies:**
    - **สำหรับผู้ใช้ CPU:**
      ```bash
      pip install -r requirements.txt
      ```
    - **สำหรับผู้ใช้ GPU (NVIDIA):**
      - **ขั้นตอนที่ 1:** ถอนการติดตั้ง PyTorch เวอร์ชันปัจจุบัน (ถ้ามี)
        ```bash
        pip uninstall torch torchaudio torchvision
        ```
      - **ขั้นตอนที่ 2:** ติดตั้ง PyTorch เวอร์ชันที่รองรับ CUDA 12.1
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
      - **ขั้นตอนที่ 3:** ติดตั้งไลบรารีที่เหลือ
        ```bash
        pip install -r requirements.txt
        ```

## การใช้งาน

1.  **ตรวจสอบให้แน่ใจว่า Ollama และ Qdrant กำลังทำงานอยู่**

2.  **รันแอปพลิเคชัน:**
    ```bash
    python rag_pdf.py
    ```

3.  **เปิดเว็บเบราว์เซอร์** และไปที่ URL ที่แสดงใน Terminal (โดยปกติคือ `http://127.0.0.1:7860`)

4.  **อัปโหลดไฟล์:**
    - ไปที่แท็บ "แอดมิน - อัปโหลด PDF"
    - เลือกไฟล์ PDF ที่ต้องการ แล้วกดปุ่ม "ประมวลผล PDF"
    - รอจนกว่าสถานะจะแจ้งว่าประมวลผลสำเร็จ

5.  **เริ่มแชท:**
    - ไปที่แท็บ "แชท"
    - เลือก LLM Model ที่ต้องการจาก Dropdown (`pdf-gemma`, `pdf-qwen`, `pdf-llama`)
    - พิมพ์คำถามเกี่ยวกับเอกสารในช่องข้อความและกด Enter

import streamlit as st
import tempfile
import os
import json 

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Ứng dụng thư viện streamlit để tạo Front end 
st.set_page_config(page_title="Strict RAG & Quiz Gen", layout="wide", page_icon="🎓")
st.title("🎓 Mini NotebookLM - Trợ Giảng & Tạo Bài Tập")
st.caption("Ứng dụng AI trả lời câu hỏi và tự động tạo bài trắc nghiệm (JSON) từ tài liệu.")

# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = [] 

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None 

# SIDEBAR: KHU VỰC TẢI FILE & TẠO QUIZ (JSON)
with st.sidebar:
    st.header("📂 1. Cung cấp tài liệu")
    uploaded_file = st.file_uploader("Tải lên file PDF của bạn", type="pdf")
    
    if uploaded_file and st.button("Xử lý tài liệu"):
        with st.spinner("Đang trích xuất và lập chỉ mục véc-tơ..."):
            # Đọc file PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            reader = PdfReader(tmp_file_path)
            raw_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text
            os.remove(tmp_file_path)
            
            # Băm nhỏ văn bản (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=75,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(raw_text)
            
            # Vector Embeddings & ChromaDB
            embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
            vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
            st.session_state.vector_db = vectorstore
            st.success("Hệ thống đã khóa vùng tri thức vào tài liệu này!")

    st.divider()

    # THÊM MỚI: TÍNH NĂNG TẠO BÀI KIỂM TRA (QUIZ GENERATOR)
    st.header("📝 2. Tạo Bài Trắc Nghiệm (Export JSON)")
    quiz_topic = st.text_input("Nhập chủ đề muốn kiểm tra:", value="Tổng hợp kiến thức quan trọng nhất")
    num_questions = st.number_input("Số lượng câu hỏi:", min_value=1, max_value=15, value=5)

    if st.button("Tạo & Tải Quiz (JSON)"):
        if st.session_state.vector_db is None:
            st.error("Vui lòng xử lý tài liệu PDF ở bước 1 trước!")
        else:
            with st.spinner("Đang phân tích tài liệu và sinh JSON..."):
                # 1. Truy xuất thông tin diện rộng (Lấy 8 đoạn văn bản liên quan đến chủ đề)
                retriever_quiz = st.session_state.vector_db.as_retriever(search_kwargs={"k": 8})
                quiz_docs = retriever_quiz.get_relevant_documents(quiz_topic)
                quiz_context = "\n\n".join([doc.page_content for doc in quiz_docs])

                # 2. Khởi tạo LLM với cờ `format="json"` để ÉP KIỂU đầu ra
                quiz_llm = Ollama(model="llama3.2", temperature=0.1, format="json")

                # 3. Prompt Kỹ thuật cao (Strict JSON Prompt)
                quiz_prompt_template = """
                Bạn là một hệ thống AI chuyên tạo dữ liệu JSON. Dựa vào NGỮ CẢNH dưới đây, hãy tạo ra {num_questions} câu hỏi trắc nghiệm tiếng Việt về chủ đề "{topic}".
                
                RÀNG BUỘC CỰC KỲ QUAN TRỌNG:
                - ĐẦU RA PHẢI LÀ ĐỊNH DẠNG JSON HỢP LỆ 100%. Không có bất kỳ chữ nào khác nằm ngoài cấu trúc JSON.
                - Sử dụng đúng Schema (cấu trúc) mẫu dưới đây.
                
                SCHEMA JSON MẪU:
                {{
                  "title": "Bài kiểm tra: {topic}",
                  "questions": [
                    {{
                      "questionText": "Nội dung câu hỏi ở đây?",
                      "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
                      "correctAnswerIndex": 0,
                      "explanation": "Giải thích chi tiết lý do dựa trên tài liệu."
                    }}
                  ]
                }}

                NGỮ CẢNH:
                {context}
                """
                QUIZ_PROMPT = PromptTemplate(
                    template=quiz_prompt_template, 
                    input_variables=["num_questions", "topic", "context"]
                )
                
                # 4. Chạy luồng sinh JSON
                chain = QUIZ_PROMPT | quiz_llm
                raw_result = chain.invoke({
                    "num_questions": num_questions, 
                    "topic": quiz_topic, 
                    "context": quiz_context
                })

                # 5. Xử lý và kiểm tra tính hợp lệ của JSON trước khi cho tải xuống
                try:
                    quiz_json_object = json.loads(raw_result)
                    # Định dạng lại JSON cho đẹp (indent=2) và hỗ trợ tiếng Việt (ensure_ascii=False)
                    formatted_json = json.dumps(quiz_json_object, indent=2, ensure_ascii=False)
                    
                    st.success("Tạo Bài kiểm tra thành công! Hãy tải xuống bên dưới.")
                    # Nút Tải Xuống
                    st.download_button(
                        label="📥 Tải xuống file quiz_data.json",
                        data=formatted_json,
                        file_name="quiz_data.json",
                        mime="application/json"
                    )
                except json.JSONDecodeError:
                    st.error("AI tạo ra JSON không hợp lệ. Vui lòng thử bấm tạo lại.")
                    st.text("Kết quả thô bị lỗi:")
                    st.code(raw_result)

# ==========================================
# KHU VỰC CHAT CHÍNH (GIỮ NGUYÊN)
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("🔍 Xem nguồn tài liệu trích dẫn gốc"):
                for idx, doc in enumerate(message["sources"]):
                    st.caption(f"**Đoạn trích {idx+1}:** {doc}")

user_query = st.chat_input("Đặt câu hỏi về nội dung bên trong tài liệu đã nạp")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        if st.session_state.vector_db is None:
            response = "Hệ thống chưa có tri thức nền. Vui lòng nạp file tài liệu PDF ở cột bên trái trước."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Đang quét dữ liệu véc-tơ để đối chiếu..."):
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 4})
                retrieved_docs = retriever.get_relevant_documents(user_query)
                
                llm = Ollama(model="llama3.2", temperature=0.0)
                
                prompt_template = """
                ROLE: Bạn là một trợ lý ảo vô cùng nghiêm túc, đóng vai trò chấm điểm và giải đáp dựa trên TÀI LIỆU KHÓA HỌC ĐƯỢC CUNG CẤP.
                
                NHIỆM VỤ: Trả lời câu hỏi của người dùng bằng cách CHỈ sử dụng thông tin có trong phần 'Ngữ cảnh (Context)' dưới đây. 
                
                QUY TẮC TUYỆT ĐỐI:
                1. Không được sử dụng bất kỳ kiến thức nào ngoài đời thực hoặc ngoài đoạn văn văn bản dưới đây.
                2. Nếu câu trả lời không thể tìm thấy hoặc không được suy ra một cách trực tiếp từ Ngữ cảnh, bạn BẮT BUỘC phải trả lời chính xác từ ngữ sau: "Tôi không biết thông tin này vì không có trong tài liệu học tập được cung cấp."
                3. Tuyệt đối không tự bịa ra câu trả lời, không đoán mò, không mở rộng lý thuyết nằm ngoài phạm vi văn bản.

                Ngữ cảnh (Context):
                {context}

                Câu hỏi: {question}

                Câu trả lời tiếng Việt chính xác và ngắn gọn:
                """
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                result = qa_chain.invoke({"query": user_query})
                response = result["result"]
                source_texts = [doc.page_content for doc in retrieved_docs]
                
                st.markdown(response)
                
                with st.expander("🔍 Xem nguồn tài liệu trích dẫn gốc"):
                    for idx, text in enumerate(source_texts):
                        st.caption(f"**Đoạn trích {idx+1}:** {text}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": source_texts
                })
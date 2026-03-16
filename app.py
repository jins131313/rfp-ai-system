import time
import random
import tempfile
import os
import glob
from functools import wraps
import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import PyPDF2
from docx import Document
import io


# Streamlit 웹 UI 구성

st.set_page_config(page_title="RFP 초안작성 AI 시스템", page_icon="📝", layout="centered")

st.title("📝 제안요청서 초안작성 AI 시스템(국립부경대 김진명 作)")
st.markdown("참고 문서를 첨부하고 지시사항을 구체적으로 입력하면 AI가 맞춤형 초안을 작성해 줍니다.")
st.markdown("집중적으로 작성할 파트를 선택하세요(예시 : 요약본, 요구사항 상세) ※제안요청서 전체를 작성하려고 할 시 에러 발생 가능성 있음")

# API 키 설정 (로컬 테스트용)

# 1. 클라우드 배포 상태일 때
API_KEY = st.secrets["GEMINI_API_KEY"]

# 1. 과거 HUG 제안요청서 PDF 데이터 로드 (캐싱 적용)


@st.cache_data
def load_reference_rfps(folder_path="reference_rfps"):
    pdf_texts = {}
    if not os.path.exists(folder_path):
        return pdf_texts
    
    # 폴더 안의 모든 PDF 파일 경로를 가져옵니다.
    filepaths = glob.glob(os.path.join(folder_path, "*.pdf"))
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        text = ""
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                # 앞부분(주로 개요 및 요구사항) 위주로 빠르게 읽기 위해 최대 30페이지만 읽음
                for page in reader.pages[:30]: 
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            pdf_texts[filename] = text
        except Exception as e:
            pass # 암호가 걸려있거나 깨진 PDF는 건너뜀
    return pdf_texts


# 2. 문서 유사도 분석 함수 (과거 사업 Top 5 추출)

def get_top_5_similar_rfps(query_text, corpus_dict):
    if not corpus_dict or not query_text:
        return None
    
    filenames = list(corpus_dict.keys())
    corpus_texts = list(corpus_dict.values())
    
    try:
        # 사용자 입력(또는 초안) + 과거 RFP 텍스트들을 묶어서 형태소 벡터화
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query_text] + corpus_texts)
        
        # 첫 번째(현재 사업)와 나머지(과거 사업들) 간의 코사인 유사도 계산
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        results = []
        for idx, name in enumerate(filenames):
            clean_name = name.replace(".pdf", "")
            results.append({
                "유사 과거 사업명": clean_name,
                "원본파일명": name,
                "유사도(%)": round(cosine_sim[idx] * 100, 2)
            })
            
        # 유사도 높은 순으로 정렬 후 상위 5개만 추출
        df = pd.DataFrame(results).sort_values(by="유사도(%)", ascending=False).head(5)
        
        # 보기 좋게 인덱스를 1부터 시작하도록 조정
        df.index = range(1, len(df) + 1)
        return df
    except Exception as e:
        return None


# 3. 지수 백오프 및 API 호출 함수

def retry_with_exponential_backoff(max_retries=5, base_delay=2.0, max_delay=60.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "quota" in error_msg or "resourceexhausted" in error_msg:
                        if attempt == max_retries - 1:
                            st.error(f"최대 재시도 횟수 초과.")
                            raise e
                        delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * delay)
                        sleep_time = delay + jitter
                        st.warning(f"API 대기 중... {sleep_time:.1f}초 후 재시도합니다.")
                        time.sleep(sleep_time)
                    else:
                        raise e
        return wrapper
    return decorator

@retry_with_exponential_backoff()
def generate_draft(api_key, prompt, uploaded_files=None):
    genai.configure(api_key=api_key)
    generation_config = genai.types.GenerationConfig(max_output_tokens=8192, temperature=0.2)
    model = genai.GenerativeModel('gemini-2.5-pro', generation_config=generation_config)
    
    contents = [prompt]
    temp_file_paths = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension in ['.txt', '.csv', '.md']:
                try:
                    text_data = uploaded_file.getvalue().decode('utf-8')
                except:
                    text_data = uploaded_file.getvalue().decode('cp949', errors='replace')
                contents.append(f"\n\n--- [참고자료: {uploaded_file.name}] ---\n{text_data}\n-------------------\n")
            elif file_extension == '.pdf':
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_paths.append(tmp_file.name)
        try:
            gemini_files = []
            for path in temp_file_paths:
                uploaded_gemini_file = genai.upload_file(path)
                gemini_files.append(uploaded_gemini_file)
            contents.extend(gemini_files)
            response = model.generate_content(contents)
            return response.text
        finally:
            for path in temp_file_paths:
                if os.path.exists(path):
                    os.remove(path)
    else:
        response = model.generate_content(contents)
        return response.text



# 백그라운드에서 과거 RFP 파일들을 미리 로딩
reference_rfps_dict = load_reference_rfps()

uploaded_files = st.file_uploader(
    "기존 제안요청서나 요구사항 정의서를 업로드하세요 (초안 생성 참고용)", 
    type=['pdf', 'txt', 'csv'], 
    accept_multiple_files=True
)

section_choice = st.radio(
    "집중적으로 작성할 파트를 선택하세요:",
    ["전체 요약본 (짧게)", "1~2장. 사업 개요 및 현황", "3장. 요구사항 상세", "4장. 제안 안내 및 평가 기준"],
    horizontal=True
)

user_input = st.text_area("이번 사업의 핵심 요구사항이나 특별히 강조할 내용을 입력하세요.", height=100)

if st.button(f"초안 생성 및 과거 유사 사업 탐색", type="primary"):
    if not API_KEY:
        st.error("API 키를 코드에 입력해 주세요.")
    elif not user_input:
        st.warning("작성할 내용을 입력해 주세요.")
    else:
        with st.spinner("AI가 초안을 작성하고, 사내 과거 사업 DB와 유사도를 비교 중입니다..."):
            try:
                SYSTEM_PROMPT = f"""
                당신은 주택도시보증공사(HUG)의 관리직 직원입니다.
                주어진 과업을 추진하기 위해 제안요청서 초안을 작성해야 합니다.

                [작성 규칙]
                1. 반드시 명확하고 간결한 '개조식(~함, ~임)'으로 작성할 것.
                2. 예산, 일정, 요구사항 등 수치나 명확한 팩트가 있다면 표 형식으로 정리할 것.
                3. 규정에 부합하는 용어와 공공기관 행정 용어를 적절히 사용할 것.
                
                [HUG 표준 목차 (참고용 배경지식)]
                1. 사업 개요 / 2. 공사업무 현황 / 3. 사업 추진방안 / 4. 요구사항 상세 / 5. 제안서 작성요령 / 6. 제안 안내사항

                [🚨 절대 준수 지시사항 🚨]
                전체 목차를 모두 작성해서는 절대 안 됩니다. 
                오직 사용자가 선택한 **[{section_choice}]** 파트 하나만 집중적으로 매우 상세하고 길게 작성하십시오.
                선택된 파트 이외의 다른 목차는 절대 출력하지 마십시오.
                표(Table) 작성 시 셀 내부에 줄바꿈 기호(\n)나 `<br>` 등의 HTML 태그를 절대 사용하지 마십시오. 여러 항목을 나열할 때는 줄바꿈 없이 쉼표(,)나 마침표(.)로만 이어 쓰십시오.
                또한 시스템 구축, 정보화 사업, 시스템 개발, 정보화사업 컨설팅 및 감리용역, 개인정보보호영향평가 등 ICT 사업에만 요구사항 명세를 COR-00, DAR-00 등 SW 가이드라인에 맞추어 작성하십시오.
                """
                final_prompt = f"{SYSTEM_PROMPT}\n\n[이번 사업 핵심 요청사항]\n{user_input}"

                result_text = generate_draft(API_KEY, final_prompt, uploaded_files)
                
                st.markdown("### 🔍 HUG 과거 유사 사업 Top 5")
                st.caption(f"총 {len(reference_rfps_dict)}개의 과거 데이터 기반으로 현재 구상 중인 사업과 가장 유사한 레퍼런스를 추천합니다.")
                
                if reference_rfps_dict:
                    analysis_query = user_input + "\n" + result_text
                    similarity_df = get_top_5_similar_rfps(analysis_query, reference_rfps_dict)
                    
                    if similarity_df is not None and not similarity_df.empty:
                        # 화면에는 '원본파일명' 컬럼을 숨기고 깔끔하게 표출
                        st.dataframe(similarity_df[['유사 과거 사업명', '유사도(%)']], use_container_width=True)
                        
                        # 유사도 결과 표 바로 아래에 원본 PDF 다운로드 버튼 생성
                        st.markdown("#### 📥 레퍼런스 원본 파일 다운로드")
                        for idx, row in similarity_df.iterrows():
                            file_name = row['원본파일명']
                            file_path = os.path.join("reference_rfps", file_name)
                            
                            if os.path.exists(file_path):
                                with open(file_path, "rb") as pdf_file:
                                    st.download_button(
                                        label=f"📄 {file_name} 다운로드",
                                        data=pdf_file,
                                        file_name=file_name,
                                        mime="application/pdf",
                                        key=f"download_pdf_{idx}" # 버튼 식별을 위한 고유 키
                                    )
                    else:
                        st.warning("유사도 분석에 실패했습니다.")
                else:
                    st.error("⚠️ 폴더를 찾을 수 없거나 PDF 파일이 없습니다.")
                
                st.divider() # 가로 구분
                
                # 워드 파일 생성 로직
                doc = Document()
                doc.add_heading(f"제안요청서 - {section_choice}", 0)
                doc.add_paragraph(result_text)
                
                bio = io.BytesIO()
                doc.save(bio)
                bio.seek(0)
                
                col_title, col_btn = st.columns([4, 1])
                with col_title:
                    st.markdown(f"### 📄 생성된 초안 ({section_choice})")
                with col_btn:
                    st.download_button(
                        label="💾 워드(.docx)로 다운로드", 
                        data=bio, 
                        file_name=f"제안요청서_{section_choice[:2]}.docx", 
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                        use_container_width=True
                    )
                
                st.info(result_text)
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

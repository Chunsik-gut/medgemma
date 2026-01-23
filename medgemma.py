import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore")

# 페이지 설정
st.set_page_config(
    page_title="[필로소피 AI 교육] MedGemma 의료 이미지 분석기",
    page_icon="🏥",
    layout="wide"
)

# 제목
st.title("🏥 [필로소피 AI 교육] MedGemma 의료 이미지 분석기")
st.markdown("---")

# CSS 스타일 추가
st.markdown("""
<style>
.result-box {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #1f77b4;
    color: #2c3e50;
    font-size: 16px;
    line-height: 1.8;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 20px 0;
}

.result-box h3 {
    color: #1f77b4;
    margin-top: 0;
}

.result-box p {
    margin-bottom: 15px;
}

.result-box ul, .result-box ol {
    margin-left: 20px;
}

.result-box strong {
    color: #2c3e50;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# 경고 메시지
st.error("""
⚠️ **중요 안전 공지**
- 이 앱은 교육 및 연구 목적으로만 사용되어야 합니다
- 실제 의료 진단이나 치료 결정에 사용하지 마세요
- 모든 결과는 의료 전문가의 검증이 필요합니다
""")

# Gated Model 설정 안내
with st.expander("🔑 Gated Model 설정 가이드", expanded=True):
    st.markdown("""
    ### 🚨 중요: MedGemma는 Gated Model입니다!
    
    **🔒 Gated Model이란?**
    - 제한된 접근 권한을 가진 특수 모델
    - 의료 AI의 책임감 있는 사용을 위해 인증 절차 필요
    - 교육 및 연구 목적으로만 사용 허용
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 1️⃣ 계정 준비
        - **Hugging Face 가입**: https://huggingface.co/join
        - 이메일 인증 완료
        - 프로필 정보 입력
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ 모델 접근 요청
        - **MedGemma 페이지 방문**: https://huggingface.co/google/medgemma-4b-it
        - **"Request access"** 버튼 클릭
        - 사용 목적 및 약관 동의
        - **즉시 승인** (보통 몇 분 내)
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ 토큰 생성 및 사용
        - **토큰 생성**: https://huggingface.co/settings/tokens
        - **권한**: "Read" 선택
        - **토큰 복사**하여 왼쪽 사이드바에 입력
        - **모델 사용 시작** 🎉
        """)
    
    st.success("""
    ✅ **토큰을 한 번만 설정하면 계속 사용 가능합니다!**
    
    💡 **보안 팁**: 토큰은 비밀번호처럼 관리하세요. 다른 사람과 공유하지 마세요.
    """)

# MedGemma 모델 소개 섹션
with st.expander("📚 MedGemma 모델에 대해 알아보기", expanded=True):  # 중요하므로 기본으로 열어둠
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🤖 MedGemma란 무엇인가요?
        
        **MedGemma**는 Google에서 개발한 **의료 전문 AI 모델**입니다. 일반적인 AI와 달리 의료 텍스트와 의료 이미지를 이해하고 분석하는 데 특화되어 있습니다.
        
        ### ✨ 주요 특징
        
        **🎯 의료 전문화**
        - 흉부 X-ray, 피부과 이미지, 안과 검사, 조직병리학 슬라이드 분석 특화
        - 의료 텍스트와 이미지를 동시에 처리하는 멀티모달 능력
        - 의료 질문-답변, 의료 보고서 생성 등 다양한 의료 업무 지원
        
        **📊 우수한 성능**
        - 흉부 X-ray 분석: 일반 AI 81.2% → MedGemma 88.9%
        - 피부과 진단: 일반 AI 52.5% → MedGemma 71.8%
        - 안과 검사: 일반 AI 14.4% → MedGemma 64.9%
        - 의학 시험 문제: 일반 AI 50.7% → MedGemma 64.4%
        
        **🔍 3가지 버전**
        - **MedGemma 4B**: 가벼운 버전 (텍스트 + 이미지)
        - **MedGemma 27B (텍스트)**: 고성능 텍스트 전용
        - **MedGemma 27B (멀티모달)**: 최고성능 (전자의무기록 포함)
        
        **🔒 Gated Model 안내**
        - MedGemma는 의료 전문 모델로 **제한된 접근** 권한을 가집니다
        - **교육 및 연구 목적**으로만 사용 가능
        - Hugging Face 계정과 **토큰 인증** 필요
        - Google의 **Health AI Developer Foundations** 약관 동의 필수
        """)
    
    with col2:
        st.markdown("""
        ### 📈 성능 비교
        """)
        
        # 성능 데이터 시각화
        import pandas as pd
        
        performance_data = {
            '작업': ['흉부 X-ray', '피부과 진단', '안과 검사', '의학 문제'],
            '일반 AI': [81.2, 52.5, 14.4, 50.7],
            'MedGemma': [88.9, 71.8, 64.9, 64.4]
        }
        
        df = pd.DataFrame(performance_data)
        st.bar_chart(df.set_index('작업'))
        
        st.info("""
        **💡 해석**
        - 모든 의료 분야에서 일반 AI 대비 현저한 성능 향상
        - 특히 안과 검사에서 350% 이상 성능 개선
        """)

# 학습 데이터 및 안전성 섹션
with st.expander("📖 학습 데이터 및 안전성", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📚 학습에 사용된 데이터
        
        **🔓 공개 데이터셋**
        - **MIMIC-CXR**: 흉부 X-ray와 의료 보고서
        - **SLAKE**: 의료 이미지 질문-답변 데이터
        - **TCGA**: 암 유전체 데이터
        - **PMC-OA**: 의학 논문과 이미지
        - **PAD-UFES-20**: 피부 병변 이미지
        
        **🔒 비공개 데이터셋** (개인정보 완전 제거)
        - 미국 방사선과 진료소의 CT 스캔
        - 콜롬비아 원격피부과 데이터
        - 호주 피부암 이미지
        - 유럽 조직병리학 데이터
        - 안과 검사 이미지 (당뇨병성 망막병증)
        """)
    
    with col2:
        st.markdown("""
        ### 🛡️ 안전성 검증
        
        **✅ 실시한 안전성 검사**
        - **아동 안전 검사**: 아동 관련 부적절한 콘텐츠 차단
        - **콘텐츠 안전성**: 폭력, 혐오 발언, 괴롭힘 방지
        - **편견 및 차별 방지**: 성별, 인종, 나이 등 편견 제거
        - **의료 정보 정확성**: 잘못된 의료 정보 제공 방지
        
        **🔍 평가 방법**
        - 구조화된 평가 및 내부 레드팀 테스트
        - 다양한 팀의 다각도 검증
        - 책임감 있는 AI 거버넌스 위원회 검토
        
        **✨ 결과**
        - 모든 안전성 카테고리에서 안전한 수준 달성
        - 정책 위반 사례 최소화 확인
        """)

# 기술적 세부사항 섹션
with st.expander("⚙️ 기술적 세부사항", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🏗️ 모델 아키텍처
        
        **기본 구조**
        - **베이스 모델**: Gemma 3 기반
        - **아키텍처**: Decoder-only Transformer
        - **어텐션 메커니즘**: Grouped-query attention (GQA)
        - **컨텍스트 길이**: 최소 128K 토큰 지원
        
        **이미지 처리**
        - **이미지 인코더**: SigLIP (의료 데이터로 사전 훈련)
        - **해상도**: 896 x 896로 정규화
        - **토큰화**: 이미지당 256토큰으로 인코딩
        
        **입출력**
        - **입력**: 텍스트 + 이미지 (총 128K 토큰)
        - **출력**: 텍스트 (최대 8192 토큰)
        """)
    
    with col2:
        st.markdown("""
        ### 💻 시스템 요구사항
        
        **하드웨어**
        - **GPU 메모리**: 
          - 4B 모델: 최소 8GB VRAM
          - 27B 모델: 24GB+ VRAM
        - **시스템 RAM**: 16GB 이상 권장
        - **저장공간**: 10-50GB (모델별 상이)
        
        **소프트웨어**
        - **Python**: 3.8 이상
        - **PyTorch**: GPU 지원 버전
        - **Transformers**: 4.50.0 이상
        - **CUDA**: GPU 사용 시 필요
        
        **최적화 옵션**
        - **양자화**: 4bit/8bit로 메모리 사용량 감소
        - **그래디언트 체크포인팅**: 메모리 효율성 향상
        """)

# 사이드바 - 모델 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # Hugging Face 토큰 입력 (Gated Model용)
    st.subheader("🔑 Hugging Face 인증")
    hf_token = st.text_input(
        "HF 토큰 (필수)",
        type="password",
        help="MedGemma는 Gated Model입니다. https://huggingface.co/settings/tokens 에서 토큰을 생성하세요."
    )
    
    if not hf_token:
        st.error("🔒 MedGemma는 Gated Model입니다. Hugging Face 토큰이 필요합니다!")
        st.markdown("""
        **📝 토큰 생성 방법:**
        1. https://huggingface.co/join 에서 계정 생성
        2. https://huggingface.co/google/medgemma-4b-it 접속
        3. "Request access" 클릭하고 승인 대기
        4. https://huggingface.co/settings/tokens 에서 토큰 생성
        5. 위 필드에 토큰 입력
        """)
    else:
        st.success("✅ 토큰이 입력되었습니다!")
    
    st.markdown("---")
    
    # 모델 선택
    model_option = st.selectbox(
        "모델 선택",
        ["google/medgemma-4b-it"],
        help="현재 4B 모델만 지원 (GPU 메모리 제한)"
    )
    
    # GPU 사용 여부
    use_gpu = st.checkbox("GPU 사용", value=torch.cuda.is_available())
    
    st.markdown("---")
    
    # 시스템 정보
    st.subheader("💾 시스템 정보")
    st.write(f"**CUDA 사용 가능**: {'✅' if torch.cuda.is_available() else '❌'}")
    if torch.cuda.is_available():
        st.write(f"**GPU**: {torch.cuda.get_device_name()}")
        st.write(f"**VRAM**: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    
    st.markdown("---")
    st.info("""
    **💡 최적 사용 환경:**
    - GPU: RTX 3080 이상
    - VRAM: 12GB 이상
    - RAM: 32GB 이상
    
    **첫 실행 시:**
    - 모델 다운로드로 5-10분 소요
    - 안정적인 인터넷 연결 필요
    """)

# 메인 컨텐츠
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 이미지 업로드")
    
    # 파일 업로더
    uploaded_file = st.file_uploader(
        "의료 이미지를 업로드하세요",
        type=['png', 'jpg', 'jpeg'],
        help="지원 형식: PNG, JPG, JPEG"
    )
    
    # 샘플 이미지 버튼
    if st.button("📋 샘플 X-ray 이미지 사용"):
        st.info("샘플 이미지가 로드되었습니다")
        # 실제로는 샘플 이미지 URL을 사용
        uploaded_file = "sample"
    
    # 질문 입력
    st.header("❓ 질문 입력")
    question = st.text_area(
        "이미지에 대해 묻고 싶은 것을 입력하세요",
        placeholder="예: 이 X-ray에서 어떤 소견을 볼 수 있나요?",
        height=100
    )
    
    # 미리 정의된 질문들
    st.subheader("💡 추천 질문들")
    
    # 카테고리별 질문들
    question_categories = {
        "🔍 일반 분석": [
            "이 X-ray를 설명해주세요",
            "이 이미지에서 주목할 점은 무엇인가요?",
            "해부학적 구조를 설명해주세요"
        ],
        "🩺 임상 소견": [
            "비정상적인 소견이 있나요?",
            "염증의 징후가 보이나요?",
            "뼈 구조에 이상이 있나요?"
        ],
        "📋 보고서 작성": [
            "간단한 의료 보고서를 작성해주세요",
            "소견을 요약해주세요",
            "추가 검사가 필요한지 알려주세요"
        ]
    }
    
    for category, questions in question_categories.items():
        with st.expander(category):
            for i, q in enumerate(questions):
                if st.button(f"💬 {q}", key=f"preset_{category}_{i}"):
                    question = q
                    st.rerun()

with col2:
    st.header("🔍 분석 결과")
    
    if uploaded_file is not None and question and hf_token:
        try:
            # 모델 로딩 상태 표시
            with st.spinner("🤖 MedGemma 모델을 로딩중... (최초 실행시 수분 소요)"):
                @st.cache_resource
                def load_model(model_name, hf_token=None):
                    try:
                        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                        
                        # Gated model을 위한 토큰 설정
                        pipeline_kwargs = {
                            "model": model_name,
                            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
                            "device": device,
                        }
                        
                        # 토큰이 있으면 추가
                        if hf_token:
                            pipeline_kwargs["token"] = hf_token
                        
                        # Pipeline 방식으로 로드
                        pipe = pipeline("image-text-to-text", **pipeline_kwargs)
                        return pipe, device
                    except Exception as e:
                        st.error(f"모델 로딩 실패: {str(e)}")
                        if "gated" in str(e).lower() or "authentication" in str(e).lower():
                            st.error("🔒 Gated Model 오류: Hugging Face 토큰이 필요합니다!")
                        return None, None
                
                pipe, device = load_model(model_option, hf_token)
            
            if pipe is not None:
                # 이미지 처리
                if uploaded_file == "sample":
                    # 샘플 이미지의 경우
                    st.info("샘플 이미지를 사용합니다")
                    # 실제로는 샘플 이미지를 로드해야 함
                    image = None
                else:
                    # 업로드된 이미지 처리
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # 이미지 표시
                    st.image(image, caption="업로드된 이미지", use_container_width=True)
                
                # 분석 실행
                with st.spinner("🔬 이미지를 분석중... (AI가 의료 이미지를 해석하고 있습니다)"):
                    if image is not None:
                        # 메시지 구성
                        messages = [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": "당신은 전문 의료진입니다. 의료 이미지를 정확하고 상세하게 분석해주세요. 교육 목적임을 명시하고, 실제 진단이 아님을 강조해주세요."}]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": question},
                                    {"type": "image", "image": image}
                                ]
                            }
                        ]
                        
                        try:
                            # 추론 실행
                            output = pipe(text=messages, max_new_tokens=500)
                            result = output[0]["generated_text"][-1]["content"]
                            
                            # 결과 표시
                            st.success("✅ 분석 완료!")
                            st.markdown("### 📋 MedGemma 분석 결과")
                            
                            # 결과를 박스로 표시
                            st.markdown(f"""
                            <div class="result-box">
                                <h3>MedGemma 분석 결과</h3>
                                <p>{result}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 교육적 설명 추가
                            with st.expander("🎓 교육적 해석"):
                                st.markdown("""
                                **🔍 AI가 이미지를 분석하는 방법:**
                                1. **이미지 전처리**: 896x896 픽셀로 정규화
                                2. **특징 추출**: SigLIP 인코더로 의료적 특징 식별
                                3. **언어 모델 처리**: Transformer가 의료 지식과 결합하여 해석
                                4. **답변 생성**: 의료 전문 용어와 일반인도 이해할 수 있는 설명 제공
                                
                                **🎯 MedGemma의 강점:**
                                - 의료 데이터로 특별히 훈련된 전문 모델
                                - 이미지와 텍스트를 동시에 이해하는 멀티모달 능력
                                - 의료진 수준의 상세한 분석 제공
                                """)
                            
                            # 추가 정보
                            with st.expander("ℹ️ 기술적 세부사항"):
                                st.write(f"**사용된 모델**: {model_option}")
                                st.write(f"**처리 디바이스**: {device.upper()}")
                                st.write(f"**이미지 크기**: {image.size if image else 'N/A'}")
                                st.write(f"**분석 토큰 수**: ~500토큰")
                                st.write(f"**모델 파라미터**: 4B (40억 개)")
                            
                        except Exception as e:
                            st.error(f"분석 중 오류 발생: {str(e)}")
                            st.info("💡 **문제 해결 팁:**\n- GPU 메모리 부족: 브라우저 새로고침 후 재시도\n- 네트워크 오류: 인터넷 연결 확인\n- 이미지 문제: 다른 형식의 이미지 시도")
                    
                    else:
                        st.warning("이미지를 먼저 업로드해주세요")
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            st.info("💡 **일반적인 해결법:**\n- GPU 메모리 부족 시 CPU 모드로 전환\n- 첫 실행 시 모델 다운로드로 시간 소요\n- 안정적인 인터넷 연결 필요")
    
    elif uploaded_file is None:
        st.info("👆 이미지를 업로드하고 질문을 입력해주세요")
        
        # 사용 예시 표시
        st.markdown("""
        ### 🎯 사용 예시
        
        **1. 흉부 X-ray 분석**
        - 업로드: 흉부 X-ray 이미지
        - 질문: "이 X-ray에서 폐렴의 징후가 보이나요?"
        
        **2. 피부 병변 검사**
        - 업로드: 피부 병변 사진
        - 질문: "이 피부 병변의 특징을 설명해주세요"
        
        **3. 의료 교육**
        - 업로드: 다양한 의료 이미지
        - 질문: "의대생에게 이 이미지를 어떻게 설명하시겠습니까?"
        """)
        
    elif not question:
        st.info("👆 질문을 입력해주세요")
    elif not hf_token:
        st.error("🔑 Hugging Face 토큰을 먼저 입력해주세요!")
        st.markdown("""
        ### 🚨 MedGemma는 Gated Model입니다
        
        **필수 단계:**
        1. **계정 생성**: https://huggingface.co/join
        2. **모델 접근 요청**: https://huggingface.co/google/medgemma-4b-it 에서 "Request access"
        3. **승인 대기**: 보통 즉시 또는 몇 시간 내 승인
        4. **토큰 생성**: https://huggingface.co/settings/tokens 에서 "Read" 권한으로 생성
        5. **토큰 입력**: 왼쪽 사이드바의 "HF 토큰" 필드에 입력
        
        **🎓 교육 목적 사용 시 주의사항**
        - Google의 Health AI Developer Foundations 약관에 동의해야 함
        - 실제 진료 목적 사용 금지
        - 연구 및 교육 목적으로만 사용
        """)

# 하단 교육 정보
st.markdown("---")

# 교육적 가치 및 활용법
with st.expander("🎓 교육적 가치 및 AI 윤리", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌟 AI 의료 교육의 가치
        
        **👨‍🎓 의료진 교육**
        - AI와 협업하는 미래 의료 환경 체험
        - 다양한 케이스 스터디 효율적 학습
        - AI의 장단점 실제 체험
        
        **🔬 연구 활용**
        - 대용량 의료 이미지 분석 자동화
        - 연구 가설 생성 및 검증 지원
        - 임상 시험 데이터 사전 분석
        
        **🏥 임상 지원 도구**
        - 의료진 의사결정 보조 (최종 판단은 의료진)
        - 응급상황 초기 판단 지원
        - 원격 의료 서비스 향상
        """)
    
    with col2:
        st.markdown("""
        ### ⚖️ AI 윤리 및 책임감
        
        **🛡️ 환자 안전 최우선**
        - AI는 보조 도구, 최종 판단은 의료진
        - 모든 AI 결과는 의료진 검증 필수
        - 환자 개인정보 보호 철저히 준수
        
        **🔍 투명성과 설명가능성**
        - AI 결정 과정의 투명한 공개
        - 의료진이 이해할 수 있는 설명 제공
        - 불확실성과 한계 명확히 표시
        
        **🤝 인간-AI 협업**
        - AI가 인간을 대체하지 않고 보완
        - 의료진의 전문성 향상 지원
        - 환자-의료진 관계의 핵심 가치 유지
        """)

# 제한사항 및 향후 발전 방향
with st.expander("⚠️ 제한사항 및 향후 발전 방향", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚧 현재 제한사항
        
        **기술적 제한**
        - 단일 이미지만 분석 가능 (다중 이미지 X)
        - 대화형 상호작용 최적화 부족
        - 특정 의료 장비/병원 환경에 따른 성능 편차
        
        **데이터 제한**
        - 주로 영어 데이터로 훈련
        - 특정 인종/지역 편향 가능성
        - 희귀 질환에 대한 데이터 부족
        
        **윤리적 고려사항**
        - 의료진 감독 없는 독립 사용 불가
        - 법적 책임 문제
        - 환자 프라이버시 보호 필요
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 향후 발전 방향
        
        **성능 개선**
        - 다중 이미지 동시 분석 기능
        - 실시간 대화형 상호작용
        - 개인화된 의료 AI 어시스턴트
        
        **다양성 확대**
        - 다국어 지원 (한국어 포함)
        - 다양한 인종/지역 데이터 포함
        - 희귀 질환 데이터베이스 확장
        
        **통합 플랫폼**
        - 전자의무기록(EMR) 시스템 연동
        - 의료 장비와 실시간 연결
        - 글로벌 의료 지식 공유 플랫폼
        
        **안전성 강화**
        - 더 엄격한 검증 시스템
        - 실시간 모니터링 및 피드백
        - 의료진-AI 협업 프로토콜 표준화
        """)

# 사용법 및 주의사항
with st.expander("📚 사용법 및 주의사항", expanded=False):
    st.markdown("""
    ### 🔧 사용법
    1. **이미지 업로드**: 의료 이미지 (X-ray, CT, MRI 등)를 업로드
    2. **질문 입력**: 이미지에 대해 알고 싶은 내용을 구체적으로 작성
    3. **분석 실행**: MedGemma가 이미지를 분석하여 답변 제공
    4. **결과 검증**: 반드시 의료 전문가의 검증 과정 거치기
    
    ### ⚠️ 중요 주의사항
    
    **🚫 절대 금지사항**
    - 실제 환자 진단이나 치료 결정에 직접 사용
    - 의료진 상담 없이 AI 결과만으로 의료 결정
    - 응급상황에서 AI 결과에만 의존
    - 개인 의료정보의 무단 업로드
    
    **✅ 권장 사용법**
    - 의료 교육 및 연구 목적
    - 의료진의 보조 도구로 활용
    - AI 기술 이해를 위한 학습 도구
    - 가설 생성 및 추가 검사 방향 설정
    
    ### 🔧 기술적 제한사항
    - **하드웨어**: GPU 메모리 최소 8GB 필요
    - **네트워크**: 안정적인 인터넷 연결 (모델 다운로드)
    - **시간**: 첫 실행 시 5-10분 소요 (모델 로딩)
    - **지원 형식**: PNG, JPG, JPEG 이미지만 지원
    
    ### 💡 최적 사용 팁
    - **구체적인 질문**: "비정상 소견이 있나요?"보다 "폐렴의 징후가 보이나요?"
    - **맥락 제공**: 환자 연령, 증상 등 추가 정보 포함
    - **단계적 접근**: 전체적인 분석 후 세부 사항 질문
    - **비교 분석**: 여러 이미지를 순차적으로 분석하여 비교
    """)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <h4>🏥 [필로소피 AI 교육] MedGemma Streamlit App</h4>
    <p><strong>교육 및 연구 목적으로만 사용 | 실제 의료 진단 금지</strong></p>
    <p><em>Powered by Google MedGemma & Streamlit | AI 윤리와 책임감 있는 AI 사용을 지향합니다</em></p>
    
    <div style='margin-top: 20px; padding: 15px; background-color: #e8f4f8; border-radius: 10px; color: #1f1f1f;'>
        <p><strong>📞 필로소피 AI 교육 문의</strong></p>
        <p>AI 의료 기술 교육, 연구 협력, 윤리적 AI 개발에 관심이 있으시면 언제든 연락주세요.</p>
    </div>
</div>
""", unsafe_allow_html=True)

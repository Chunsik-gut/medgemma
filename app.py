import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import warnings
import os
import numpy as np
import re

try:
    import pydicom
    import pydicom.pixels
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

warnings.filterwarnings("ignore")

MODEL_ID = "google/medgemma-1.5-4b-it"

st.set_page_config(page_title="MedGemma 의료 챗봇", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stButton button { white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 MedGemma 로컬 의료 이미지 챗봇")
st.caption("NVIDIA GPU 가속 모드 | 교육 및 연구용")

if not DICOM_AVAILABLE:
    st.warning("⚠️ pydicom 미설치 — DICOM 파일 지원 안됨.")

# --- 세션 상태 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "selected_series" not in st.session_state:
    st.session_state.selected_series = None
if "compare_before" not in st.session_state:
    st.session_state.compare_before = None
if "compare_after" not in st.session_state:
    st.session_state.compare_after = None
if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = False

# --- DICOM 메타데이터 ---
def extract_dicom_metadata(ds):
    meta = {}
    fields = {
        "PatientName":       "환자명",
        "PatientAge":        "나이",
        "PatientSex":        "성별",
        "StudyDate":         "촬영일",
        "Modality":          "모달리티",
        "BodyPartExamined":  "촬영부위",
        "StudyDescription":  "검사명",
        "SeriesDescription": "시리즈명",
        "InstitutionName":   "기관명",
        "SliceThickness":    "슬라이스두께",
    }
    for tag, label in fields.items():
        try:
            val = getattr(ds, tag, None)
            if val is not None:
                if tag == "PatientSex":
                    val = {"M": "남성", "F": "여성"}.get(str(val), str(val))
                elif tag == "StudyDate" and len(str(val)) == 8:
                    v = str(val)
                    val = f"{v[:4]}-{v[4:6]}-{v[6:]}"
                meta[label] = str(val)
        except Exception:
            pass
    return meta

# --- DICOM → PIL Image ---
def dicom_to_image(ds):
    arr = pydicom.pixels.apply_rescale(ds.pixel_array, ds).astype(np.float32)
    modality = str(getattr(ds, 'Modality', '')).strip().upper()

    if modality == 'CT':
        def norm(ct, min_val, max_val):
            ct = np.clip(ct, min_val, max_val)
            ct = ct - min_val
            ct /= (max_val - min_val)
            ct *= 255.0
            return ct

        window_clips = [(-1024, 1024), (-135, 215), (0, 80)]
        arr = np.stack([norm(arr, c[0], c[1]) for c in window_clips], axis=-1)
        arr = np.round(arr).astype(np.uint8)
        return Image.fromarray(arr)
    else:
        try:
            wc = float(ds.WindowCenter) if hasattr(ds, 'WindowCenter') else None
            ww = float(ds.WindowWidth)  if hasattr(ds, 'WindowWidth')  else None
            if wc is not None and ww is not None:
                arr = np.clip(arr, wc - ww/2, wc + ww/2)
        except Exception:
            pass
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

# --- 이미지 로드 ---
def load_image(f):
    if f.name.lower().endswith('.dcm') and DICOM_AVAILABLE:
        ds = pydicom.dcmread(f)
        return dicom_to_image(ds), extract_dicom_metadata(ds)
    else:
        return Image.open(f).convert("RGB"), {}

# --- 균등 샘플링 (최대 85장) ---
def uniform_sample(files, max_slices=85):
    if len(files) <= max_slices:
        return files
    indices = [int(round(i / (max_slices - 1) * (len(files) - 1))) for i in range(max_slices)]
    return [files[i] for i in indices]

# --- Study > Series 계층 그룹화 ---
def group_files_by_study_series(files):
    studies = {}
    non_dicom = []

    for f in files:
        f.seek(0)
        if f.name.lower().endswith('.dcm') and DICOM_AVAILABLE:
            try:
                ds = pydicom.dcmread(f)
                study_uid    = str(getattr(ds, 'StudyInstanceUID',  '미확인')).strip() or '미확인'
                series_uid   = str(getattr(ds, 'SeriesInstanceUID', '미확인')).strip() or '미확인'
                patient_name = str(getattr(ds, 'PatientName',       '이름없음')).strip() or '이름없음'
                study_date   = str(getattr(ds, 'StudyDate',         '')).strip()
                study_desc   = str(getattr(ds, 'StudyDescription',  '')).strip()
                series_desc  = str(getattr(ds, 'SeriesDescription', '')).strip()
                series_num   = str(getattr(ds, 'SeriesNumber',      '')).strip()
                modality     = str(getattr(ds, 'Modality',          '')).strip().upper()
                body_part    = str(getattr(ds, 'BodyPartExamined',  '')).strip()

                if len(study_date) == 8:
                    study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"

                study_label_parts = [patient_name]
                if study_desc:
                    study_label_parts.append(study_desc)
                elif body_part:
                    study_label_parts.append(body_part)
                if study_date:
                    study_label_parts.append(study_date)
                study_label = " / ".join(study_label_parts)

                if series_num and series_desc:
                    series_label = f"Series {series_num}: {series_desc}"
                elif series_desc:
                    series_label = series_desc
                elif series_num:
                    series_label = f"Series {series_num}"
                else:
                    series_label = "Series"
                if modality:
                    series_label += f" ({modality})"

                if study_uid not in studies:
                    studies[study_uid] = {"label": study_label, "series": {}}

                if series_uid not in studies[study_uid]["series"]:
                    studies[study_uid]["series"][series_uid] = {
                        "label":    series_label,
                        "modality": modality,
                        "files":    []
                    }

                studies[study_uid]["series"][series_uid]["files"].append((f, int(getattr(ds, 'InstanceNumber', 0))))

            except Exception:
                non_dicom.append(f)
            f.seek(0)
        else:
            non_dicom.append(f)

    # InstanceNumber로 정렬 후 파일만 추출
    for study in studies.values():
        for series in study["series"].values():
            series["files"] = [f for f, _ in sorted(series["files"], key=lambda x: x[1])]

    if non_dicom:
        studies["__non_dicom__"] = {
            "label": "일반 이미지",
            "series": {
                "__non_dicom_series__": {
                    "label":    "일반 이미지",
                    "modality": "",
                    "files":    non_dicom
                }
            }
        }

    return studies

# --- 응답 파싱 ---
def parse_response(text):
    text = re.sub(r'<unused\d+>thought.*?</unused\d+>', '', text, flags=re.DOTALL)
    text = re.sub(r'<unused\d+>thought.*', '', text, flags=re.DOTALL)
    return text.strip()

# --- 모델 추론 ---
def run_model(messages, max_new_tokens=1024):
    processor = st.session_state.processor
    model = st.session_state.model

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        continue_final_message=False,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    )

    with torch.inference_mode():
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

    response = processor.post_process_image_text_to_text(generated, skip_special_tokens=True)
    decoded_input = processor.post_process_image_text_to_text(inputs["input_ids"], skip_special_tokens=True)
    result = response[0]
    index = result.find(decoded_input[0])
    if 0 <= index <= 2:
        result = result[index + len(decoded_input[0]):]
    return parse_response(result)

# --- 묶음 분석 ---
def analyze_batch(batch_files, prompt, system_text, modality="", body_part=""):
    images, first_meta = [], {}
    for i, f in enumerate(batch_files):
        f.seek(0)
        img, meta = load_image(f)
        images.append(img)
        if i == 0:
            first_meta = meta
        f.seek(0)

    model_input = []

    # instruction을 user 메시지 맨 앞에 (공식 노트북 방식)
    if modality == 'CT':
        body_part_text = f" of the {body_part.lower()}" if body_part else ""
        instruction = (
            f"You are an instructor teaching medical students. You are "
            f"analyzing a contiguous block of CT slices{body_part_text}. "
            f"Please review the slices provided below carefully.")
        model_input.append({"type": "text", "text": instruction})

    if first_meta:
        meta_text = " / ".join([f"{k}: {v}" for k, v in first_meta.items()])
        model_input.append({"type": "text", "text": f"Metadata: {meta_text}"})

    for i, img in enumerate(images):
        model_input.append({"type": "image", "image": img})
        model_input.append({"type": "text", "text": f"SLICE {i + 1}"})

    model_input.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user",   "content": model_input}
    ]

    max_tokens = max(512, min(256 + len(images) * 128, 2048))
    return run_model(messages, max_new_tokens=max_tokens)

# --- 모델 로드 ---
@st.cache_resource
def load_medgemma(hf_token):
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, token=hf_token, **model_kwargs)
    return processor, model

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")

    user_token = os.environ.get("HF_TOKEN", "")

    if user_token and not st.session_state.model_loaded:
        with st.spinner("모델 자동 로딩 중..."):
            try:
                processor, model = load_medgemma(user_token)
                st.session_state.processor = processor
                st.session_state.model = model
                st.session_state.model_loaded = True
            except Exception as e:
                st.error(f"자동 로드 실패: {str(e)}")

    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.error("GPU를 찾을 수 없습니다.")

    if st.button("모델 로드/재시작"):
        with st.spinner("로딩 중..."):
            try:
                processor, model = load_medgemma(user_token)
                st.session_state.processor = processor
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("완료!")
            except Exception as e:
                st.error(f"오류: {str(e)}")

    st.markdown("---")

    allowed_types = ['png', 'jpg', 'jpeg']
    if DICOM_AVAILABLE:
        allowed_types.append('dcm')

    uploader_key = "uploader_" + str(st.session_state.get("upload_version", 0))
    uploaded_files = st.file_uploader(
        "이미지 업로드",
        type=allowed_types,
        accept_multiple_files=True,
        key=uploader_key
    )

    active_files = []
    active_modality = ""

    if uploaded_files:
        st.info(f"{len(uploaded_files)}장 준비됨")
        if st.button("🗑️ 전체 삭제"):
            st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1
            st.session_state.selected_series = None
            st.rerun()

        has_dicom = any(f.name.lower().endswith('.dcm') for f in uploaded_files)

        if has_dicom and DICOM_AVAILABLE:
            st.markdown("---")
            studies = group_files_by_study_series(uploaded_files)

            # 비교 모드 토글
            compare_mode = st.toggle("🔀 비교 판독 모드", value=st.session_state.compare_mode)
            if compare_mode != st.session_state.compare_mode:
                st.session_state.compare_mode = compare_mode
                st.session_state.compare_before = None
                st.session_state.compare_after = None
                st.session_state.selected_series = None
                st.rerun()

            for study_uid, study in studies.items():
                with st.expander(f"📁 {study['label']}", expanded=True):
                    for series_uid, series in study["series"].items():
                        count = len(series["files"])
                        if st.session_state.compare_mode:
                            col1, col2 = st.columns(2)
                            with col1:
                                is_before = st.session_state.compare_before == series_uid
                                if st.button(f"{'✅ 이전' if is_before else '이전 CT'}", key=f"before_{series_uid}"):
                                    st.session_state.compare_before = series_uid
                                    st.session_state.messages = []
                                    st.rerun()
                            with col2:
                                is_after = st.session_state.compare_after == series_uid
                                if st.button(f"{'✅ 이후' if is_after else '이후 CT'}", key=f"after_{series_uid}"):
                                    st.session_state.compare_after = series_uid
                                    st.session_state.messages = []
                                    st.rerun()
                            st.caption(f"{series['label']} ({count}장)")
                        else:
                            is_selected = st.session_state.selected_series == series_uid
                            btn_label = f"{'✅' if is_selected else '📋'} {series['label']} ({count}장)"
                            if st.button(btn_label, key=f"series_{series_uid}"):
                                st.session_state.selected_series = series_uid
                                st.session_state.messages = []
                                st.rerun()

            # 시리즈 데이터 찾기
            all_series = {s_uid: s for st_uid, st_data in studies.items() for s_uid, s in st_data["series"].items()}

            selected_series_data = all_series.get(st.session_state.selected_series)
            before_series_data = all_series.get(st.session_state.compare_before)
            after_series_data = all_series.get(st.session_state.compare_after)

            if st.session_state.compare_mode:
                st.markdown("---")
                if before_series_data:
                    st.caption(f"이전 CT: {before_series_data['label']} ({len(before_series_data['files'])}장)")
                else:
                    st.caption("이전 CT: 미선택")
                if after_series_data:
                    st.caption(f"이후 CT: {after_series_data['label']} ({len(after_series_data['files'])}장)")
                else:
                    st.caption("이후 CT: 미선택")

                if before_series_data and after_series_data:
                    active_files = before_series_data["files"]
                    active_modality = before_series_data["modality"]
                    st.session_state.compare_before_files = before_series_data["files"]
                    st.session_state.compare_after_files = after_series_data["files"]
                    st.session_state.compare_before_modality = before_series_data["modality"]

            elif selected_series_data:
                active_files    = selected_series_data["files"]
                active_modality = selected_series_data["modality"]
                st.markdown("---")
                st.caption(f"선택: {selected_series_data['label']}")
                st.caption(f"총 {len(active_files)}장 → 균등 샘플링 85장")

                for f in active_files[:3]:
                    f.seek(0)
                    try:
                        ds = pydicom.dcmread(f)
                        st.image(dicom_to_image(ds), width=150, caption=f.name)
                    except Exception:
                        st.text(f.name)
                    f.seek(0)

        else:
            active_files = uploaded_files
            for f in uploaded_files[:3]:
                f.seek(0)
                try:
                    st.image(f, width=150)
                except Exception:
                    st.text(f.name)
                f.seek(0)
            if len(uploaded_files) > 3:
                st.caption(f"외 {len(uploaded_files) - 3}장")

    if st.button("대화 기록 삭제"):
        st.session_state.messages = []
        st.rerun()

# --- 채팅 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    wrapped_prompt = f"Please analyze the uploaded medical images. User's question: {prompt}"

    if not st.session_state.model_loaded:
        st.warning("사이드바에서 모델 로드 버튼을 눌러주세요.")
    else:
        with st.chat_message("assistant"):
            try:
                if st.session_state.get("compare_mode") and st.session_state.get("compare_before_files") and st.session_state.get("compare_after_files"):
                    before_files = st.session_state.compare_before_files
                    after_files = st.session_state.compare_after_files
                    modality = st.session_state.compare_before_modality

                    modality_map = {
                        "CT": "CT 영상의학과",
                        "MR": "MRI 영상의학과",
                        "CR": "흉부 X-ray 영상의학과",
                        "DX": "X-ray 영상의학과",
                        "OP": "안과",
                        "US": "초음파 영상의학과",
                        "NM": "핵의학과",
                        "PT": "PET 핵의학과",
                    }
                    specialty = modality_map.get(modality, "영상의학과")
                    system_text = f"You are a Korean {specialty} specialist. Respond in Korean."

                    body_part = ""
                    if modality == "CT":
                        try:
                            before_files[0].seek(0)
                            ds_tmp = pydicom.dcmread(before_files[0])
                            body_part = str(getattr(ds_tmp, "BodyPartExamined", "")).strip().upper()
                            before_files[0].seek(0)
                        except Exception:
                            pass

                    before_sampled = uniform_sample(before_files, max_slices=42)
                    after_sampled = uniform_sample(after_files, max_slices=43)

                    compare_prompt = (
                        f"The first 42 slices (SLICE 1-42) are pre-operative/previous CT images. "
                        f"The next 43 slices (SLICE 43-85) are post-operative/follow-up CT images. "
                        f"Please compare the two and describe any changes, improvements, or new findings. "
                        f"User's question: {prompt}\nRespond in Korean."
                    )

                    with st.spinner(f"비교 판독 중... (이전 {len(before_sampled)}장 + 이후 {len(after_sampled)}장)"):
                        response = analyze_batch(
                            before_sampled + after_sampled, compare_prompt, system_text,
                            modality=modality, body_part=body_part
                        )

                elif active_files:
                    modality_map = {
                        "CT": "CT 영상의학과",
                        "MR": "MRI 영상의학과",
                        "CR": "흉부 X-ray 영상의학과",
                        "DX": "X-ray 영상의학과",
                        "OP": "안과",
                        "US": "초음파 영상의학과",
                        "NM": "핵의학과",
                        "PT": "PET 핵의학과",
                    }
                    specialty = modality_map.get(active_modality, "영상의학과")
                    system_text = (
                        f"You are a Korean {specialty} specialist."
                    )

                    # body_part 추출
                    body_part = ""
                    if active_files and active_modality == 'CT':
                        try:
                            active_files[0].seek(0)
                            ds_tmp = pydicom.dcmread(active_files[0])
                            body_part = str(getattr(ds_tmp, 'BodyPartExamined', '')).strip().upper()
                            active_files[0].seek(0)
                        except Exception:
                            pass

                    sampled_files = uniform_sample(active_files)

                    with st.spinner(f"분석 중... ({len(sampled_files)}장 / 전체 {len(active_files)}장)"):
                        response = analyze_batch(
                            sampled_files, wrapped_prompt, system_text,
                            modality=active_modality, body_part=body_part
                        )

                    first_file = active_files[0]
                    first_file.seek(0)
                    _, meta = load_image(first_file)
                    first_file.seek(0)
                    if meta:
                        with st.expander("📋 환자 메타데이터"):
                            for k, v in meta.items():
                                st.markdown(f"**{k}**: {v}")

                else:
                    with st.spinner("응답 생성 중..."):
                        messages = [
                            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                            {"role": "user",   "content": [{"type": "text", "text": prompt}]}
                        ]
                        response = run_model(messages, max_new_tokens=512)

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

st.markdown("---")
st.caption("주의: 본 결과는 AI가 생성한 것으로 실제 의사의 진단을 대체할 수 없습니다.")
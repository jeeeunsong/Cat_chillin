import os
import streamlit as st
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import tempfile
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import queue
import threading
import av
import mediapipe as mp

# 하드코딩된 모델 경로
model_path = "./assets/model_v0.01"

# 고양이 소리 클래스 정의
cat_sound_classes = [
    "그르렁 (만족/행복)",
    "하악 (위협/경고)"
]

# MediaPipe Audio Classifier 설정
BaseOptions = mp.tasks.BaseOptions
AudioClassifier = mp.tasks.audio.AudioClassifier
AudioClassifierOptions = mp.tasks.audio.AudioClassifierOptions
AudioRunningMode = mp.tasks.audio.RunningMode
AudioData = mp.tasks.components.containers.AudioData

# 결과를 저장할 전역 변수
classification_result = None

# 결과 콜백 함수
def result_callback(result, output_timestamp):
    global classification_result
    classification_result = result

# 오디오 분류기 초기화
options = AudioClassifierOptions(
    base_options=BaseOptions(model_asset_path='./assets/yamnet.tflite'),
    running_mode=AudioRunningMode.AUDIO_STREAM,
    max_results=1,
    score_threshold=0.5,
    category_allowlist=['Speech', 'Music', 'Animal sounds'],
    result_callback=result_callback
)

classifier = AudioClassifier.create_from_options(options)

# 모델 로드 함수 수정 - 위젯 없음
@st.cache_resource
def load_model(model_path, num_labels):
    """학습된 모델 로드 및 캐싱 - 위젯 없음"""
    error_msg = None
    try:
        # 특성 추출기 로드
        try:
            # 먼저 모델 경로에서 특성 추출기 로드 시도
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        except:
            # 실패하면 기본 모델에서 특성 추출기 로드
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
        # 모델 로드
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        return feature_extractor, model, error_msg
    except Exception as e:
        import traceback
        error_msg = f"모델 로드 중 오류 발생: {e}\n{traceback.format_exc()}"
        return None, None, error_msg

# 모델 로드하기
with st.spinner("모델 로딩 중... 잠시만 기다려주세요."):
    feature_extractor, model, error_msg = load_model(model_path, len(cat_sound_classes))
    if feature_extractor and model:
        st.success("모델이 성공적으로 로드되었습니다!")
    else:
        st.warning("모델 로드에 실패했습니다. 네트워크 연결을 확인하고 다시 시도해주세요.")
        if error_msg:
            st.error(error_msg)

# 오디오 파일 예측 함수
def predict_audio_file(audio_file, feature_extractor, model):
    """오디오 파일에서 예측하기"""
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        # 오디오 로드 및 전처리
        audio_data, sample_rate = librosa.load(tmp_path, sr=16000)
        os.unlink(tmp_path)  # 임시 파일 삭제
        
        # 파형 시각화
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(np.linspace(0, len(audio_data)/sample_rate, len(audio_data)), audio_data)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        st.pyplot(fig)
        
        # 특성 추출
        inputs = feature_extractor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence_scores = predictions[0].tolist()
        
        # 결과 표시
        st.subheader("분석 결과:")
        
        # 예측된 클래스
        st.markdown(f"### 감지된 소리: **{cat_sound_classes[predicted_class]}**")
        
        # 신뢰도 점수
        st.subheader("클래스별 신뢰도:")
        for i, cls in enumerate(cat_sound_classes):
            score = confidence_scores[i] * 100
            st.progress(int(score))
            st.write(f"{cls}: {score:.2f}%")
            
        return predicted_class, confidence_scores
        
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
        return None, None

# 실시간 오디오 처리를 위한 클래스
class AudioProcessor:
    def __init__(self, feature_extractor, model):
        self.feature_extractor = feature_extractor
        self.model = model
        self.result_queue = queue.Queue()
        self.audio_buffer = []
        self.sample_rate = 16000
        self.last_process_time = time.time()
        self.is_speech = False
        
    def process_audio(self, frame):
        """오디오 프레임 처리"""
        try:
            # 오디오 데이터 변환
            sound = frame.to_ndarray()
            sound = sound.reshape(-1)
            
            # 16-bit PCM에서 float32로 변환
            if sound.dtype == np.int16:
                sound = sound.astype(np.float32) / 32768.0
            
            # MediaPipe AudioData 생성
            audio_data = AudioData.create_from_array(sound, self.sample_rate)
            
            # 오디오 분류 수행
            classifier.classify_async(audio_data, int(time.time() * 1000))
            
            # 분류 결과 처리
            if classifier.has_result():
                result = classifier.get_result()
                if result and result.classifications:
                    for classification in result.classifications:
                        for category in classification.categories:
                            if category.category_name in ['Speech', 'Music', 'Animal sounds'] and category.score > 0.5:
                                self.is_speech = True
                                # 버퍼에 추가
                                self.audio_buffer.extend(sound.tolist())
                                
                                # 2초마다 분석 (과도한 처리 방지를 위한 시간 확인 추가)
                                current_time = time.time()
                                buffer_duration = len(self.audio_buffer) / self.sample_rate
                                
                                if buffer_duration >= 2.0 and (current_time - self.last_process_time) >= 1.0:
                                    self.last_process_time = current_time
                                    
                                    # 버퍼에서 최신 2초 오디오 가져오기
                                    audio_data = np.array(self.audio_buffer[-self.sample_rate*2:])
                                    
                                    # 필요한 경우 오디오 정규화
                                    if np.max(np.abs(audio_data)) > 0:
                                        audio_data = audio_data / np.max(np.abs(audio_data))
                                    
                                    # 특성 추출
                                    try:
                                        inputs = self.feature_extractor(audio_data, sampling_rate=self.sample_rate, 
                                                                      return_tensors="pt", padding=True)
                                        
                                        # 예측
                                        with torch.no_grad():
                                            outputs = self.model(**inputs)
                                            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                                            predicted_class = torch.argmax(predictions, dim=-1).item()
                                            confidence_scores = predictions[0].tolist()
                                        
                                        # 결과 큐에 추가
                                        self.result_queue.put((predicted_class, confidence_scores))
                                    except Exception as e:
                                        print(f"모델 예측 중 오류: {e}")
                                    
                                    # 버퍼 크기 유지 (마지막 3초 보존)
                                    self.audio_buffer = self.audio_buffer[-self.sample_rate*3:]
                                break
                            else:
                                self.is_speech = False
                                self.audio_buffer = []  # 음성이 아닐 때는 버퍼 초기화
                
        except Exception as e:
            print(f"오디오 처리 중 오류: {e}")
        
        return frame

# 탭 설정
tab1, tab2 = st.tabs(["파일 업로드", "실시간 분석"])

with tab1:
    st.header("오디오 파일 분석")
    uploaded_file = st.file_uploader("고양이 소리 파일 업로드", type=['wav', 'mp3'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("분석 시작"):
            if feature_extractor and model:
                predict_audio_file(uploaded_file, feature_extractor, model)
            else:
                st.warning("모델이 로드되지 않았습니다.")

with tab2:
    st.header("실시간 오디오 분석")
    st.write("마이크로 고양이 소리를 녹음하여 실시간으로 분석합니다.")
    
    if feature_extractor and model:
        processor = AudioProcessor(feature_extractor, model)
        
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        webrtc_ctx = webrtc_streamer(
            key="cat-sound",
            mode=WebRtcMode.SENDRECV,
            audio_frame_callback=processor.process_audio,
            media_stream_constraints={"video": False, "audio": True},
            audio_receiver_size=1024,
            async_processing=True,
            rtc_configuration=rtc_configuration
        )
        
        if webrtc_ctx.state.playing:
            st.write("실시간 분석 중... 고양이 소리를 들려주세요!")
            
            result_placeholder = st.empty()
            confidence_placeholders = [st.empty() for _ in range(len(cat_sound_classes))]
            vad_status = st.empty()
            
            while webrtc_ctx.state.playing:
                # VAD 상태 표시
                vad_status.markdown(f"### 음성 감지 상태: {'감지됨' if processor.is_speech else '감지되지 않음'}")
                
                if not processor.result_queue.empty():
                    predicted_class, confidence_scores = processor.result_queue.get()
                    
                    # 결과 표시
                    result_placeholder.markdown(f"### 감지된 소리: **{cat_sound_classes[predicted_class]}**")
                    
                    # 신뢰도 표시
                    for i, placeholder in enumerate(confidence_placeholders):
                        score = confidence_scores[i] * 100
                        placeholder.progress(int(score))
                        placeholder.write(f"{cat_sound_classes[i]}: {score:.2f}%")
                
                time.sleep(0.1)
    else:
        st.warning("모델이 로드되지 않았습니다.")

# 앱 정보
st.sidebar.markdown("---")
st.sidebar.subheader("앱 정보")
st.sidebar.info("""
이 애플리케이션은 Hugging Face의 wav2vec 모델을 사용하여 고양이 음성을 분류합니다.
실제 사용을 위해서는 고양이 소리 데이터셋으로 fine-tuning된 모델이 필요합니다.
""")

# 설치 필요 라이브러리
# pip install streamlit torch transformers librosa soundfile matplotlib pydub streamlit-webrtc
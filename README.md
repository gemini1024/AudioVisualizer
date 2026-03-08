# AudioVisualizer

오디오 파일을 FFT 스펙트럼 시각화 MP4 영상으로 변환하는 CLI 도구.

## 특징

- **FFT 스펙트럼**: 64개 멜 스케일 주파수 대역 막대 그래프
- **미러 레이아웃**: 화면 중앙을 기준으로 좌우 대칭, 저음이 중앙 / 고음이 양 끝
- **EMA 스무딩**: 프레임 간 지수이동평균으로 부드러운 움직임
- **글로우 이펙트**: 다운샘플 블러 후 additive compositing
- **고속 렌더링**: PIL + multiprocessing + ffmpeg stdin 파이프 (~6-7x vs matplotlib)
- **광범위한 포맷 지원**: ffmpeg 기반 디코딩 (m4a, mp3, wav, flac, aac 등)

## 요구사항

**Python 패키지**

```
librosa
numpy
Pillow
```

```bash
pip install librosa numpy Pillow
```

**시스템 도구** (별도 설치)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
apt install ffmpeg
```

## 사용법

```bash
python3 visualizer.py input.m4a
python3 visualizer.py input.m4a -o output.mp4
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `input` | — | 입력 오디오 파일 |
| `-o`, `--output` | `<input>_visualized.mp4` | 출력 MP4 경로 |
| `--fps` | `30` | 프레임 레이트 |
| `--width` | `1920` | 영상 너비 (px) |
| `--height` | `1080` | 영상 높이 (px) |
| `--color` | 블루→핑크 그라디언트 | 기준 색상 hex (예: `#FF6B6B`) |
| `--bg` | `#0A0A0A` | 배경색 hex |
| `--bars` | `64` | 주파수 막대 수 |
| `--decay` | `0.75` | EMA 스무딩 강도 (0~1, 클수록 잔상 길어짐) |
| `--workers` | `cpu_count - 1` | 렌더링 워커 프로세스 수 |
| `--preview` | off | 첫 15초만 렌더링 (빠른 확인용) |

### 예시

```bash
# 기본 렌더링
python3 visualizer.py song.m4a

# 60fps, 커스텀 색상
python3 visualizer.py song.m4a --fps 60 --color "#FF6B6B"

# 어두운 배경, 막대 수 조정
python3 visualizer.py song.wav --bars 128 --bg "#000000"

# 빠른 미리보기 (15초)
python3 visualizer.py song.m4a --preview

# 워커 수 지정
python3 visualizer.py song.m4a --workers 4
```

## 성능

3분 곡 기준 (1920×1080, 30fps, 5400프레임):

| 단계 | 시간 |
|---|---|
| FFT 사전 계산 (librosa) | ~5초 |
| PIL 프레임 렌더 (멀티프로세스) | ~22초 |
| ffmpeg 인코딩 (병렬) | ~18초 |
| **합계** | **~28초** |

## 동작 원리

```
ffmpeg 디코딩 (f32le mono)
    ↓
librosa.stft → 멜 필터뱅크 (64개 대역) → dB 정규화 → EMA 스무딩
    ↓
(n_frames × 64) float32 공유 메모리
    ↓
multiprocessing.Pool → PIL 프레임 렌더링 (글로우 + 미러 막대)
    ↓
ffmpeg stdin 파이프 → libx264 + aac → MP4
```

# AudioVisualizer — Claude Instructions

## Project Overview
오디오 파일을 FFT 스펙트럼 MP4 영상으로 변환하는 단일 파일 CLI 도구.

- 메인 파일: `visualizer.py` (단일 파일, 분리 금지)
- 의존성: `requirements.txt`

## Tech Stack
- **librosa**: STFT, 멜 필터뱅크, dB 변환 전용 (오디오 로드에는 사용하지 않음)
- **Pillow (PIL)**: 프레임 렌더링
- **ffmpeg/ffprobe**: 오디오 디코딩 + 영상 인코딩 (시스템 바이너리)
- **multiprocessing.SharedMemory**: 워커 간 스펙트럼 데이터 공유
- Python 3.14, macOS (Homebrew), `--break-system-packages`로 설치

## Architecture
```
load_audio_ffmpeg()         # ffprobe → sr 감지, ffmpeg → f32le PCM
    ↓
precompute_spectrum()       # STFT → mel(64) → dB → normalize → EMA → (n_frames, 64) float32
    ↓
SharedMemory                # 워커 프로세스 간 공유
    ↓
Pool._worker_init()         # 프로세스당 1회: SharedMemory 열기, 폰트 로드
Pool.render_frame(i)        # PIL로 프레임 생성 → raw bytes
    ↓
ffmpeg stdin pipe           # rawvideo rgb24 → libx264 + aac → MP4
```

## Critical Rules

### 멀티프로세싱 (macOS spawn)
- `_worker_init`, `render_frame`은 반드시 **모듈 최상위 레벨**에 정의
- 클로저나 람다는 pickle 불가 → 사용 금지
- `if __name__ == '__main__': main()` 가드 필수
- `shm.unlink()`는 반드시 `finally` 블록에서 호출 (크래시 시 메모리 누수 방지)

### 오디오 로딩
- `librosa.load()` 사용 금지: m4a/aac에서 soundfile 실패 → deprecated audioread fallback 발생
- 대신 `load_audio_ffmpeg()` 사용 (ffmpeg 기반)

### moviepy, matplotlib 사용 금지
- 제거된 의존성 — 다시 추가하지 말 것

## Visual Layout
- 화면 중앙 기준 좌우 미러 대칭
- bar[0] (저음): 중앙에 단일 막대
- bar[1..n-1] (고음으로 갈수록): 좌우 대칭으로 바깥쪽 배치
- 막대 높이: `usable_h / 4` (위아래 각각 최대 25%)
- 색상: HSV 240°→330° (파랑→핑크) 또는 `--color` 기준 ±45°

## CLI Flags
모든 기존 플래그 유지 필수 (하위 호환):
`input`, `-o`, `--fps`, `--width`, `--height`, `--color`, `--bg`, `--window`(무시됨), `--bars`, `--decay`, `--workers`, `--preview`

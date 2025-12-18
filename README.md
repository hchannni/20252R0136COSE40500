# 25-2 COSE405

## 환경 설정

```bash
# 클론 후 진입
git clone <repo-url>
cd 20252R0136COSE40500

# venv 생성/활성화 (macOS/Linux)
python3 -m venv venv
source venv/bin/activate

# venv (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 필수 패키지 설치 (requirements.txt)
pip install --upgrade pip
pip install -r requirements.txt
```

## 주요 실행 예시

- MNIST CNN (기본 설정, PyTorch 모듈 사용):
  ```bash
  python mnist_cnn_classifier.py
  ```
- Conv/Pool gradient check:
  ```bash
  python conv_pool_layer.py
  ```
- 문자 RNN 샘플 생성:
  ```bash
  python rnn_char_generator.py
  ```

## GitHub 초기 세팅 메모

```bash
git init
git remote add origin <repo-url>
git branch -M main
git add .
git commit -m "init"
git push -u origin main
```

# 1. Tổng quan về bài toán Text-to-Speech (TTS)
- TTS là nhiệm vụ biến văn bản thành giọng nói tự nhiên, bao gồm nhiều thành phần:
  - Xử lý ngôn ngữ (tokenization, Grapheme -> Phoneme, text normalization)
  - Mô hình biểu diễn âm học (acoustic model: text -> spectrogram / mel)
  - Vocoder (spectrogram -> waveform).
- Lịch sử phát triển có thể tóm gọn thành 3 level:
  - (1) luật / nối ghép (rule-based / concatenative / statistical parametric)
  - (2) TTS dựa trên deep-learning (Tacotron / FastSpeech + neural vocoder)
  - (3) personalized / few-shot / zero-shot voice cloning.
- Nhiều survey và tổng quan gần đây tập trung vào cách kết hợp mô hình transformer/flow/diffusion với vocoder GAN/flow để cân bằng chất lượng và tốc độ.

# 2. Các hướng triển khai
## Level 1 - Luật, Concatenative, Statistical Parametric
- Ý tưởng: Dùng luật ngôn ngữ/âm tiết, hoặc ghép âm mẫu (concatenation), hoặc HMM/parametric để sinh tham số âm thanh rồi vocoder cổ điển.
- Ưu điểm:
  - Chạy rất nhanh, ít cần dữ liệu lớn.
  - Kiểm soát tốt (deterministic), dễ debug cho từng quy tắc.
  - Phù hợp cho thiết bị giới hạn tài nguyên, hoặc ngôn ngữ/địa phương ít dữ liệu.
- Nhược điểm:
  - Giọng nghe "robotic", ít tự nhiên/biến thiên cảm xúc.
  - Khó mở rộng cho phát âm tinh tế, ngữ điệu phong phú.
- Phù hợp dùng:
  - Ứng dụng nhúng, hướng dẫn giọng tĩnh (IVR), nơi ưu tiên độ ổn định & tài nguyên thấp.

## Level 2 - Neural TTS (Tacotron, FastSpeech, VITS) + Neural Vocoder (WaveNet, WaveGlow, HiFi-GAN, BigVGAN)
- Ý tưởng: Học end-to-end hoặc 2-giai đoạn: text -> mel (acoustic model), mel -> waveform (neural vocoder). Các tiến bộ: Tacotron2 -> FastSpeech/FastSpeech2 (tốc độ), VITS (end-to-end dựa trên flow/vae) và vocoder GAN (HiFi-GAN, BigVGAN) giúp chất lượng cao & nhanh. 
- Ưu điểm:
  - Giọng rất tự nhiên, giàu ngữ điệu; có thể đạt chất lượng gần người thật.
  = Có thể train đa-speaker, multilingual với kiến trúc phù hợp.
  - Tùy chọn trade-off tốc độ <-> chất lượng (non-autoregressive như FastSpeech cho tốc độ).
- Nhược điểm:
  - Cần lượng dữ liệu lớn (đặc biệt cho đa-người/đa-ngôn ngữ).
  - Mô hình nặng, tốn tài nguyên huấn luyện và/hoặc inference (mặc dù có nhiều mô hình tối ưu).
  - Vấn đề kiểm soát prosody / style vẫn thách thức.
- Phù hợp dùng:
  - Ứng dụng sản xuất audio tự nhiên (audiobook, voice assistants, dubbing), dịch vụ đám mây TTS.

## Level 3 - Few-shot / Zero-shot / Personalized TTS (voice cloning)
- Ý tưởng: Tạo giọng của một người mới chỉ từ vài giây mẫu (few-second) — bằng cách dùng speaker encoder, speaker-conditioning, meta-learning, hoặc fine-tune nhanh trên mô hình đa-speaker tiền huấn luyện. Ví dụ YourTTS, XTTS-v2, các hệ thống zero-shot recent. 
- Ưu điểm:
  - Cho phép cá nhân hoá nhanh, tiết kiệm dữ liệu ghi âm.
  - Người dùng có thể “tạo” giọng riêng trong vài giây, tiện cho personalization.
- Nhược điểm:
  - Khó giữ cân bằng giữa tính tự nhiên và giống giọng; độ tương đồng thường kém hơn fine-tune.
  - Dễ gặp issues: overfitting khi fine-tune ít dữ liệu; rò rỉ danh tính (privacy / misuse).
  - Mô hình phức tạp, tốn tài nguyên để đạt chất lượng cao.
- Phù hợp dùng:
  - Ứng dụng cá nhân hóa (assistants cá nhân), tạo giọng cho người dùng, nhưng cần biện pháp an toàn/đồng ý.
 
# 3. Cách nghiên cứu & pipeline hiện đại để giảm nhược điểm và tận dụng ưu điểm
## 1. Dùng pipeline nhiều bước nhưng modular (text frontend -> acoustic model -> vocoder)
- Mỗi module tối ưu cho nhiệm vụ riêng (ví dụ tách tốt phoneme/ prosody).
- Nhiều hệ thống thương mại sử dụng pipeline 2-giai đoạn với neural vocoder tối ưu (HiFi-GAN, BigVGAN) để đạt tốc độ và chất lượng. 
## 2. Pretrain lớn, adapt nhỏ - multilingual / multispeaker pretraining + PEFT (parameter-efficient fine-tuning)
- Train một backbone trên tập dữ liệu lớn, đa ngôn ngữ/đa speaker; khi cần một voice mới hoặc ngôn ngữ mới, chỉ fine-tune một số tham số nhỏ (adapter, LoRA, bias-only, hoặc chỉ fine-tune decoder). Giảm overfitting và chi phí huấn luyện. Nhiều công trình 2024–2025 tập trung cả vào PEFT cho TTS. 
## 3. Speaker encoder / embedding + disentanglement (voice / prosody / content)
- Tách embedding cho speaker identity (giọng) và cho style/prosody (cảm xúc/độ dài/nhấn). Khi kết hợp với reference-encoder hoặc prosody-VAE, mô hình có thể chuyển prosody giữa mẫu tham chiếu và nội dung, giúp điều khiển tốt hơn và tránh huấn luyện tốn kém.
## 4. Meta-learning & few-shot learning
- Dùng meta-learning (MAML, LAML, episodic training) để làm cho mô hình "nhanh thích nghi" với speaker mới từ vài mẫu, cải thiện zero/few-shot. Nhiều paper gần đây áp dụng meta-learning cho ngôn ngữ ít dữ liệu. 
## 5. Data augmentation & synthetic data
- Tạo thêm dữ liệu bằng voice conversion, speed/pitch perturbation, hoặc synthetic TTS để tăng robustness cho speaker adaptation, đặc biệt cho ngôn ngữ ít dữ liệu.
## 6. End-to-end probabilistic models + better vocoders
- Mô hình như VITS (flow+VAE) cho phép end-to-end, gỡ bỏ bước tách mel, giảm lỗi alignment; kết hợp với vocoder GAN/flow (HiFi-GAN, BigVGAN) để cân bằng chất lượng và tốc độ. Các bản zero-shot SOTA (YourTTS, ZSE-VITS) thường xây dựng trên nền VITS và thêm speaker encoder / consistency losses. 
## 7. Non-autoregressive models để tăng tốc (FastSpeech family)
- FastSpeech / FastSpeech2 và các biến thể flow/diffusion cho phép sinh nhanh, ổn định thời gian inference, phù hợp production.
## 8. Kiểm định chất lượng và an toàn
- Pipeline còn phải bao gồm: detection misuse (voice cloning misuse), consent & watermarking (đánh dấu giọng tổng hợp), và metric evaluation: MOS/SMOS, speaker similarity, WER, prosody metrics. Điều này là cần thiết cho ứng dụng thương mại.

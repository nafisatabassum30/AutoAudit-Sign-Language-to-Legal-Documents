AUTOAUDIT TEAMMATE PACKAGE - ALL FILES WITHOUT RAW VIDEOS

Included:
1. Source code:
   - vision
   - llm
   - all root scripts

2. Models:
   - output/sign_sentence_v2_validated
   - output/sign_sentence_v3_v2plus_cnn_attention_continue
   - output/llm_lora_banglat5_final if available

3. Results:
   - outputs/sign_sentence_v2_validated
   - outputs/sign_sentence_v3_v2plus_cnn_attention_continue
   - outputs/Test_data_result if available
   - failed/experimental outputs if available

4. Data included:
   - data/raw
   - data/train
   - data/split_v2
   - data/Test_data
   - data/keypoints_full

5. Failed experiment archive:
   - archive_failed_v3_experiments

Not included:
1. Full raw video dataset:
   - data/videos

2. Virtual environment:
   - venv

Reason:
The raw video dataset is large and excluded. The extracted keypoints_full folder is included because it is only around 0.16 GB and useful for training/evaluation without the raw videos.

Current best practical model:
- v2 BiLSTM
- Validation accuracy: about 55.45%
- Test accuracy: about 55.66%

CNN-LSTM-Attention experiment:
- v3 v2plus CNN-Attention continue
- Validation accuracy: about 33.78%
- Test accuracy: about 35.95%

Demo:
1. Install requirements:
   pip install -r requirements.txt

2. Check v2 result:
   Get-Content outputs\sign_sentence_v2_validated\final_metrics.json

3. Check v3 result:
   Get-Content outputs\sign_sentence_v3_v2plus_cnn_attention_continue\final_metrics.json

4. Run prediction with your own video:
   python vision\predict_from_video_v2_exact.py --model_dir output\sign_sentence_v2_validated --video path\to\video.mp4

Note:
If full raw videos are needed, data/videos must be shared separately.

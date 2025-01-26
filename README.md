# voice-assistant-demo

CPU で動く常駐型音声アシスタント

## システム構成

本システムは以下のコンポーネントで構成されています：

- **FastAPI サーバー**: 音声合成リクエストを受け付けるエンドポイントを提供
- **whisper_mic**: 音声入力を待ち受けるバックグラウンドプロセス
- **Assistant**: 音声認識結果を処理し、LLM による応答を生成
- **TTS サーバー**: テキストを音声に変換するスタンドアロンサーバー

### 動作フロー

1. whisper_mic が音声入力を検知
2. 音声認識を実行し、結果を Assistant に渡す
3. Assistant が LLM で応答を生成
4. FastAPI サーバーに音声合成リクエストを送信
5. 合成された音声を再生

## 必要環境

- Python 3.8 以上（3.12 で動作確認）
- rye
- 十分なマシンスペック（メモリ 8GB 以上推奨、Macbook M1 Pro で動作確認）
- OpenAI のアカウント (API キー)

## セットアップ

1. rye sync を実行
2. .env.sample を.env にコピー
3. OPENAI_API_KEY を.env に設定
4. model_assets ディレクトリにモデルを追加:
   ```bash
   git clone https://huggingface.co/litagin/sbv2_koharune_ami model_assets/koharune-ami
   ```

## 実行

```bash
rye run python src/main.py

01-26 10:12:12 |  INFO  | bert_models.py:92 | Loaded the Languages.JP BERT model from ku-nlp/deberta-v2-large-japanese-char-wwm
01-26 10:12:13 |  INFO  | bert_models.py:154 | Loaded the Languages.JP BERT tokenizer from ku-nlp/deberta-v2-large-japanese-char-wwm
[01/26/25 10:12:15] INFO     No mic index provided, using     whisper_mic.py:123
                             default
[01/26/25 10:12:17] INFO     Mic setup complete               whisper_mic.py:137
                    INFO     Listening...                     whisper_mic.py:293
INFO:     Started server process [16878]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

この状態になれば正常に起動しています。マイクに音声を入力すると音声認識が実行され、LLM による返答が音声合成で再生されます。

```
[01/26/25 10:12:21] INFO Transcribing... whisper_mic.py:225
in: こんにちは
01-26 10:12:30 | INFO | tts_model.py:259 | Start generating audio data from text:
こんにちは！何かお手伝いできることがあれば、お知らせください
01-26 10:12:30 | INFO | infer.py:24 | Using JP-Extra model
/Users/ryunosuke/Documents/GitHub/voice-assistant-demo/.venv/lib/python3.12/site-packages/torch/nn/utils/weight_norm.py:143: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
WeightNorm.apply(module, name, dim)
01-26 10:12:31 | INFO | safetensors.py:50 | Loaded 'model_assets/koharune-ami/koharune-ami.safetensors' (iteration 60)
01-26 10:12:33 | INFO | tts_model.py:324 | Audio data generated successfully
INFO: 127.0.0.1:53722 - "POST /speak HTTP/1.1" 200 OK
out: こんにちは！何かお手伝いできることがあれば、お知らせください。
```

## 設定項目

必要に応じて以下の設定項目を調整してください。

### whisper 設定 (src/whisper_mic.py)

- model: 使用する whisper モデル (デフォルト: turbo)
- implementation: whisper 実装 (デフォルト: faster-whisper)
- device: 使用デバイス (デフォルト: cpu)

### マイク設定 (src/whisper_mic.py)

- energy: 音声検出の感度 (デフォルト: 300)
- pause: 音声終了検出までの無音時間 (デフォルト: 1 秒)

### OpenAI 設定 (src/assistant.py)

- model: 使用する GPT モデル (デフォルト: gpt-4o-mini)
- temperature, max_tokens など

### 音声合成設定 (src/tts_server.py)

model_assets に Style-Bert-VITS2 のモデルを配置することで、任意のモデルを使用することができます。
詳細は [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2?tab=readme-ov-file#%E5%AD%A6%E7%BF%92) のドキュメントを参照してください。

## [Mac] Dock への追加

以下の手順で Dock に追加することで、ショートカットや自動起動を設定することができます。

1. 以下の内容で `voice_assistant.command` を追加

```sh
cd {path_to_project}
rye run python src/main.py
```

2. Finder でディレクトリを開き、 `voice_assistant.command` を Dock にドラッグ & ドロップ
3. Dock からクリックして起動

Dock で「右クリック > ログイン時に開く」で、ログイン時に自動起動するように設定できます。

また、Finder から「右クリック > 情報を見る」で、アイコンを設定することもできます。

## クレジット

サンプルとして以下の音声モデルをお借りしています。モデルの利用の際は、ライセンスや利用規約を十分に確認してください。

- Style-BertVITS2 モデル:
  - https://huggingface.co/litagin/sbv2_koharune_ami
  - 小春音アミ、あみたろの声素材工房 (https://amitaro.net/)

その他、以下のパッケージを利用しています。

- [whisper](https://github.com/openai/whisper)
- [whisper_mic](https://github.com/mallorbc/whisper_mic)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Style-BertVITS2](https://github.com/litagin02/Style-Bert-VITS2)

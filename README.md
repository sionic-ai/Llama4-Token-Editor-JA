# 日本語トークンアナライザー & バイアスツール (Japanese Token Analyzer & Bias Tool)

本ツールは、大規模言語モデル（LLM）のトークナイザーを分析し、特に日本語や英語などの言語がどのように扱われているかに焦点を当てるためのツールです。トークンをカテゴリ分類し、これらのカテゴリに基づいてトークンの生成確率（logit bias）を調整する方法の例を、ローカルでのHugging Face Transformers利用とOpenAI互換API利用の両方で提供します。

元々は、Llama-4のような新しいモデルにおける日本語（または他言語）のトークン表現が、以前のモデルと比較してどのように改善されたかを評価するために使用されました。

## 主な機能

-   **トークンカテゴリ分析**: モデルの語彙（特殊トークンを除く）を分析し、以下のカテゴリに分類します:
    -   `pure_english`: 純粋な英語トークン（基本的なラテン文字のみ）
    -   `contains_japanese`: 日本語関連文字（ひらがな、カタカナ、漢字、日本語の句読点/記号）を少なくとも1つ含むトークン
    -   `pure_japanese_script`: 日本語のスクリプト文字（ひらがな、カタカナ、漢字）のみで構成されるトークン
    -   `contains_digit`: 数字を含むトークン
    -   `special_char_pattern`: 英数字、空白、言語文字以外の文字のみで構成されるトークン
    -   `uncategorized`: 上記のいずれにも分類されないトークン
-   **Logit Biasの適用**: 生成に影響を与えるためにlogit biasを適用する方法を示します:
    -   望ましいトークン（例：日本語トークン）の確率を増加させます。
    -   望ましくないトークンの確率を減少させます。
-   **提供される例**:
    -   Hugging Face `transformers` を使用したローカルでの生成 (`adjust_japanese_bias.py`)
    -   `openai` ライブラリを使用したAPIベースの生成 (`openai_call_jp.py`) (OpenAI, OpenRouter等に対応)
    -   vLLM統合のための概念的な例

## 分析テスト済みモデル例

分析スクリプト (`token_analyzer_jp.py`) は、以下のトークナイザーでテストされています（ただし、他の多くのモデルにも適用可能です）:

-   `stabilityai/japanese-stablelm-instruct-gamma-7b` (**`trust_remote_code=True` が必要**)
-   `elyza/ELYZA-japanese-Llama-2-7b-instruct` (**`trust_remote_code=True` が必要**)
-   `mistralai/Mixtral-8x7B-Instruct-v0.1` (多言語モデルの例として)
-   `meta-llama/Meta-Llama-3.1-8B-Instruct` (多言語モデルの例として)

使用例 (`adjust_japanese_bias.py`, `openai_call_jp.py`) は、様々なCausal LMモデルに適合させることができますが、モデルによっては `trust_remote_code=True` が必要になる場合があります。

## 分析比較例（仮）

以下の表は、異なるモデルのトークナイザーが日本語をどのように表現しているかの比較（非特殊トークン対象）の**概念的な例**を示しています。正確な数値を得るには、対象モデルで `token_analyzer_jp.py` を実行してください。

*   **日本語を含む (`contains_japanese`)**: ひらがな、カタカナ、漢字、日本語の句読点/記号のいずれかを含むトークン。BPE（Byte Pair Encoding）において、これらの文字の一部が他の言語のトークンと共有されることがあります。
*   **純粋な日本語スクリプト (`pure_japanese_script`)**: トークンがひらがな、カタカナ、漢字のみで構成されている（句読点や他の言語文字を含まない）。

| 分析項目                     | Japanese StableLM (stabilityai) | ELYZA Llama2 (elyza) | Mixtral (mistralai) | Llama 3.1 (meta) |
| :--------------------------- | :------------------------------ | :------------------- | :------------------ | :--------------- |
| 語彙サイズ                   | ~32k                            | ~32k                 | ~32k                | ~128k            |
| 特殊トークン数（除外）       | 数十                            | 数十                 | 数十                | ~256             |
| 分析対象非特殊トークン数     | **~31k**                        | **~31k**             | ~31k                | ~128k            |
| 純粋な英語トークン数         | **少ない**                      | **少ない**           | ~22k                | ~28k             |
| 日本語を含むトークン数       | **多い**                        | **多い**             | 中程度              | 少ない           |
| 純粋な日本語スクリプト数     | **中程度**                      | **中程度**           | 少ない              | 非常に少ない     |
| 特殊文字パターントークン数   | ~1k                             | ~1k                  | ~1.5k               | ~2.5k            |
| 未分類トークン数             | **少ない**                      | **少ない**           | ~3k                 | ~70k             |

*(**注意:** 上記の数値は**例示**です。特に日本語特化モデル（StableLM, ELYZA）は、日本語トークンの数が多く、英語や未分類が少なくなる傾向があります。多言語モデル（Mixtral, Llama 3）は異なる分布を示します。正確な数値は `token_analyzer_jp.py` を実行して確認してください。)*

## インストール

1.  **リポジトリをクローン:**
    ```bash
    git clone https://github.com/your-username/Japanese-Token-Analyzer.git # 必要に応じてリポジトリURLを修正
    cd Japanese-Token-Analyzer
    ```

2.  **依存関係をインストール:**
    ```bash
    pip install -r requirements.txt
    ```
    *   PyTorch は、ご使用の CUDA バージョンに合わせて別途インストールする必要がある場合があります。[PyTorch インストール手順](https://pytorch.org/get-started/locally/)を参照してください。
    *   `device_map='auto'` や `'balanced'` を使用する場合は `accelerate` が必要です (`pip install accelerate`)。
    *   モデルによっては追加の依存関係が必要な場合があります（例: `sentencepiece`）。モデルカードを確認してください。

## 使用方法

### 1. トークナイザー分析の実行

`token_analyzer_jp.py` スクリプトを実行して、モデルのトークナイザーを分析し、カテゴリファイル（JSON と TXT）を生成します。

```bash
# 例: Japanese StableLM Instruct Gamma 7B
python token_analyzer_jp.py --model_id "stabilityai/japanese-stablelm-instruct-gamma-7b" --output_dir ./analysis_results_stablelm

# 例: ELYZA Japanese Llama 2 7B Instruct
python token_analyzer_jp.py --model_id "elyza/ELYZA-japanese-Llama-2-7b-instruct" --output_dir ./analysis_results_elyza
Use code with caution.
Markdown
パラメータ:
--model_id: Hugging Face モデルID またはモデル/トークナイザーへのローカルパス。(必須)
--min_token_id: 分析範囲の最小トークンID（デフォルト: 0）。特殊トークンは常に除外されます。
--output_dir: すべての出力ファイル（JSON と .txt リスト）を保存するディレクトリ（デフォルト: カレントディレクトリ）。
出力ファイル (output_dir 内):
token_analysis_jp_*.json: 詳細な分析結果（統計、除外された特殊IDリスト、分類された非特殊トークンIDリスト）。
contains_japanese_*.txt: 「日本語を含む」トークンIDのリスト（日本語バイアス調整に有用）。
pure_japanese_script_*.txt: 「純粋な日本語スクリプト」トークンIDのリスト。
uncategorized_*.txt: 定義されたカテゴリに分類されなかった非特殊トークンIDのリスト。
（その他、pure_english_*.txt なども生成されます）
2. Logit Bias の適用 (ローカル実行例)
adjust_japanese_bias.py を使用して、ローカルのHugging Faceモデルでテキストを生成し、分析出力から読み込んだ日本語トークンにバイアスを適用します。
# まず、対象モデルでアナライザーを実行したことを確認してください:
# python token_analyzer_jp.py --model_id "stabilityai/japanese-stablelm-instruct-gamma-7b" --output_dir .

# 次に、生成例を実行します:
python adjust_japanese_bias.py \
    --model_id "stabilityai/japanese-stablelm-instruct-gamma-7b" \
    --prompt "日本の有名な観光地といえば、" \
    --japanese_ids_file "./contains_japanese_japanese-stablelm-instruct-gamma-7b.txt" \
    --japanese_bias 2.0 \
    --max_length 60 \
    --temperature 0.6
Use code with caution.
Bash
adjust_japanese_bias.py の主な引数:
--model_id: 読み込むモデル（IDが正しく対応するように分析済みモデルと一致させる必要があります）。
--japanese_ids_file: 日本語トークンIDを含む .txt (または .json) ファイルへのパス。ファイル名はモデルによって異なります。アナライザーの出力に合わせてください。
--japanese_bias: 正の値は日本語の確率を増加させ、0はバイアスを無効にします。
--prompt: 入力テキスト。
重要: stabilityai/japanese-stablelm-instruct-gamma-7b のようなモデルを使用するには trust_remote_code=True が必要になる場合があります。これはモデルリポジトリからコードを実行するため、信頼できるソースであることを確認してください。
3. Logit Bias の適用 (OpenAI API 例)
openai_call_jp.py を使用して、OpenAI互換API（OpenRouterなど）経由でテキストを生成し、日本語トークンにバイアスを適用します。
# アナライザーから contains_japanese_*.txt ファイルがあることを確認してください。
# APIキーを環境変数に設定します: export OPENROUTER_API_KEY="your-key"

python openai_call_jp.py \
    --model_name "mistralai/mixtral-8x7b-instruct" `# APIプロバイダーでの正確なモデル名を確認` \
    --prompt "日本のアニメでおすすめは？" \
    --japanese_ids_file "./contains_japanese_mixtral-8x7b-instruct.txt" `# モデルに対応するファイル名` \
    --japanese_bias 10.0 \
    --max_tokens 80
Use code with caution.
Bash
openai_call_jp.py の主な引数:
--model_name: APIプロバイダーで使用される正確なモデル識別子（例: OpenRouterでは異なる文字列である可能性があります）。プロバイダーのドキュメントで確認してください。
--api_key: APIキー（または OPENROUTER_API_KEY 環境変数）。
--base_url: APIエンドポイントURL。
--japanese_ids_file: 日本語IDを含む .txt (または .json) ファイルへのパス。ファイル名はモデルによって異なります。
--japanese_bias: バイアス値（OpenAI APIでは通常 -100 から 100）。
4. vLLM 統合 (概念)
vLLMのOpenAI互換エンドポイントを使用する場合の基本的な考え方は同じです:
japanese_token_ids.txt などから目的のトークンID（例：日本語ID）を読み込みます。
logit_bias 辞書を作成します（文字列のIDをバイアス値にマッピング）。
vLLMの /v1/chat/completions エンドポイントへのAPIリクエストに logit_bias を含めます。
# vLLM OpenAIエンドポイント利用の概念（サーバー改修またはクライアントリクエスト内）

# 1. IDの読み込み (例: load_japanese_token_ids_from_txtを使用)
# japanese_ids = load_japanese_token_ids_from_txt("contains_japanese_your_model.txt")

# 2. logit_bias辞書の作成
# logit_bias_value = 20.0 # 例
# logit_bias = {str(token_id): logit_bias_value for token_id in japanese_ids}

# 3. リクエストペイロードに含める (vLLMの /v1/chat/completions 呼び出し時)
# request_payload = {
#     "model": "your_deployed_model_name",
#     "messages": [{"role": "user", "content": "日本の文化について教えてください。"}],
#     "logit_bias": logit_bias,
#     "max_tokens": 100,
#     "temperature": 0.7
# }
# response = requests.post("http://your_vllm_server/v1/chat/completions", json=request_payload)
Use code with caution.
Python
日本語トークン分析の原理
このツールは、非特殊 トークンのデコードされた文字列を分析し、その文字のUnicodeプロパティに基づいて日本語文字を識別します:
Unicode Range Checks: ひらがな (U+3040-U+309F)、カタカナ (U+30A0-U+30FF, U+FF65-U+FF9F)、一般的なCJK統合漢字 (U+4E00-U+9FFF, U+3400-U+4DBF)、および一般的な日本語の句読点や全角記号の範囲に基づいて文字を分類します。
contains_japanese: 上記のいずれかの日本語関連文字を1つでも含むトークンを識別します。
pure_japanese_script: ひらがな、カタカナ、漢字のみで構成されるトークンを識別します。
貢献
バグレポートや機能リクエストは、GitHubのIssuesを通じて提出してください。プルリクエストも歓迎します！
ライセンス
MIT License
Use code with caution.

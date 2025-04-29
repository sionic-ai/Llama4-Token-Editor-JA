# Llama4 Token Editor（日本語バージョン）

**Llama4 Token Editor**は、大規模言語モデル（LLM）のトークナイザーを分析し、特定カテゴリーに属するトークンの重み（logits）を調整できるツールです。  
もともと韓国語向けだった例をベースに、**日本語**（ひらがな・カタカナ（全角/半角）・漢字・全角英数字など）の解析を強化したバージョンとなっています。  
Llama系を中心に、モデル内部のボキャブラリー（vocab）がどの程度日本語を扱っているかを調べて、必要に応じてログ確率（logits）バイアスを付与できます。

---

## 主な機能

- **トークン分類解析**  
  - モデル全体のボキャブラリーをスキャンし、以下のようなカテゴリに仕分けします：  
    - `contains_japanese`：ひらがな/カタカナ（全角/半角）/漢字/全角ASCIIなど、日本語関連文字を1文字以上含む  
    - `pure_japanese_script`：ひらがな・カタカナ・漢字のみで構成されたトークン  
    - `contains_katakana_half`：半角カタカナを含むトークン  
    - `contains_fullwidth_ascii`：全角数字/全角英字などを含むトークン  
    - そのほか `pure_english`・`special_char_pattern`・`uncategorized` など  
  - BPEで分割された**部分文字列**（例：UTF-8が途中で切れている場合）でも「日本語を形成する可能性」があればきちんと分類  

- **トークン重み付け（logitsバイアス）調整**  
  - 解析結果として生成される `categorized_token_ids.txt` や `uncategorized_token_ids.txt` などのファイルを参照し、  
    モデル推論時に特定トークンのスコアを上げたり下げたりできます。  
  - 例：  
    - 日本語トークンに+2.0のバイアスを与え、モデルが日本語単語を優先的に使うように誘導  
    - 特殊文字パターンに-1.0のペナルティを付与し、不要な記号の多用を抑制 など  

---

## インストール

```bash
git clone https://github.com/sionic-ai/Llama4-Token-Editor-JA.git
cd Llama4-Token-Editor-JA
pip install -r requirements.txt
```

---

## 使い方

### 1. トークン解析の実行

```bash
python token_analyzer_ja.py --model_id "モデルのパスまたはID"
```

**主な引数**:

- `--min_token_id`：解析対象トークンIDの下限（デフォルト：102）  
  通常0～101に特殊トークン（[CLS], [PAD]など）が多く含まれるため、誤ってそれらにバイアスを与えない目的で除外  
- `--output_dir`：解析結果ファイルの出力先（デフォルト：`token_analysis_output`）

解析終了後、以下のようなファイルが生成されます：

1. **JSONファイル**（例：`token_analysis_jp_<モデル名>.json`）  
   - カテゴリ別のトークン数、トークンID一覧、統計情報などを格納  
2. **カテゴリ別テキストファイル**  
   - `contains_japanese_<モデル名>.txt` や `pure_japanese_script_<モデル名>.txt` など  
   - 中身はトークンIDのリストで、後ほどバイアス付与時に流用可能

---

### 2. トークン重み付け（logitsバイアス）の例

#### a) Transformers でのバイアス調整

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch

class TokenBiasLogitsProcessor:
    def __init__(self, token_ids, bias_value):
        self.token_ids = token_ids
        self.bias_value = bias_value

    def __call__(self, input_ids, scores):
        for tid in self.token_ids:
            scores[:, tid] += self.bias_value
        return scores

model = AutoModelForCausalLM.from_pretrained("your_japanese_model")
tokenizer = AutoTokenizer.from_pretrained("your_japanese_model")

# 例: "contains_japanese_..." ファイルから日本語トークンIDを読み込む
with open("contains_japanese_myModel.txt", "r", encoding="utf-8") as f:
    data = f.read()
    # "japanese_ids = [123,456,789]" のように定義されていると仮定
    line_clean = data.replace("japanese_ids = [", "").replace("]", "")
    japanese_token_ids = [int(x) for x in line_clean.split(",")]

bias_value = 2.0  # 日本語トークンに加算する値

prompt = "今日は何をしましょうか？"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

processor = LogitsProcessorList([
    TokenBiasLogitsProcessor(japanese_token_ids, bias_value)
])
output = model.generate(
    input_ids,
    max_length=80,
    do_sample=True,
    temperature=0.7,
    logits_processor=processor
)

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)
```

#### b) vLLM の OpenAI互換APIで logit_bias を設定

```python
# vllm/entrypoints/openai/api_server.py の一例
@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    # ...
    # 例: 日本語トークンIDが [105714, 115668, ...] など
    logit_bias_list = [105714, 115668, 104949]
    logit_bias = {str(tid): 10 for tid in logit_bias_list}
    request.logit_bias = logit_bias

    generator = await handler.create_chat_completion(request, raw_request)
    # ...
```

---

## 分析例：Llama系モデル比較

| モデル名                           | 語彙サイズ<br>(特殊トークン数) | 解析対象トークン数 | 日本語関連<br>(contains_japanese) | その比率(%) | 純粋日本語<br>(pure_japanese_script) | その比率(%) | 特殊文字<br>(special_char_pattern) | 未分類<br>(uncategorized) |
|----------------------------------|--------------------------------|--------------------|-----------------------------------|------------|--------------------------------------|------------|------------------------------------|---------------------------|
| **Llama-4-Scout-17B-16E**        | 200,000<br>(3)                 | 199,898            | 14,571                            | 7.29%      | 12,175                               | 6.09%      | 4,233                              | 48,152                    |
| **Llama-3.3-70B-Instruct**       | 128,000<br>(3)                 | 127,898            | 5,728                             | 4.48%      | 4,605                                | 3.60%      | 3,752                              | 20,689                    |
| **Mistral-Small-3.1-24B-2503**   | 131,072<br>(1,000)             | 130,072            | 5,623                             | 4.32%      | 4,632                                | 3.56%      | 2,762                              | 33,693                    |
| **Qwen3-0.6B**                   | 151,643<br>(14)                | 151,541            | 27,851                            | 18.38%     | 27,296                               | 18.01%     | 6,192                              | 23,689                    |

- **Llama-4-Scout**  
  - 全体の約7.3%が「日本語関連トークン」で、純粋日本語トークンも6%強  
  - “ひらがな/カタカナ/漢字”のパーツが他モデルより多い傾向  
- **Llama-3.3-70B**  
  - 約4.48%が日本語関連  
  - 未分類は約2万とやや多め  
- **Mistral-Small-3.1-24B**  
  - 4.32%が日本語関連  
  - 未分類が約3.3万と比較的多いが、Llama-4-Scoutほどの日本語量はカバーしていない

**結論**：  
Qwenを抜きに考えると、**Llama-4-Scout**が日本語トークン量では優位に見えます。  
一方、MistralやLlama-3.3は単に日本語が少ないのではなく、「未分類」に入っている多言語や混合文字トークンもあるため、どの程度日本語を使いたいか次第で選択が変わります。

---

## 日本語トークン判定ロジック

- `is_japanese_related_char`  
  - 文字単位で、**ひらがな** (U+3040～309F)、**カタカナ（全角/半角）**、**CJK漢字** (U+4E00～U+9FFF, 拡張A～など)、**全角英数字**(FF10～FF19, FF21～... など)、日本語句読点 などを判定  
- `can_be_japanese_utf8`  
  - バイト単位で分析し、UTF-8で3〜4バイトの途中切れでも日本語文字に結合しうるかをチェック  
- `is_complete_japanese_utf8`  
  - 正確に3or4バイトで1文字の日本語を表すか判定（CJK拡張漢字などの4バイト対応）

このような仕組みにより、BPEで一部だけ区切られたトークンも**「日本語文字になり得るもの」**として拾い上げます。

---

## 生成結果の例

**プロンプト**: 「今日はとてもいい天気ですね」  
- **日本語トークンに+2.0バイアス** → 生成例:
  ```
  本当に素晴らしい天気ですね。散歩やカフェに行くのも気持ちが良さそうです。
  ```
- **バイアスなし** → 生成例:
  ```
  Yes, the weather is quite nice. Would you like more details on the local climate?
  ```

日本語トークンに明確なバイアスを付けることで、英語や特殊文字を避け、日本語主体のテキストを生成しやすくなります。

---

## ライセンス

MIT License

バグ報告や機能提案は [GitHub Issues](https://github.com/sionic-ai/Llama4-Token-Editor-JP/issues) へお願いします。  
PR（Pull Request）も大歓迎です！

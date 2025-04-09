# adjust_japanese_bias.py
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,  # より汎用的なモデルクラスを使用
    LogitsProcessorList,
    LogitsProcessor,
)
import json
import argparse
import os
import re
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TokenBiasLogitsProcessor(LogitsProcessor):
    """指定されたトークンIDにバイアスを追加するLogitsProcessor。"""

    def __init__(self, token_ids_to_bias: Dict[int, float]):
        if not isinstance(token_ids_to_bias, dict):
            raise ValueError("`token_ids_to_bias` は辞書である必要があります。")
        self.token_ids_to_bias = token_ids_to_bias
        self._token_ids = set(token_ids_to_bias.keys())  # 高速チェック用（任意）

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        for token_id, bias in self.token_ids_to_bias.items():
            if 0 <= token_id < vocab_size:
                scores[:, token_id] += bias
            # else:
            #     logging.warning(f"トークンID {token_id} (バイアス {bias}) は語彙範囲外 ({vocab_size}) です。スキップします。")
        return scores


def load_japanese_token_ids_from_json(json_path: str) -> List[int]:
    """分析JSONファイルから日本語関連のトークンIDを読み込みます。"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
        # JSON構造内の正しいキーを使用 ('contains_japanese' または 'pure_japanese_script')
        token_ids_data = analysis_data.get("token_ids", {})
        # バイアスには 'contains_japanese' を使うのが一般的
        japanese_ids = token_ids_data.get("contains_japanese", [])
        logging.info(
            f"{len(japanese_ids)} 個の日本語トークンIDをJSONから読み込みました: {json_path}"
        )
        return japanese_ids
    except FileNotFoundError:
        logging.error(f"分析JSONファイルが見つかりません: {json_path}")
        return []
    except Exception as e:
        logging.error(f"JSONからのトークンID読み込みエラー ({json_path}): {e}")
        return []


def load_japanese_token_ids_from_txt(txt_path: str) -> List[int]:
    """生成されたTXTファイルから日本語トークンIDを読み込みます。"""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
            # "var_name = [id1,id2,...]" 形式からIDを抽出
            match = re.search(r"\[(.*?)\]", content)
            if match:
                ids_str = match.group(1)
                japanese_ids = [
                    int(id_str) for id_str in ids_str.split(",") if id_str.strip()
                ]
                logging.info(
                    f"{len(japanese_ids)} 個の日本語トークンIDをTXTから読み込みました: {txt_path}"
                )
                return japanese_ids
            else:
                logging.error(
                    f"TXTファイル内でIDリストのパターンが見つかりません: {txt_path}"
                )
                return []
    except FileNotFoundError:
        logging.error(f"分析TXTファイルが見つかりません: {txt_path}")
        return []
    except Exception as e:
        logging.error(f"TXTからのトークンID読み込みエラー ({txt_path}): {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Transformersを使用してローカルでテキスト生成 (オプションで日本語トークンバイアス適用)"
    )
    # 日本語モデルの例を使用 (または分析したモデルを指定)
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/japanese-stablelm-instruct-gamma-7b",
        help="Hugging FaceモデルIDまたはローカルパス。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="日本の首都は",
        help="生成のための入力プロンプト。",
    )
    # 日本語IDファイルを指すように変更
    parser.add_argument(
        "--japanese_ids_file",
        type=str,
        default="contains_japanese_japanese-stablelm-instruct-gamma-7b.txt",  # デフォルトを具体的に
        help="日本語トークンIDを含むファイルへのパス (TXTまたはJSON)。ファイル名はモデルによって変わります。",
    )
    parser.add_argument(
        "--japanese_bias",
        type=float,
        default=1.5,  # 日本語バイアス
        help="日本語トークンのロジットに追加するバイアス値 (0でバイアス無効)。",
    )
    parser.add_argument(
        "--max_length", type=int, default=100, help="生成されるシーケンスの最大長。"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="サンプリング温度。"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="デバイスマップ戦略 ('auto', 'balanced', 特定デバイスなど)。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],  # bf16が推奨されることが多い
        help="モデル読み込み時のTorch dtype (float16, bfloat16, float32)。",
    )

    args = parser.parse_args()

    # Torch dtype 設定
    torch_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(
        args.dtype, torch.bfloat16
    )  # デフォルトをbf16に変更

    # --- 1. トークナイザーの読み込み ---
    logging.info(f"トークナイザーを読み込み中: {args.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                logging.warning(
                    "PADトークンが見つかりません。EOSトークンをPADトークンとして使用します。"
                )
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id  # IDも設定
            else:
                logging.error(
                    "PADトークンもEOSトークンも見つかりません。パディングが必要です。"
                )
                # 必要であればデフォルトのPADトークンを追加するロジックをここに入れる
                # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                return
    except Exception as e:
        logging.error(f"トークナイザーの読み込みエラー: {e}")
        return

    # --- 2. モデルの読み込み ---
    logging.info(
        f"モデルを読み込み中: {args.model_id} (dtype={args.dtype}, device_map='{args.device_map}')"
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=args.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model.eval()  # 評価モードに設定
    except ImportError as e:
        logging.error(
            f"インポートエラー: {e}。このモデルは特定の依存関係が必要か、trust_remote_code=Trueが不十分かもしれません。"
        )
        return
    except Exception as e:
        logging.error(f"モデルの読み込みエラー: {e}")
        logging.info(
            "device_mapを使用する場合は 'accelerate' がインストールされていることを確認してください (`pip install accelerate`)"
        )
        return

    # --- 3. 入力の準備 ---
    # 注意: チャット形式の場合は tokenizer.apply_chat_template を使用
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    try:
        input_ids = input_ids.to(model.device)
        logging.info(f"入力IDをデバイスに移動しました: {input_ids.device}")
    except Exception as e:
        logging.warning(
            f"入力IDをモデルデバイス ({model.device}) に移動できませんでした: {e}。生成が失敗する可能性があります。"
        )

    # --- 4. Logits Processor の設定 ---
    logits_processor = None
    if args.japanese_bias > 0:
        logging.info(f"日本語トークンバイアスを適用試行: {args.japanese_bias}")
        japanese_ids = []
        # ファイル拡張子に基づいてTXTまたはJSONから読み込み
        ids_file = args.japanese_ids_file
        if not os.path.exists(ids_file):
            logging.warning(
                f"指定されたIDファイルが見つかりません: {ids_file}. デフォルトのファイル名を試します。"
            )
            # モデル名からデフォルトファイル名を推測
            model_name_part = args.model_id.split("/")[-1].replace("-", "_")
            default_txt_file = f"contains_japanese_{model_name_part}.txt"
            if os.path.exists(default_txt_file):
                ids_file = default_txt_file
                logging.info(f"デフォルトのTXTファイルを使用します: {ids_file}")
            else:
                default_json_file = f"token_analysis_jp_{model_name_part}.json"
                if os.path.exists(default_json_file):
                    ids_file = default_json_file
                    logging.info(f"デフォルトのJSONファイルを使用します: {ids_file}")
                else:
                    logging.error(
                        "IDファイルが見つかりません。バイアスは適用されません。"
                    )

        if os.path.exists(ids_file):
            if ids_file.endswith(".txt"):
                japanese_ids = load_japanese_token_ids_from_txt(ids_file)
            elif ids_file.endswith(".json"):
                japanese_ids = load_japanese_token_ids_from_json(ids_file)
            else:
                logging.warning(
                    f"不明なIDファイル形式: {ids_file}。TXTとJSONローダーを試します。"
                )
                japanese_ids = load_japanese_token_ids_from_txt(
                    ids_file
                ) or load_japanese_token_ids_from_json(ids_file)

        if japanese_ids:
            token_bias_dict = {
                token_id: args.japanese_bias for token_id in japanese_ids
            }
            token_bias_processor = TokenBiasLogitsProcessor(token_bias_dict)
            logits_processor = LogitsProcessorList([token_bias_processor])
            logging.info(
                f"{len(japanese_ids)} 個のバイアス付きIDを持つLogitsProcessorを作成しました。"
            )
        elif os.path.exists(ids_file):  # ファイルは存在するがIDが読み込めなかった場合
            logging.warning(
                f"ファイル {ids_file} から日本語IDを読み込めませんでした。バイアスなしで続行します。"
            )
        # ファイルが存在しない場合は既にエラーログが出ている

    # --- 5. 生成設定 ---
    # max_lengthではなくmax_new_tokensを使用するのが一般的
    max_new_tokens = args.max_length - input_ids.shape[1]
    if max_new_tokens <= 0:
        logging.warning(
            f"max_length ({args.max_length}) がプロンプト長 ({input_ids.shape[1]}) 以下です。max_new_tokensを50に設定します。"
        )
        max_new_tokens = 50

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,  # pad_token_idを設定することが重要
        # 必要に応じて top_k, top_p などを追加
    }

    # --- 6. バイアスありで生成 ---
    if logits_processor:
        print("\n--- 日本語トークンバイアスありで生成 ---")
        try:
            with torch.no_grad():
                output_with_processor = model.generate(
                    input_ids=input_ids,
                    logits_processor=logits_processor,
                    **generation_config,
                )
            # 生成された部分のみをデコード
            generated_ids_with_processor = output_with_processor[0][
                input_ids.shape[1] :
            ]
            generated_text_with_processor = tokenizer.decode(
                generated_ids_with_processor, skip_special_tokens=True
            )
            print("出力 (バイアスあり):")
            print(args.prompt + generated_text_with_processor)  # プロンプトを先頭に追加
        except Exception as e:
            logging.error(f"バイアスあり生成中のエラー: {e}", exc_info=True)

    # --- 7. バイアスなしで生成 ---
    print("\n--- 日本語トークンバイアスなしで生成 ---")
    try:
        with torch.no_grad():
            output_no_processor = model.generate(
                input_ids=input_ids,
                # logits_processor なし
                **generation_config,
            )
        # 生成された部分のみをデコード
        generated_ids_no_processor = output_no_processor[0][input_ids.shape[1] :]
        generated_text_without_processor = tokenizer.decode(
            generated_ids_no_processor, skip_special_tokens=True
        )
        print("出力 (バイアスなし):")
        print(args.prompt + generated_text_without_processor)  # プロンプトを先頭に追加
    except Exception as e:
        logging.error(f"バイアスなし生成中のエラー: {e}", exc_info=True)


if __name__ == "__main__":
    main()

# openai_call_jp.py
from openai import OpenAI
import json
import os
import argparse
import re
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- 日本語トークンID読み込み関数 (上記 adjust_japanese_bias.py と同様) ---
def load_japanese_token_ids_from_json(json_path: str) -> List[int]:
    """分析JSONファイルから日本語関連のトークンIDを読み込みます。"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
        token_ids_data = analysis_data.get("token_ids", {})
        japanese_ids = token_ids_data.get(
            "contains_japanese", []
        )  # 'contains_japanese' を使用
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
        description="OpenAI APIを使用してテキスト生成 (オプションで日本語logitバイアス適用)"
    )
    # OpenRouter等で利用可能な日本語対応モデル名を指定する必要がある
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/mixtral-8x7b-instruct",  # 例: Mixtralは多言語対応
        help="APIで使用するモデル名 (プロバイダーのドキュメントで確認してください)。",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="APIキー (OpenRouterまたはOpenAI)。デフォルトはOPENROUTER_API_KEY環境変数。",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="APIベースURL (例: OpenRouterエンドポイント)。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="日本の有名な食べ物といえば？",
        help="生成のための入力プロンプト。",
    )
    # 日本語IDファイルを指定
    parser.add_argument(
        "--japanese_ids_file",
        type=str,
        default="contains_japanese_mixtral-8x7b-instruct.txt",  # ファイル名はモデル依存
        help="日本語トークンIDを含むファイルへのパス (TXTまたはJSON)。",
    )
    parser.add_argument(
        "--japanese_bias",
        type=float,
        default=5.0,  # 日本語バイアス
        help="日本語トークンのLogitバイアス値 (-100～100)。0でバイアス無効。",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=100, help="生成する最大トークン数。"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="サンプリング温度。"
    )

    args = parser.parse_args()

    if not args.api_key:
        logging.error(
            "APIキーが指定されていません。OPENROUTER_API_KEY環境変数を設定するか、--api_keyを使用してください。"
        )
        return

    # --- 1. OpenAIクライアントの初期化 ---
    try:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        logging.info(f"OpenAIクライアントを初期化しました (ベースURL: {args.base_url})")
    except Exception as e:
        logging.error(f"OpenAIクライアントの初期化エラー: {e}")
        return

    # --- 2. 日本語IDの読み込みとlogit_biasの準備 ---
    logit_bias_dict = None
    if args.japanese_bias != 0:
        japanese_ids = []
        # ファイル拡張子に基づいてTXTまたはJSONから読み込み
        ids_file = args.japanese_ids_file
        if not os.path.exists(ids_file):
            logging.warning(f"指定されたIDファイルが見つかりません: {ids_file}。")
            # デフォルトファイル名の推測 (モデル名が必要)
            model_name_part = args.model_name.split("/")[-1].replace(
                "-", "_"
            )  # APIモデル名を使用
            default_txt_file = f"contains_japanese_{model_name_part}.txt"
            if os.path.exists(default_txt_file):
                ids_file = default_txt_file
                logging.info(f"デフォルトのTXTファイルを使用します: {ids_file}")
            else:
                # JSONファイルも試す (ファイル名形式が一致していると仮定)
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
                logging.warning(f"不明なIDファイル形式: {ids_file}")
                japanese_ids = load_japanese_token_ids_from_txt(
                    ids_file
                ) or load_japanese_token_ids_from_json(ids_file)

        if japanese_ids:
            # OpenAI APIは-100から100の範囲を期待
            clamped_bias = max(-100.0, min(100.0, args.japanese_bias))
            if clamped_bias != args.japanese_bias:
                logging.warning(
                    f"API互換性のため、バイアス値を {args.japanese_bias} から {clamped_bias} に制限しました。"
                )
            # キーは文字列である必要がある
            logit_bias_dict = {str(token_id): clamped_bias for token_id in japanese_ids}
            logging.info(
                f"{len(logit_bias_dict)} 件のエントリを持つlogitバイアス辞書を準備しました (バイアス値: {clamped_bias})"
            )
        elif os.path.exists(ids_file):
            logging.warning(
                f"ファイル {ids_file} から日本語IDを読み込めませんでした。バイアスなしで続行します。"
            )
        # ファイルが存在しない場合は既にエラーログ済み

    # --- 3. 共通リクエストパラメータ ---
    # APIプロバイダーがモデル名を正しくマッピングし、チャット補完形式をサポートしている必要がある
    request_params = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        # 必要に応じて stream=False などを追加
    }

    # --- 4. Logitバイアスありで生成 ---
    if logit_bias_dict:
        print(f"\n--- 日本語Logitバイアスあり ({args.japanese_bias}) で生成 ---")
        try:
            biased_params = request_params.copy()
            biased_params["logit_bias"] = logit_bias_dict
            response_biased = client.chat.completions.create(**biased_params)
            print("出力 (バイアスあり):")
            if response_biased.choices:
                print(response_biased.choices[0].message.content)
            else:
                print("応答の選択肢が受信されませんでした。")
                # print(f"Full response: {response_biased}") # デバッグ用
        except Exception as e:
            logging.error(f"バイアスありAPI呼び出し中のエラー: {e}")

    # --- 5. Logitバイアスなしで生成 ---
    print("\n--- 日本語Logitバイアスなしで生成 ---")
    try:
        # logit_biasなしの元のrequest_paramsを使用
        response_unbiased = client.chat.completions.create(**request_params)
        print("出力 (バイアスなし):")
        if response_unbiased.choices:
            print(response_unbiased.choices[0].message.content)
        else:
            print("応答の選択肢が受信されませんでした。")
            # print(f"Full response: {response_unbiased}") # デバッグ用
    except Exception as e:
        logging.error(f"バイアスなしAPI呼び出し中のエラー: {e}")


if __name__ == "__main__":
    main()

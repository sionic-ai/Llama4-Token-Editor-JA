# token_analyzer_jp.py
import transformers
import json
import argparse
from typing import Dict, List, Set, Any
from tqdm import tqdm
import os
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 日本語文字識別関数 (Unicode文字セットを使用) ---

# 高速チェックのためのセットを使用
HIRAGANA = set(chr(c) for c in range(0x3040, 0x30A0))
KATAKANA = set(chr(c) for c in range(0x30A0, 0x3100)) # 全角ブロック
KATAKANA_HW = set(chr(c) for c in range(0xFF65, 0xFFA0)) # 半角ブロック
KANJI_COMMON = set(chr(c) for c in range(0x4E00, 0xA000)) # CJK統合漢字 (U+4E00 - U+9FFF)
KANJI_EXT_A = set(chr(c) for c in range(0x3400, 0x4DC0)) # CJK統合漢字拡張A (U+3400 - U+4DBF)
JP_PUNCT = set("、。「」『』【】・（）：；？！") # 一般的な日本語の句読点
JP_FULLWIDTH_SYMBOLS = set(chr(c) for c in range(0xFF01, 0xFF65)) # 全角英数字・記号

ENGLISH_LOWER = set(chr(c) for c in range(ord('a'), ord('z') + 1))
ENGLISH_UPPER = set(chr(c) for c in range(ord('A'), ord('Z') + 1))
ENGLISH_BASIC = ENGLISH_LOWER | ENGLISH_UPPER

def is_japanese_related_char(char: str) -> bool:
    """文字がひらがな、カタカナ、漢字、一般的な日本語の句読点、または全角/半角記号であるかを確認します。"""
    return (char in HIRAGANA or char in KATAKANA or char in KATAKANA_HW or
            char in KANJI_COMMON or char in KANJI_EXT_A or
            char in JP_PUNCT or char in JP_FULLWIDTH_SYMBOLS)

def is_pure_japanese_script_char(char: str) -> bool:
    """文字が厳密にひらがな、カタカナ、または漢字であるかを確認します。"""
    return char in HIRAGANA or char in KATAKANA or char in KATAKANA_HW or char in KANJI_COMMON or char in KANJI_EXT_A

# --- その他のヘルパー関数 ---
def is_special_char_pattern(token: str) -> bool:
    """トークンが英数字、空白、および定義された言語文字以外の文字のみで構成されているかを確認します。"""
    if not token: return False
    # すべての文字が英数字でなく、空白でなく、定義された言語セットに含まれていないことを確認
    return all(not c.isalnum() and not c.isspace() and
               c not in HIRAGANA and c not in KATAKANA and c not in KATAKANA_HW and
               c not in KANJI_COMMON and c not in KANJI_EXT_A and
               c not in JP_PUNCT and c not in JP_FULLWIDTH_SYMBOLS
               for c in token)

# --- メイン分析関数 ---

def analyze_token_categories(model_id: str, min_token_id: int = 0) -> Dict[str, Any]:
    """
    トークナイザーの語彙を分析し、日本語、英語、その他のカテゴリに分類します。
    特殊トークンは除外されます。
    """
    logging.info(f"モデルの分析を開始: {model_id}")

    # --- 1. トークナイザーと語彙情報の読み込み ---
    try:
        # カスタムトークナイザー/モデルには trust_remote_code=True が必要な場合が多い
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        vocab_size = tokenizer.vocab_size
        max_token_id = vocab_size - 1
        special_ids = set(tokenizer.all_special_ids)
        if tokenizer.pad_token_id is not None: special_ids.add(tokenizer.pad_token_id)
        logging.info(f"トークナイザー読み込み完了。語彙サイズ: {vocab_size}。特殊ID数: {len(special_ids)}")
    except Exception as e:
        logging.error(f"{model_id} のトークナイザー読み込みに失敗しました: {e}")
        return None

    # --- 2. カテゴリ定義と反復準備 ---
    categories: Dict[str, Set[int]] = {
        "pure_english": set(),          # 基本的なラテン文字のみ (a-z, A-Z)
        "contains_japanese": set(),       # ひらがな、カタカナ、漢字、日本語の句読点/記号を含む
        "pure_japanese_script": set(),  # ひらがな、カタカナ、漢字のみ (句読点/記号/他言語なし)
        "contains_digit": set(),        # 数字 (0-9) を含む
        "special_char_pattern": set(),  # 英数字、空白、言語文字以外の文字のみ
        "uncategorized": set(),         # 上記いずれにも分類されない
    }

    # 分析対象のトークンID (範囲内、特殊トークンを除く)
    analyzed_token_ids = {
        token_id for token_id in range(vocab_size)
        if min_token_id <= token_id <= max_token_id and token_id not in special_ids
    }
    logging.info(f"{len(analyzed_token_ids)} 個の非特殊トークンを分析します (ID範囲: {min_token_id} ～ {max_token_id})")

    # --- 3. トークンの反復処理と分類 ---
    error_count = 0
    for token_id in tqdm(analyzed_token_ids, desc="トークン分析中"):
        try:
            # トークン文字列をデコード
            decoded_token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if not decoded_token: continue # 空のトークンはスキップ

            # 現在のトークンの文字フラグ
            has_english = False
            has_japanese = False
            has_pure_jp_script = False
            has_digit = False

            # 文字の分析
            all_chars_pure_jp_script = True
            all_chars_english_basic = True
            char_set = set(decoded_token) # 高速化のためユニーク文字でチェック

            if not char_set.isdisjoint(ENGLISH_BASIC): has_english = True
            if not char_set.isdisjoint(HIRAGANA | KATAKANA | KATAKANA_HW | KANJI_COMMON | KANJI_EXT_A | JP_PUNCT | JP_FULLWIDTH_SYMBOLS): has_japanese = True
            if not char_set.isdisjoint(set("0123456789")): has_digit = True

            # 純粋性フラグの確認 (全文字をチェック)
            for char in decoded_token:
                is_pure_jp_script_char = char in HIRAGANA or char in KATAKANA or char in KATAKANA_HW or char in KANJI_COMMON or char in KANJI_EXT_A
                is_basic_english_char = char in ENGLISH_BASIC

                if is_pure_jp_script_char: has_pure_jp_script = True # 少なくとも1つ純粋な文字があればマーク
                else: all_chars_pure_jp_script = False # ひらがな/カタカナ/漢字以外があれば純粋ではない

                if not is_basic_english_char: all_chars_english_basic = False

            # フラグに基づいてカテゴリに割り当て
            assigned = False
            if has_japanese:
                categories["contains_japanese"].add(token_id)
                # 純粋な日本語スクリプト: ひらがな、カタカナ、漢字のみを含む
                if has_pure_jp_script and all_chars_pure_jp_script:
                    categories["pure_japanese_script"].add(token_id)
                assigned = True # 言語コンテンツを優先

            # 日本語コンテンツが検出されなかった場合のみ、純粋な英語を割り当て
            if has_english and all_chars_english_basic and not has_japanese:
                categories["pure_english"].add(token_id)
                assigned = True

            # 言語/英語に分類されていない場合、数字を含むか確認
            if has_digit and not assigned:
                 categories["contains_digit"].add(token_id)
                 assigned = True

            # 上記に分類されていない場合、特殊文字パターンを確認
            if not assigned and is_special_char_pattern(decoded_token):
                 categories["special_char_pattern"].add(token_id)
                 assigned = True

            # すべてのチェック後も割り当てられていない場合、未分類に
            if not assigned:
                 categories["uncategorized"].add(token_id)

        except Exception as e:
            error_count += 1
            if error_count <= 20: # 最初のエラー数件のみログ記録
                logging.warning(f"トークンID {token_id} の分析中にエラーが発生しました: {e}", exc_info=False)
            continue

    if error_count > 20:
        logging.warning(f"トークン分析中に {error_count} 件のエラーが発生しました。20件目以降のログは抑制されました。")

    # --- 4. 結果の集計 ---
    # 最終的な未分類を再計算 (ロジック変更時の差異を吸収)
    all_categorized_ids = set().union(*[s for k, s in categories.items() if k != "uncategorized"])
    final_uncategorized_ids = analyzed_token_ids - all_categorized_ids
    categories["uncategorized"] = final_uncategorized_ids # セットを更新

    logging.info("分析完了。結果を集計中...")
    analysis_result = {
        'model_id': model_id,
        'vocab_size': vocab_size,
        'num_special_tokens': len(special_ids),
        'analysis_details': {
             'min_token_id_analyzed': min_token_id,
             'max_token_id_analyzed': max_token_id,
             'num_tokens_analyzed': len(analyzed_token_ids),
             'num_errors': error_count,
             'excluded_special_ids': sorted(list(special_ids)),
        },
        'statistics': {name: len(ids) for name, ids in categories.items()},
        'token_ids': {name: sorted(list(ids)) for name, ids in categories.items()}
    }

    return analysis_result

# --- ファイル保存および要約表示関数 ---

def save_analysis_results(analysis_result: Dict[str, Any], output_dir: str, base_filename: str = "token_analysis_jp"):
    """詳細な分析結果をJSONファイルに保存します。"""
    if not analysis_result:
        logging.error("保存する分析結果がありません。")
        return
    try:
        # 一意のファイル名のためにモデル名の一部を使用
        model_name_part = analysis_result['model_id'].split('/')[-1].replace('-', '_')
        output_filename = f"{base_filename}_{model_name_part}.json"
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        logging.info(f"分析結果を保存しました: {output_path}")
    except Exception as e:
        logging.error(f"分析結果の保存中にエラーが発生しました ({output_path}): {e}")

def save_token_list(token_ids: List[int], category_name: str, output_dir: str, model_id: str):
    """トークンIDのリストをテキストファイルに保存します。"""
    if not token_ids:
        logging.info(f"カテゴリ '{category_name}' にトークンが見つかりませんでした。ファイル保存をスキップします。")
        return
    try:
        model_name_part = model_id.split('/')[-1].replace('-', '_')
        output_filename = f"{category_name}_{model_name_part}.txt"
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            ids_string = ",".join(map(str, token_ids))
            # ファイル内で一貫した変数名形式を使用
            f.write(f"{category_name}_ids = [{ids_string}]")
        logging.info(f"{len(token_ids)} 個の '{category_name}' トークンIDを保存しました: {output_path}")
    except Exception as e:
        logging.error(f"'{category_name}' トークンIDリストの保存中にエラーが発生しました: {e}")

def print_analysis_summary(analysis_result: Dict[str, Any]):
    """分析統計の要約を表示します。"""
    if not analysis_result: return

    stats = analysis_result.get('statistics', {})
    details = analysis_result.get('analysis_details', {})

    print("\n" + "=" * 50)
    print(f" 分析サマリー: {analysis_result.get('model_id', 'N/A')}")
    print("-" * 50)
    print(f"  語彙サイズ: {analysis_result.get('vocab_size', 0):,}")
    print(f"  除外された特殊トークン数: {analysis_result.get('num_special_tokens', 0):,}")
    print(f"  分析対象トークンID範囲: {details.get('min_token_id_analyzed', 'N/A')} - {details.get('max_token_id_analyzed', 'N/A')}")
    print(f"  分析された非特殊トークン数: {details.get('num_tokens_analyzed', 0):,}")
    print(f"  分析中のエラー数: {details.get('num_errors', 0):,}")
    print("-" * 50)
    print( "  カテゴリ別トークン数:")
    print(f"    純粋な英語:             {stats.get('pure_english', 0):,}")
    print(f"    日本語を含む:           {stats.get('contains_japanese', 0):,}")
    print(f"    純粋な日本語スクリプト: {stats.get('pure_japanese_script', 0):,}")
    print(f"    数字を含む:             {stats.get('contains_digit', 0):,}")
    print(f"    特殊文字パターン:       {stats.get('special_char_pattern', 0):,}")
    print(f"    未分類:                 {stats.get('uncategorized', 0):,}")
    print("=" * 50 + "\n")

def print_example_tokens(model_id: str, category_name: str, token_ids: List[int], max_tokens: int = 10):
    """特定のカテゴリのトークン例を表示します。"""
    if not token_ids: return

    print(f"\n--- トークン例: {category_name} (最大 {max_tokens} 件) ---")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        ids_to_show = token_ids[:max_tokens]
        for token_id in ids_to_show:
            try:
                token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                print(f"  ID: {token_id:<6d} | Token: {repr(token)}")
            except Exception as e: print(f"  ID: {token_id:<6d} | デコードエラー: {e}")
    except Exception as e: logging.error(f"トークン例表示のためのトークナイザー読み込みエラー: {e}")
    print("-" * (len(category_name) + 24))

# --- メイン実行ロジック ---

def run_full_analysis(model_id: str, min_token_id: int, output_dir: str):
    """分析を実行し、関連するすべての出力を保存します。"""
    analysis_result = analyze_token_categories(model_id, min_token_id)

    if analysis_result:
        # 詳細なJSON結果を保存
        save_analysis_results(analysis_result, output_dir) # base_filenameはデフォルトの"token_analysis_jp"を使用

        # バイアス調整やさらなる分析に役立つトークンリストを保存
        token_ids = analysis_result.get('token_ids', {})
        save_token_list(token_ids.get('contains_japanese', []), 'contains_japanese', output_dir, model_id)
        save_token_list(token_ids.get('pure_japanese_script', []), 'pure_japanese_script', output_dir, model_id)
        save_token_list(token_ids.get('pure_english', []), 'pure_english', output_dir, model_id)
        save_token_list(token_ids.get('uncategorized', []), 'uncategorized', output_dir, model_id)

        # サマリーと例を表示
        print_analysis_summary(analysis_result)
        print_example_tokens(model_id, 'Uncategorized', token_ids.get('uncategorized', []), max_tokens=20)
        print_example_tokens(model_id, 'Contains Japanese', token_ids.get('contains_japanese', []), max_tokens=10)
        print_example_tokens(model_id, 'Pure Japanese Script', token_ids.get('pure_japanese_script', []), max_tokens=10)

    else:
        logging.error("分析を完了できませんでした。")

def main():
    parser = argparse.ArgumentParser(
        description="日本語トークナイザーアナライザー - 語彙を分析し、日本語コンテンツを分類します。"
    )
    parser.add_argument('--model_id', type=str, required=True,
                        help="分析対象のモデルのHugging Face IDまたはローカルパス。")
    parser.add_argument('--min_token_id', type=int, default=102,
                        help="分析範囲の最小トークンID (デフォルト: 0)。特殊トークンは常に除外されます。")
    parser.add_argument('--output_dir', type=str, default='.',
                        help="分析結果 (JSONおよびTXTリスト) を保存するディレクトリ。")

    args = parser.parse_args()

    run_full_analysis(args.model_id, args.min_token_id, args.output_dir)

if __name__ == "__main__":
    main()
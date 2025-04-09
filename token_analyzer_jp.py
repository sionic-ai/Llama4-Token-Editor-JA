#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日本語用トークナイザーアナライザー
上位のSlack議論で提示された「日本語文字に関わる全範囲（ひらがな、カタカナ全角・半角、漢字（基本～拡張B-F）、全角英数字、CJK互換漢字など）」を
できる限り網羅するように改修した完全版のコードです。

すべての注釈・コメントは日本語のみを使用し、韓国語その他言語は含んでいません。
記号や変数名の英語表記（例: tokenizer, vocab_size）はPythonの一般的慣習として残していますが、
説明やコメントはすべて日本語で記載しています。
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Set, Any
from tqdm import tqdm
import transformers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================================
# ここから先は日本語文字判定用のユニコード範囲を定義します。
# Slackで挙がった話題をすべて含めるため、下記のように広範に設定しています。
# =========================================================================

# ひらがな (U+3040 ~ U+309F)
HIRAGANA = set(chr(c) for c in range(0x3040, 0x30A0))

# 全角カタカナ (U+30A0 ~ U+30FF)
KATAKANA = set(chr(c) for c in range(0x30A0, 0x3100))

# 半角カタカナ (U+FF65 ~ U+FF9F)
KATAKANA_HW = set(chr(c) for c in range(0xFF65, 0xFFA0))

# CJK統合漢字 (U+4E00 ~ U+9FFF)
KANJI_COMMON = set(chr(c) for c in range(0x4E00, 0xA000))

# CJK統合漢字拡張A (U+3400 ~ U+4DBF)
KANJI_EXT_A = set(chr(c) for c in range(0x3400, 0x4DC0))

# CJK統合漢字拡張B-F (U+20000 ~ U+2FA1F)
# 本来は拡張B: 0x20000~0x2A6DF, C, D, E, F, 互換補助など複数範囲がありますが、
# 簡易的にまとめて 0x20000 ~ 0x2FA1F とします。
KANJI_EXT_B_TO_F = set(chr(c) for c in range(0x20000, 0x2FA20))

# CJK互換漢字 (U+F900 ~ U+FAFF)
KANJI_COMPAT = set(chr(c) for c in range(0xF900, 0xFB00))

# 日本語でよく使われる句読点・記号 (全角・半角)
JP_PUNCT = set("、。「」『』【】・（）：；？！｡｢｣､")
JP_SYMBOLS_ETC = set("　〜・￥")

# 全角ASCII印字可能文字 (U+FF01 ~ U+FF5E 程度)
JP_FULLWIDTH_ASCII_PRINTABLE = set(chr(c) for c in range(0xFF01, 0xFF5F))

# 全角数字 (U+FF10 ~ U+FF19)
JP_FULLWIDTH_DIGITS = set(chr(c) for c in range(0xFF10, 0xFF1A))

# 全角英大文字 (U+FF21 ~ U+FF3A), 全角英小文字 (U+FF41 ~ U+FF5A)
JP_FULLWIDTH_LATIN_UPPER = set(chr(c) for c in range(0xFF21, 0xFF3B))
JP_FULLWIDTH_LATIN_LOWER = set(chr(c) for c in range(0xFF41, 0xFF5B))
JP_FULLWIDTH_LATIN = JP_FULLWIDTH_LATIN_UPPER | JP_FULLWIDTH_LATIN_LOWER

# 基本的な英語文字セット (ASCII a-z, A-Z)
ENGLISH_LOWER = set(chr(c) for c in range(ord("a"), ord("z") + 1))
ENGLISH_UPPER = set(chr(c) for c in range(ord("A"), ord("Z") + 1))
ENGLISH_BASIC = ENGLISH_LOWER | ENGLISH_UPPER

# =========================================================================
# 日本語関連文字かどうかを判定するためのヘルパー関数
# =========================================================================


def is_japanese_related_char(char: str) -> bool:
    """
    文字が日本語の関連文字（ひらがな、カタカナ(全角/半角)、漢字(基本～拡張B-F, 互換)、
    全角英数字、全角記号、主要句読点など）に該当するかを判定します。
    """
    if (
        char in HIRAGANA
        or char in KATAKANA
        or char in KATAKANA_HW
        or char in KANJI_COMMON
        or char in KANJI_EXT_A
        or char in KANJI_EXT_B_TO_F
        or char in KANJI_COMPAT
        or char in JP_PUNCT
        or char in JP_SYMBOLS_ETC
        or char in JP_FULLWIDTH_ASCII_PRINTABLE
        or char in JP_FULLWIDTH_DIGITS
        or char in JP_FULLWIDTH_LATIN
    ):
        return True
    return False


def is_pure_japanese_script_char(char: str) -> bool:
    """
    文字が厳密に「日本語の書記体系（ひらがな、カタカナ(全角/半角)、漢字）」のみを構成するか判定。
    記号や全角英数字は含まない。
    """
    if (
        char in HIRAGANA
        or char in KATAKANA
        or char in KATAKANA_HW
        or char in KANJI_COMMON
        or char in KANJI_EXT_A
        or char in KANJI_EXT_B_TO_F
        # or char in KANJI_COMPAT
    ):
        return True
    return False


def is_special_char_pattern(token: str) -> bool:
    """
    トークンが下記に該当しない文字のみで構成されるかを判定:
      - ASCII英数字 (isalnum)
      - スペース (isspace)
      - 日本語関連文字 (is_japanese_related_char)
      - 基本英語 (ENGLISH_BASIC)
    これらのいずれにも当てはまらない文字だけで構成されていれば True。
    """
    if not token:
        return False

    for c in token:
        if c.isalnum():
            return False
        if c.isspace():
            return False
        if c in ENGLISH_BASIC:
            return False
        if is_japanese_related_char(c):
            return False
    return True


# =========================================================================
# トークンの分析メインロジック
# =========================================================================


def analyze_token_categories(model_id: str, min_token_id: int = 0) -> Dict[str, Any]:
    """
    指定モデルのトークナイザーをロードし、min_token_id 以上の通常トークンを解析して
    各種カテゴリに仕分けし、その結果を返す。
    特殊トークンは対象外。
    """
    logging.info(f"分析開始: {model_id}")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
    except Exception as e:
        logging.error(f"トークナイザー読み込み失敗: {e}")
        return {}

    vocab_size = tokenizer.vocab_size
    max_token_id = vocab_size - 1

    special_ids = set(tokenizer.all_special_ids)
    common_special_tokens = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.unk_token_id,
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }
    for sid in common_special_tokens:
        if sid is not None:
            special_ids.add(sid)

    logging.info(f"語彙サイズ: {vocab_size}")
    logging.info(f"特殊トークン数: {len(special_ids)}")

    categories: Dict[str, Set[int]] = {
        "contains_japanese": set(),
        "pure_japanese_script": set(),
        "pure_english": set(),
        "contains_hiragana": set(),
        "contains_katakana_full": set(),
        "contains_katakana_half": set(),
        "contains_kanji": set(),
        "contains_jp_punct_symbol": set(),
        "contains_fullwidth_ascii": set(),
        "contains_basic_english": set(),
        "contains_digit": set(),
        "special_char_pattern": set(),
        "uncategorized": set(),
    }

    targets = [
        tid
        for tid in range(vocab_size)
        if (tid >= min_token_id and tid not in special_ids)
    ]
    if not targets:
        logging.warning(
            "分析対象トークンがありません。min_token_idの設定を確認してください。"
        )
        return {
            "model_id": model_id,
            "vocab_size": vocab_size,
            "num_special_tokens": len(special_ids),
            "analysis_details": {
                "min_token_id_analyzed": min_token_id,
                "max_token_id_analyzed": max_token_id,
                "num_tokens_analyzed": 0,
                "num_errors": 0,
                "excluded_special_ids": sorted(list(special_ids)),
            },
            "statistics": {k: 0 for k in categories},
            "token_ids": {k: [] for k in categories},
        }

    error_count = 0
    for token_id in tqdm(targets, desc="トークン解析中", unit="token"):
        try:
            decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if not decoded:
                continue

            has_hira = False
            has_kata_f = False
            has_kata_h = False
            has_kanji = False
            has_jp_punc = False
            has_fw_ascii = False
            has_basic_eng = False
            has_digit_flag = False

            all_pure_jp = True
            all_basic_eng = True

            for ch in decoded:
                # カテゴリ判定フラグ
                if ch in HIRAGANA:
                    has_hira = True
                if ch in KATAKANA:
                    has_kata_f = True
                if ch in KATAKANA_HW:
                    has_kata_h = True
                if (
                    ch in KANJI_COMMON
                    or ch in KANJI_EXT_A
                    or ch in KANJI_EXT_B_TO_F
                    or ch in KANJI_COMPAT
                ):
                    has_kanji = True
                if (ch in JP_PUNCT) or (ch in JP_SYMBOLS_ETC):
                    has_jp_punc = True
                if (
                    ch in JP_FULLWIDTH_ASCII_PRINTABLE
                    or ch in JP_FULLWIDTH_DIGITS
                    or ch in JP_FULLWIDTH_LATIN
                ):
                    has_fw_ascii = True
                if ch in ENGLISH_BASIC:
                    has_basic_eng = True
                if "0" <= ch <= "9":
                    has_digit_flag = True

                # 純粋な日本語だけで構成されているか
                if not is_pure_japanese_script_char(ch):
                    all_pure_jp = False

                # 純粋な英語だけで構成されているか
                if ch not in ENGLISH_BASIC:
                    all_basic_eng = False

            # カテゴリ分類
            is_jp_related = (
                has_hira
                or has_kata_f
                or has_kata_h
                or has_kanji
                or has_jp_punc
                or has_fw_ascii
            )
            if is_jp_related:
                categories["contains_japanese"].add(token_id)
                if has_hira:
                    categories["contains_hiragana"].add(token_id)
                if has_kata_f:
                    categories["contains_katakana_full"].add(token_id)
                if has_kata_h:
                    categories["contains_katakana_half"].add(token_id)
                if has_kanji:
                    categories["contains_kanji"].add(token_id)
                if has_jp_punc:
                    categories["contains_jp_punct_symbol"].add(token_id)
                if has_fw_ascii:
                    categories["contains_fullwidth_ascii"].add(token_id)

                if all_pure_jp and (has_hira or has_kata_f or has_kata_h or has_kanji):
                    categories["pure_japanese_script"].add(token_id)

            if has_basic_eng:
                categories["contains_basic_english"].add(token_id)
                if all_basic_eng and (not is_jp_related):
                    categories["pure_english"].add(token_id)

            if has_digit_flag:
                categories["contains_digit"].add(token_id)

            # 特殊文字パターン
            if (
                not is_jp_related
                and not has_basic_eng
                and not has_digit_flag
                and is_special_char_pattern(decoded)
            ):
                categories["special_char_pattern"].add(token_id)

        except Exception as e:
            error_count += 1
            if error_count <= 20:
                logging.warning(f"トークンID {token_id} の解析中にエラー: {e}")
            continue

    # 未分類を特定
    all_categorized = set()
    for cname, cset in categories.items():
        if cname == "uncategorized":
            continue
        all_categorized |= cset

    uncategorized_set = set(targets) - all_categorized
    categories["uncategorized"] = uncategorized_set

    # 結果整理
    analysis_result = {
        "model_id": model_id,
        "vocab_size": vocab_size,
        "num_special_tokens": len(special_ids),
        "analysis_details": {
            "min_token_id_analyzed": min_token_id,
            "max_token_id_analyzed": max_token_id,
            "num_tokens_analyzed": len(targets),
            "num_errors": error_count,
            "excluded_special_ids": sorted(list(special_ids)),
        },
        "statistics": {k: len(v) for k, v in categories.items()},
        "token_ids": {k: sorted(list(v)) for k, v in categories.items()},
    }
    return analysis_result


# =========================================================================
# 分析結果の保存・表示関数
# =========================================================================


def save_analysis_results(
    analysis_result: Dict[str, Any],
    output_dir: str = "token_analysis_output",
    base_filename: str = "token_analysis_jp",
):
    """
    JSONファイルとして分析結果を保存します。
    """
    if not analysis_result:
        logging.error("分析結果がありません。保存をスキップします。")
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_name_part = (
            os.path.basename(analysis_result["model_id"])
            .replace("/", "_")
            .replace("-", "_")
        )
        output_path = os.path.join(
            output_dir, f"{base_filename}_{model_name_part}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        logging.info(f"JSON保存完了: {output_path}")
    except Exception as e:
        logging.error(f"JSON保存中にエラーが発生: {e}")


def save_token_list(
    token_ids: List[int], category_name: str, output_dir: str, model_id: str
):
    """
    指定カテゴリのトークンID一覧をテキストファイルに書き出します。
    """
    if not token_ids:
        logging.info(
            f"'{category_name}' に該当トークンがありません。保存をスキップします。"
        )
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        model_name_part = os.path.basename(model_id).replace("/", "_").replace("-", "_")
        filename = f"{category_name}_{model_name_part}.txt"
        outpath = os.path.join(output_dir, filename)
        with open(outpath, "w", encoding="utf-8") as f:
            id_str = ",".join(str(i) for i in token_ids)
            f.write(f"{category_name}_ids = [{id_str}]\n")
        logging.info(
            f"{len(token_ids)} 個の'{category_name}'トークンIDを保存しました: {outpath}"
        )
    except Exception as e:
        logging.error(f"'{category_name}'トークンIDリストの保存中にエラー: {e}")


def print_analysis_summary(analysis_result: Dict[str, Any]):
    """
    統計情報をコンソールに出力します。
    """
    if not analysis_result:
        logging.warning("分析結果がありません。表示をスキップします。")
        return

    st = analysis_result["statistics"]
    dt = analysis_result["analysis_details"]
    print("\n==============================")
    print(f"モデルID: {analysis_result['model_id']}")
    print(f"語彙サイズ: {analysis_result['vocab_size']}")
    print(f"特殊トークン数: {analysis_result['num_special_tokens']}")
    print(f"解析対象トークン数: {dt['num_tokens_analyzed']}")
    print(f"解析中エラー数: {dt['num_errors']}")
    print("------------------------------")
    print("カテゴリ別トークン数:")
    print(f"  日本語関連: {st['contains_japanese']}")
    print(f"     純粋日本語: {st['pure_japanese_script']}")
    print(f"     ひらがな含む: {st['contains_hiragana']}")
    print(f"     全角カタカナ含む: {st['contains_katakana_full']}")
    print(f"     半角カタカナ含む: {st['contains_katakana_half']}")
    print(f"     漢字含む: {st['contains_kanji']}")
    print(f"     日本語句読点/記号含む: {st['contains_jp_punct_symbol']}")
    print(f"     全角ASCII含む: {st['contains_fullwidth_ascii']}")
    print(f"  基本英語含む: {st['contains_basic_english']}")
    print(f"     純粋英語: {st['pure_english']}")
    print(f"  数字含む: {st['contains_digit']}")
    print(f"  特殊文字パターン: {st['special_char_pattern']}")
    print(f"  未分類: {st['uncategorized']}")
    print("==============================\n")


def print_example_tokens(
    model_id: str, category_name: str, token_ids: List[int], max_tokens: int = 10
):
    """
    指定カテゴリのトークンIDを数件サンプル表示します。
    """
    if not token_ids:
        print(f"\n--- {category_name} のトークン例 ---")
        print("  該当トークンはありません。")
        return

    print(f"\n--- {category_name} のトークン例 (最大{max_tokens}件) ---")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    for tid in token_ids[:max_tokens]:
        try:
            txt = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            print(f"  ID: {tid:<6d} | Token: {repr(txt)}")
        except Exception as e:
            print(f"  ID: {tid:<6d} | デコード失敗: {e}")


# =========================================================================
# メインの処理フロー
# =========================================================================


def run_full_analysis(
    model_id: str, min_token_id: int = 0, output_dir: str = "token_analysis_output"
):
    """
    トークン解析のフルプロセス:
      1) 解析実行
      2) JSON結果保存
      3) カテゴリ別トークンIDリスト保存
      4) 統計表示
      5) 例示表示
    """
    logging.info("=== トークン解析開始 ===")
    logging.info(f"モデルID: {model_id}")
    logging.info(f"最小トークンID: {min_token_id}")
    logging.info(f"出力先ディレクトリ: {output_dir}")

    result = analyze_token_categories(model_id, min_token_id)
    if not result:
        logging.error("解析結果が空です。処理を終了します。")
        return

    save_analysis_results(result, output_dir=output_dir)

    token_ids_map = result["token_ids"]
    categories_to_save = [
        "contains_japanese",
        "pure_japanese_script",
        "contains_hiragana",
        "contains_katakana_full",
        "contains_katakana_half",
        "contains_kanji",
        "contains_fullwidth_ascii",
        "pure_english",
        "special_char_pattern",
        "uncategorized",
    ]
    for cat in categories_to_save:
        save_token_list(token_ids_map[cat], cat, output_dir, result["model_id"])

    print_analysis_summary(result)

    # 例表示を行うカテゴリ
    example_targets = [
        ("uncategorized", 15),
        ("special_char_pattern", 15),
        ("pure_japanese_script", 10),
        ("contains_japanese", 10),
        ("pure_english", 10),
    ]
    for cat_name, max_show in example_targets:
        print_example_tokens(model_id, cat_name, token_ids_map[cat_name], max_show)

    logging.info("=== トークン解析完了 ===")


def main():
    """
    コマンドライン引数を処理し、分析を実行するスクリプトのエントリーポイント。
    """
    parser = argparse.ArgumentParser(
        description="日本語トークナイザーアナライザー: 広範囲の日本語関連文字を含むトークンを分類し、統計を出力します。"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="unsloth/Llama-4-Scout-17B-16E-Instruct",
        required=False,
        help="分析対象のモデル（ローカルパスまたはHugging FaceモデルID）",
    )
    parser.add_argument(
        "--min_token_id", type=int, default=102, help="分析を開始する最小トークンID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="token_analysis_output",
        help="分析結果を保存するディレクトリ",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログ出力レベル",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    run_full_analysis(
        model_id=args.model_id,
        min_token_id=args.min_token_id,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

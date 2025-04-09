# token_analyzer_jp.py
import transformers
import json
import argparse
from typing import Dict, List, Set, Any
from tqdm import tqdm
import os
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 文字セット定義 ---
# Unicode範囲に基づいて、高速チェックのためのセットを使用

# 基本的な日本語文字セット
HIRAGANA = set(chr(c) for c in range(0x3040, 0x30A0))  # ひらがな (ぁ-ん)
KATAKANA = set(chr(c) for c in range(0x30A0, 0x3100))  # 全角カタカナ (ァ-ヶ, ヷ-ヺ, ー)
KATAKANA_HW = set(
    chr(c) for c in range(0xFF65, 0xFFA0)
)  # 半角カタカナ (･-ﾟ) + 濁点/半濁点
KANJI_COMMON = set(
    chr(c) for c in range(0x4E00, 0xA000)
)  # CJK統合漢字 (U+4E00 - U+9FFF)
KANJI_EXT_A = set(
    chr(c) for c in range(0x3400, 0x4DC0)
)  # CJK統合漢字拡張A (U+3400 - U+4DBF)

# 日本語でよく使われる句読点・記号
JP_PUNCT = set("、。「」『』【】・（）：；？！｡｢｣､")  # 全角・半角の主要な句読点
JP_SYMBOLS_ETC = set("　〜・￥")  # 全角スペース、波ダッシュ、中点、円マークなど

# 全角ASCII文字 (Slackでの議論に基づき追加)
JP_FULLWIDTH_DIGITS = set(chr(c) for c in range(0xFF10, 0xFF1A))  # 全角数字 (０-９)
JP_FULLWIDTH_LATIN_UPPER = set(
    chr(c) for c in range(0xFF21, 0xFF3B)
)  # 全角英大文字 (Ａ-Ｚ)
JP_FULLWIDTH_LATIN_LOWER = set(
    chr(c) for c in range(0xFF41, 0xFF5B)
)  # 全角英小文字 (ａ-ｚ)
JP_FULLWIDTH_LATIN = JP_FULLWIDTH_LATIN_UPPER | JP_FULLWIDTH_LATIN_LOWER
JP_FULLWIDTH_ASCII_PRINTABLE = set(
    chr(c) for c in range(0xFF01, 0xFF5F)
)  # 全角ASCII印字可能文字 (! から ~ まで)

# 基本的な英語文字セット
ENGLISH_LOWER = set(chr(c) for c in range(ord("a"), ord("z") + 1))  # a-z
ENGLISH_UPPER = set(chr(c) for c in range(ord("A"), ord("Z") + 1))  # A-Z
ENGLISH_BASIC = ENGLISH_LOWER | ENGLISH_UPPER  # A-Z, a-z

# --- ヘルパー関数 (文字種判定) ---


def is_japanese_related_char(char: str) -> bool:
    """
    文字が日本語に関連するカテゴリ（ひらがな、カタカナ(全/半)、漢字、
    日本語の句読点/記号、全角ASCII文字）のいずれかに属するかを確認します。
    """
    return (
        char in HIRAGANA
        or char in KATAKANA
        or char in KATAKANA_HW
        or char in KANJI_COMMON
        or char in KANJI_EXT_A
        or char in JP_PUNCT
        or char in JP_SYMBOLS_ETC
        or char in JP_FULLWIDTH_ASCII_PRINTABLE
    )  # 全角ASCII文字を含むように更新


def is_pure_japanese_script_char(char: str) -> bool:
    """
    文字が厳密に「日本語の書記体系の文字」（ひらがな、カタカナ(全/半)、漢字）
    であるかを確認します。句読点や記号は含みません。
    """
    # 定義は変更なし：ひらがな、カタカナ（全角・半角）、漢字のみ
    return (
        char in HIRAGANA
        or char in KATAKANA
        or char in KATAKANA_HW
        or char in KANJI_COMMON
        or char in KANJI_EXT_A
    )


def is_special_char_pattern(token: str) -> bool:
    """
    トークンが、英数字(ASCII)、空白、および定義済みの主要な言語文字/記号
    （日本語関連、基本英語）以外の文字のみで構成されているかを確認します。
    """
    if not token:
        return False
    # isalnum() (ASCII英数字), isspace() でもなく、
    # 定義済みの日本語関連文字セット、基本英語文字セットのいずれにも含まれない文字のみか？
    return all(
        not c.isalnum()
        and not c.isspace()
        and not is_japanese_related_char(
            c
        )  # 日本語関連を除外 (これで全角ASCIIも除外される)
        and c not in ENGLISH_BASIC  # 基本英語を除外
        for c in token
    )


# --- メイン分析関数 ---


def analyze_token_categories(model_id: str, min_token_id: int = 0) -> Dict[str, Any]:
    """
    指定されたモデルのトークナイザー語彙を分析し、トークンをカテゴリに分類します。
    - 特殊トークンは分析対象から除外されます。
    - トークンは複数のカテゴリに属することがあります。
    - 日本語の部分トークン問題に対応するため、日本語関連文字を含むトークン(`contains_japanese`)
      を包括的に識別します。
    """
    logging.info(f"モデルの分析を開始: {model_id}")

    # --- 1. トークナイザーと語彙情報の読み込み ---
    try:
        # カスタムトークナイザーのために trust_remote_code=True を使用
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        vocab_size = tokenizer.vocab_size
        max_token_id = vocab_size - 1

        # 特殊トークンIDの収集 (all_special_idsが不完全な場合も考慮)
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
        for token_id in common_special_tokens:
            if token_id is not None:
                special_ids.add(token_id)

        logging.info(
            f"トークナイザー読み込み完了。語彙サイズ: {vocab_size}。特殊ID数: {len(special_ids)}"
        )
        logging.debug(f"除外される特殊トークンID: {sorted(list(special_ids))}")

    except Exception as e:
        logging.error(f"{model_id} のトークナイザー読み込みに失敗しました: {e}")
        return None

    # --- 2. カテゴリ定義と準備 ---
    categories: Dict[str, Set[int]] = {
        # 主要カテゴリ
        "contains_japanese": set(),  # 日本語関連文字(下記すべてを含む可能性)を少なくとも1つ含む
        "pure_japanese_script": set(),  # ひらがな, カタカナ(全/半), 漢字「のみ」で構成
        "pure_english": set(),  # 基本的なラテン文字(a-z, A-Z)「のみ」で構成
        # 詳細カテゴリ (日本語関連の部分集合または関連カテゴリ)
        "contains_hiragana": set(),  # ひらがなを含む
        "contains_katakana_full": set(),  # 全角カタカナを含む
        "contains_katakana_half": set(),  # 半角カタカナを含む
        "contains_kanji": set(),  # 漢字(常用+拡張A)を含む
        "contains_jp_punct_symbol": set(),  # 日本語の句読点・記号を含む
        "contains_fullwidth_ascii": set(),  # 全角ASCII文字(英数記号)を含む
        # その他のカテゴリ
        "contains_basic_english": set(),  # 基本的なラテン文字(a-z, A-Z)を含む (純粋でなくても)
        "contains_digit": set(),  # アラビア数字 (0-9) を含む
        "special_char_pattern": set(),  # 定義済み文字以外の特殊文字のみで構成
        "uncategorized": set(),  # 上記いずれの特性も持たない (最終計算)
    }

    # 分析対象のトークンIDセット (指定範囲内かつ特殊トークンを除く)
    analyzed_token_ids = {
        token_id
        for token_id in range(vocab_size)
        if min_token_id <= token_id <= max_token_id and token_id not in special_ids
    }
    num_analyzed = len(analyzed_token_ids)
    logging.info(
        f"{num_analyzed:,} 個の非特殊トークンを分析します (ID範囲: {min_token_id} ～ {max_token_id})"
    )
    if num_analyzed == 0:
        logging.warning(
            "分析対象のトークンがありません。min_token_id やモデルを確認してください。"
        )
        # 空の結果を返すか、エラーとするか検討。ここでは空の結果を返す。
        analysis_result = {
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
            "statistics": {name: 0 for name in categories},
            "token_ids": {name: [] for name in categories},
        }
        return analysis_result

    # --- 3. トークンの反復処理と分類 ---
    error_count = 0
    for token_id in tqdm(
        analyzed_token_ids, desc=f"トークン分析中 ({model_id})", unit="token"
    ):
        try:
            # トークン文字列をデコード
            # clean_up_tokenization_spaces=False: SentencePiece等の先頭スペース(_)を保持
            decoded_token = tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            )

            # 空文字列やデコード不能なケースはスキップ (通常は起こらないはず)
            if not decoded_token:
                logging.debug(
                    f"Token ID {token_id} resulted in empty string, skipping."
                )
                continue

            # --- 文字レベルのフラグ初期化 ---
            has_hiragana = False
            has_katakana_full = False
            has_katakana_half = False
            has_kanji = False
            has_jp_punct_symbol = False
            has_fullwidth_ascii = False
            has_basic_english = False
            has_digit = False
            has_other_char = False  # 未知のカテゴリの文字が存在するか

            # トークン全体の純粋性フラグ
            all_pure_jp_script = True
            all_basic_english = True
            all_special_pattern = True  # is_special_char_pattern の文字のみか

            # --- 文字ごとのチェック ---
            for char in decoded_token:
                # 各カテゴリへの所属チェック
                is_hira = char in HIRAGANA
                is_kata_f = char in KATAKANA
                is_kata_h = char in KATAKANA_HW
                is_kanji = char in KANJI_COMMON or char in KANJI_EXT_A
                is_jp_ps = char in JP_PUNCT or char in JP_SYMBOLS_ETC
                is_fw_ascii = char in JP_FULLWIDTH_ASCII_PRINTABLE
                is_basic_eng = char in ENGLISH_BASIC
                is_digit = "0" <= char <= "9"
                is_space = (
                    char.isspace()
                )  # スペースは特別扱いしないが、純粋性判定で考慮

                # 各「含む」フラグを更新
                if is_hira:
                    has_hiragana = True
                if is_kata_f:
                    has_katakana_full = True
                if is_kata_h:
                    has_katakana_half = True
                if is_kanji:
                    has_kanji = True
                if is_jp_ps:
                    has_jp_punct_symbol = True
                if is_fw_ascii:
                    has_fullwidth_ascii = True
                if is_basic_eng:
                    has_basic_english = True
                if is_digit:
                    has_digit = True

                # 純粋性フラグの更新
                # 1. 純粋な日本語スクリプトか？ (ひらがな, カタカナ全/半, 漢字 以外が含まれていれば False)
                is_pure_jp = is_hira or is_kata_f or is_kata_h or is_kanji
                if not is_pure_jp:
                    all_pure_jp_script = False

                # 2. 純粋な基本英語か？ (a-z, A-Z 以外が含まれていれば False)
                if not is_basic_eng:
                    all_basic_english = False

                # 3. 特殊文字パターンか？ (isalnum, isspace, 日本語関連, 基本英語 に該当すれば False)
                is_jp_related = (
                    is_pure_jp or is_jp_ps or is_fw_ascii
                )  # is_japanese_related_char と同等
                if char.isalnum() or is_space or is_jp_related or is_basic_eng:
                    all_special_pattern = False

                # その他の文字フラグ (上記のいずれでもない場合)
                if not (
                    is_jp_related
                    or is_basic_eng
                    or is_digit
                    or is_space
                    or char.isalnum()
                ):
                    # 注意: is_special_char_pattern に該当する文字はここに該当しうる
                    if not is_special_char_pattern(
                        char
                    ):  # 個別文字がspecial patternにも属さない場合
                        has_other_char = True

            # --- フラグに基づいてカテゴリに割り当て (複数のカテゴリに属しうる) ---
            # 1. 日本語関連のカテゴリ
            is_related_to_jp = (
                has_hiragana
                or has_katakana_full
                or has_katakana_half
                or has_kanji
                or has_jp_punct_symbol
                or has_fullwidth_ascii
            )
            if is_related_to_jp:
                categories["contains_japanese"].add(token_id)
                if has_hiragana:
                    categories["contains_hiragana"].add(token_id)
                if has_katakana_full:
                    categories["contains_katakana_full"].add(token_id)
                if has_katakana_half:
                    categories["contains_halfwidth_katakana"].add(
                        token_id
                    )  # 半角カタカナ含む
                if has_kanji:
                    categories["contains_kanji"].add(token_id)
                if has_jp_punct_symbol:
                    categories["contains_jp_punct_symbol"].add(token_id)
                if has_fullwidth_ascii:
                    categories["contains_fullwidth_ascii"].add(
                        token_id
                    )  # 全角ASCII含む

                # 純粋な日本語スクリプトか？ (「〜」や「、」などが含まれていない)
                # all_pure_jp_script は is_pure_japanese_script_char に該当する文字のみで構成されているかをチェック
                if all_pure_jp_script and (
                    has_hiragana or has_katakana_full or has_katakana_half or has_kanji
                ):
                    # 空白のみのトークンなどは除外するため、実際にスクリプト文字が最低1つは含まれることも確認
                    categories["pure_japanese_script"].add(token_id)

            # 2. 英語関連のカテゴリ
            if has_basic_english:
                categories["contains_basic_english"].add(token_id)
                # 純粋な英語か？ (日本語関連文字を含まず、全て基本英語文字)
                if all_basic_english and not is_related_to_jp:
                    categories["pure_english"].add(token_id)

            # 3. 数字カテゴリ
            if has_digit:
                # 全角数字「０」などは is_related_to_jp = True なのでここには来ない想定
                # 純粋なアラビア数字を含むトークン
                categories["contains_digit"].add(token_id)

            # 4. 特殊文字パターンカテゴリ
            # is_special_char_pattern() はトークン全体を評価する
            # 日本語関連でもなく、基本英語も含まず、数字も含まず、かつ特殊文字パターンに合致するか？
            # より単純に、all_special_pattern フラグを使う（ただし、このフラグは単一文字の集合判定なので厳密には異なる）
            # is_special_char_pattern 関数を直接使うのが安全
            if (
                not is_related_to_jp
                and not has_basic_english
                and not has_digit
                and is_special_char_pattern(decoded_token)
            ):
                categories["special_char_pattern"].add(token_id)

            # 注意: uncategorized はループ終了後に最終計算

        except Exception as e:
            error_count += 1
            if error_count <= 20:  # 最初のエラー数件のみログ記録
                logging.warning(
                    f"トークンID {token_id} ('{decoded_token[:20]}...') の分析中に予期せぬエラー: {e}",
                    exc_info=False,
                )
            # エラーが多発する場合はデバッグレベルで詳細情報を出す
            logging.debug(f"Error details for Token ID {token_id}", exc_info=True)
            continue  # エラーが発生しても次のトークンへ

    if error_count > 0:
        logging.warning(
            f"トークン分析中に合計 {error_count} 件のエラーが発生しました。"
        )
        if error_count > 20:
            logging.warning("エラーログは最初の20件まで表示されました。")

    # --- 4. 結果の集計 ---
    logging.info("分析完了。結果を集計中...")

    # 最終的な未分類を計算
    # まず、いずれかのカテゴリに分類されたIDの全体集合を作る
    all_categorized_ids = set()
    for name, ids in categories.items():
        if name != "uncategorized":  # 未分類カテゴリ自体は除外
            all_categorized_ids.update(ids)

    # 分析対象IDから、分類済みIDを引いたものが未分類
    final_uncategorized_ids = analyzed_token_ids - all_categorized_ids
    categories["uncategorized"] = final_uncategorized_ids  # セットを更新

    # 結果辞書の作成
    analysis_result = {
        "model_id": model_id,
        "vocab_size": vocab_size,
        "num_special_tokens": len(special_ids),
        "analysis_details": {
            "min_token_id_analyzed": min_token_id,
            "max_token_id_analyzed": max_token_id,
            "num_tokens_analyzed": num_analyzed,
            "num_errors": error_count,
            "excluded_special_ids": sorted(list(special_ids)),
        },
        # 各カテゴリのトークン数を統計情報として格納
        "statistics": {name: len(ids) for name, ids in categories.items()},
        # 各カテゴリに属するトークンIDのリストを格納 (ソート済み)
        "token_ids": {name: sorted(list(ids)) for name, ids in categories.items()},
    }

    logging.info("集計完了。")
    return analysis_result


# --- ファイル保存および要約表示関数 ---


def save_analysis_results(
    analysis_result: Dict[str, Any],
    output_dir: str,
    base_filename: str = "token_analysis_jp",
):
    """詳細な分析結果をJSONファイルに保存します。"""
    if not analysis_result:
        logging.error("保存する分析結果がありません。")
        return
    try:
        # モデル名からファイル名を生成 (ディレクトリトラバーサル対策のため basename を使用)
        model_name_part = (
            os.path.basename(analysis_result["model_id"])
            .replace("/", "_")
            .replace("-", "_")
        )
        output_filename = f"{base_filename}_{model_name_part}.json"
        output_path = os.path.join(output_dir, output_filename)
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            # ensure_ascii=False で日本語文字をそのまま出力, indent=2 で整形
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        logging.info(f"分析結果をJSONファイルに保存しました: {output_path}")
    except Exception as e:
        logging.error(
            f"分析結果のJSON保存中にエラーが発生しました ({output_path}): {e}"
        )


def save_token_list(
    token_ids: List[int], category_name: str, output_dir: str, model_id: str
):
    """指定されたカテゴリのトークンIDリストをテキストファイルに保存します。"""
    if not token_ids:
        logging.info(
            f"カテゴリ '{category_name}' にトークンが見つかりませんでした。ファイル保存をスキップします。"
        )
        return
    try:
        # モデル名からファイル名を生成
        model_name_part = os.path.basename(model_id).replace("/", "_").replace("-", "_")
        output_filename = f"{category_name}_{model_name_part}.txt"
        output_path = os.path.join(output_dir, output_filename)
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            # カンマ区切りの文字列として書き出す
            ids_string = ",".join(map(str, token_ids))
            # Pythonリスト形式で保存（他のツールでの利用を想定）
            f.write(f"{category_name}_ids = [{ids_string}]\n")
        logging.info(
            f"{len(token_ids):,} 個の '{category_name}' トークンIDをテキストファイルに保存しました: {output_path}"
        )
    except Exception as e:
        logging.error(
            f"'{category_name}' トークンIDリストのテキストファイル保存中にエラーが発生しました: {e}"
        )


def print_analysis_summary(analysis_result: Dict[str, Any]):
    """コンソールに分析結果の統計サマリーを表示します。"""
    if not analysis_result:
        logging.warning("表示する分析結果がありません。")
        return

    stats = analysis_result.get("statistics", {})
    details = analysis_result.get("analysis_details", {})
    model_id = analysis_result.get("model_id", "N/A")
    vocab_size = analysis_result.get("vocab_size", 0)
    num_special = analysis_result.get("num_special_tokens", 0)

    print("\n" + "=" * 60)
    print(f" トークン分析サマリー: {model_id}")
    print("-" * 60)
    print(f"  語彙サイズ (Vocab Size):          {vocab_size:,}")
    print(f"  除外された特殊トークン数:       {num_special:,}")
    print(
        f"  分析対象トークンID範囲:         {details.get('min_token_id_analyzed', 'N/A')} - {details.get('max_token_id_analyzed', 'N/A')}"
    )
    print(
        f"  分析された非特殊トークン数:     {details.get('num_tokens_analyzed', 0):,}"
    )
    print(f"  分析中のエラー数:               {details.get('num_errors', 0):,}")
    print("-" * 60)
    print("  カテゴリ別トークン数 (重複あり):")
    print(f"    日本語関連を含む (Contains JP):  {stats.get('contains_japanese', 0):,}")
    print(
        f"      うち純粋な日本語スクリプト:   {stats.get('pure_japanese_script', 0):,}"
    )
    print(f"        - ひらがな含む:           {stats.get('contains_hiragana', 0):,}")
    print(
        f"        - 全角カタカナ含む:       {stats.get('contains_katakana_full', 0):,}"
    )
    print(
        f"        - 半角カタカナ含む:       {stats.get('contains_halfwidth_katakana', 0):,}"
    )
    print(f"        - 漢字含む:               {stats.get('contains_kanji', 0):,}")
    print(
        f"      うち日本語句読点/記号含む:  {stats.get('contains_jp_punct_symbol', 0):,}"
    )
    print(
        f"      うち全角ASCII文字含む:      {stats.get('contains_fullwidth_ascii', 0):,}"
    )
    print(
        f"    基本英語を含む (Contains EN):    {stats.get('contains_basic_english', 0):,}"
    )
    print(f"      うち純粋な基本英語のみ:       {stats.get('pure_english', 0):,}")
    print(f"    アラビア数字を含む (Contains Digit):{stats.get('contains_digit', 0):,}")
    print(
        f"    特殊文字パターンのみ:           {stats.get('special_char_pattern', 0):,}"
    )
    print(f"    未分類 (Uncategorized):         {stats.get('uncategorized', 0):,}")
    print("=" * 60 + "\n")


def print_example_tokens(
    model_id: str, category_name: str, token_ids: List[int], max_tokens: int = 10
):
    """指定されたカテゴリのトークンIDから、実際のトークン文字列の例を表示します。"""
    if not token_ids:
        print(f"\n--- トークン例: {category_name} ---")
        print("  (このカテゴリに属するトークンはありません)")
        print("-" * (len(category_name) + 24))
        return

    print(f"\n--- トークン例: {category_name} (最大 {max_tokens} 件表示) ---")
    try:
        # この関数内で再度トークナイザーをロードするのは非効率だが、簡潔さのため
        # 大量に呼び出す場合は、外部でロードしたtokenizerを渡す方が良い
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        ids_to_show = token_ids[:max_tokens]
        for token_id in ids_to_show:
            try:
                # clean_up_tokenization_spaces=False で SentencePiece の '_' なども表示
                token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                # repr() を使って特殊文字や空白を分かりやすく表示
                print(f"  ID: {token_id:<6d} | Token: {repr(token)}")
            except Exception as e:
                print(f"  ID: {token_id:<6d} | デコードエラー: {e}")
    except Exception as e:
        logging.error(f"トークン例表示のためのトークナイザー読み込み中にエラー: {e}")
    finally:
        print("-" * (len(category_name) + 24))  # フッターライン


# --- メイン実行ロジック ---


def run_full_analysis(model_id: str, min_token_id: int, output_dir: str):
    """
    トークン分析の全プロセス（分析、結果保存、サマリー表示、例表示）を実行します。
    """
    logging.info(f"=== トークン分析開始 ===")
    logging.info(f"モデルID: {model_id}")
    logging.info(f"最小トークンID: {min_token_id}")
    logging.info(f"出力ディレクトリ: {output_dir}")

    # 1. メインの分析を実行
    analysis_result = analyze_token_categories(model_id, min_token_id)

    if analysis_result:
        # 2. 詳細なJSON結果を保存
        # base_filenameはデフォルトの"token_analysis_jp"を使用
        save_analysis_results(analysis_result, output_dir)

        # 3. 主要カテゴリのトークンIDリストをテキストファイルに保存
        #    logit bias調整などに利用可能
        token_ids = analysis_result.get("token_ids", {})
        categories_to_save = [
            "contains_japanese",
            "pure_japanese_script",
            "contains_halfwidth_katakana",
            "contains_fullwidth_ascii",
            "pure_english",
            "special_char_pattern",
            "uncategorized",
        ]
        for category_name in categories_to_save:
            save_token_list(
                token_ids.get(category_name, []), category_name, output_dir, model_id
            )

        # 4. 分析サマリーをコンソールに表示
        print_analysis_summary(analysis_result)

        # 5. いくつかのカテゴリについてトークン例を表示
        #    特に興味深いカテゴリや未分類を中心に表示
        categories_to_show_examples = [
            ("Uncategorized", 20),
            ("Special Char Pattern", 20),
            ("Pure Japanese Script", 15),
            ("Contains Halfwidth Katakana", 15),
            ("Contains Fullwidth ASCII", 15),
            ("Contains Japanese", 10),  # 参考用
            ("Pure English", 10),  # 参考用
        ]
        for category_name, max_num in categories_to_show_examples:
            # カテゴ리 키 이름은 Python 변수명 스타일에 맞게 변환해야 할 수도 있음
            # 여기서는 save_token_list 와 동일한 이름을 사용한다고 가정
            py_category_name = category_name.lower().replace(" ", "_").replace("/", "_")
            print_example_tokens(
                model_id,
                category_name,
                token_ids.get(py_category_name, []),
                max_tokens=max_num,
            )

        logging.info(f"=== トークン分析完了 ===")

    else:
        logging.error("分析を完了できませんでした。ログを確認してください。")


def main():
    """コマンドライン引数を処理し、分析を実行するエントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="日本語トークナイザーアナライザー - モデルの語彙を分析し、トークンをカテゴリ分類します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # デフォルト値をヘルプに表示
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="分析対象モデルのHugging Face IDまたはローカルパス。",
    )
    parser.add_argument(
        "--min_token_id",
        type=int,
        default=102,  # 一般的なモデルで特殊トークン以降のIDを開始点とする例
        help="分析を開始する最小トークンID。これより小さいIDは特殊トークンでなくても無視されます。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="token_analysis_output",  # デフォルトの出力ディレクトリ名
        help="分析結果 (JSONおよびTXTリスト) を保存するディレクトリ。",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ロギングレベルを設定します。",
    )

    args = parser.parse_args()

    # コマンドライン引数に基づいてロギングレベルを再設定
    logging.getLogger().setLevel(args.log_level.upper())

    # 分析実行
    run_full_analysis(args.model_id, args.min_token_id, args.output_dir)


if __name__ == "__main__":
    main()

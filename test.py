#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import logging
import transformers
import os
from token_analyzer_jp import (
    # 利用するヘルパー関数
    is_japanese_related_char,
    is_pure_japanese_script_char,
    is_special_char_pattern,
    # メイン分析関数
    analyze_token_categories,
    # 定義された文字セット（テストケースで参照するため）
    HIRAGANA,
    KATAKANA,
    KATAKANA_HW,
    KANJI_COMMON,
    KANJI_EXT_A,
    JP_PUNCT,
    JP_SYMBOLS_ETC,
    JP_FULLWIDTH_ASCII_PRINTABLE,
    ENGLISH_BASIC,
)

# テスト実行中のINFOレベル以上のログは抑制 (必要に応じて変更)
# 詳細なログを見たい場合は、以下の行をコメントアウトし、必要に応じてレベル調整
logging.disable(logging.CRITICAL)
# logging.basicConfig(level=logging.DEBUG) # デバッグ時に有効化

# --- 定数 ---
# 実際のモデルID（モックしない）
TARGET_MODEL_ID = "unsloth/Llama-4-Scout-17B-16E-Instruct"
# テスト対象とするトークンIDの開始点 (特殊トークンを避ける)
MIN_TEST_TOKEN_ID = 102
# テストで検証する特定トークンIDの例 (モデル依存)
# (ID, デコード文字列(参考), 期待されるカテゴリ(リスト), 期待されないカテゴリ(リスト))
EXPECTED_TOKEN_CATEGORIES = [
    (
        30162,
        " 日本",
        ["contains_japanese", "contains_kanji"],
        ["pure_japanese_script", "pure_english"],
    ),
    (31185, "語", ["contains_japanese", "contains_kanji", "pure_japanese_script"], []),
    (
        30088,
        "です",
        ["contains_japanese", "contains_hiragana", "pure_japanese_script"],
        [],
    ),
    (
        30472,
        "トークン",
        ["contains_japanese", "contains_katakana_full", "pure_japanese_script"],
        [],
    ),
    (
        105743,
        "ｶﾞ",
        ["contains_japanese", "contains_katakana_half", "pure_japanese_script"],
        [],
    ),
    (
        30004,
        "、",
        ["contains_japanese", "contains_jp_punct_symbol"],
        ["pure_japanese_script"],
    ),
    (
        99796,
        "ＡＢＣ",
        ["contains_japanese", "contains_fullwidth_ascii"],
        ["pure_japanese_script"],
    ),
    (319, " a", ["contains_basic_english"], ["pure_english", "contains_japanese"]),
    (450, " Apple", ["contains_basic_english"], ["pure_english", "contains_japanese"]),
    (13, " ", [], ["contains_japanese", "pure_english", "special_char_pattern"]),
    (29900, " 123", ["contains_digit"], ["contains_japanese", "pure_english"]),
    (30587, " Code", ["contains_basic_english"], []),
    (
        32100,
        "株式会社",
        ["contains_japanese", "contains_kanji", "pure_japanese_script"],
        [],
    ),
    (
        106324,
        "ChatGPT",
        ["contains_basic_english", "pure_english"],
        ["contains_japanese"],
    ),
    (
        125933,
        "・",
        ["contains_japanese", "contains_jp_punct_symbol"],
        ["pure_japanese_script"],
    ),
    (
        30008,
        "「",
        ["contains_japanese", "contains_jp_punct_symbol"],
        ["pure_japanese_script"],
    ),
    (100, "<0x00>", [], []),
    (2, "</s>", [], []),
    (29871, "\n", [], ["contains_japanese", "pure_english"]),
    (120128, " #", [], ["contains_japanese", "special_char_pattern"]),
    (127991, " 🔥", [], ["contains_japanese", "special_char_pattern"]),
    (
        12756,
        "ovo",
        ["contains_basic_english", "pure_english"],
        ["contains_japanese", "special_char_pattern"],
    ),  # 수정됨: 'ovo'는 pure_english
]


# ----- ヘルパー関数のテストクラス -----
class HelperFunctionTests(unittest.TestCase):
    def test_is_japanese_related_charの拡張ケース(self):
        # 改善された is_japanese_related_char 関数をテスト
        test_cases = [
            ("あ", True),
            ("ア", True),
            ("漢", True),
            ("ー", True),
            ("ｶ", True),
            ("ﾟ", True),
            ("。", True),
            ("　", True),
            ("・", True),
            ("￥", True),
            ("「", True),
            ("､", True),
            ("Ａ", True),
            ("ｂ", True),
            ("０", True),
            ("！", True),
            ("～", True),
            ("A", False),
            ("1", False),
            ("$", False),
            (" ", False),
            ("\n", False),
            ("α", False),
        ]
        for char, expected in test_cases:
            with self.subTest(char=char, expected=expected):
                actual = is_japanese_related_char(char)
                self.assertEqual(
                    actual,
                    expected,
                    f"文字 '{char}' (U+{ord(char):04X}) の is_japanese_related_char 結果が期待値と異なります",
                )

    def test_is_pure_japanese_script_charの拡張ケース(self):
        # 改善された is_pure_japanese_script_char 関数をテスト (期待値修正済み)
        test_cases = [
            ("あ", True),
            ("ア", True),
            ("漢", True),
            ("ｶ", True),
            ("ー", True),
            ("･", True),
            ("﨑", False),
            ("。", False),
            ("　", False),
            ("Ａ", False),
            ("1", False),
            (" ", False),
            ("\n", False),
        ]
        for char, expected in test_cases:
            with self.subTest(char=char, expected=expected):
                actual = is_pure_japanese_script_char(char)
                self.assertEqual(
                    actual,
                    expected,
                    f"文字 '{char}' (U+{ord(char):04X}) の is_pure_japanese_script_char 結果が期待値と異なります",
                )

    def test_is_special_char_patternの拡張ケース(self):
        # 改善された is_special_char_pattern 関数をテスト (期待値修正済み)
        test_cases = [
            ("!!!", True),
            ("@#$", True),
            ("&&&", True),
            ("+-*/", True),
            ("---", True),
            ("===", True),
            ("abc", False),
            ("あいう", False),
            ("123", False),
            ("ＡＢＣ", False),
            ("カタカナ", False),
            ("半角ｶﾅ", False),
            ("漢字", False),
            (" ", False),
            ("　", False),
            ("a#$", False),
            ("#あ$", False),
            ("#1$", False),
            ("#Ａ$", False),
            ("", False),
            ("「」", False),
            (" ---", False),
            (" #", False),
            (" 🔥", False),  # スペースが含まれると False
        ]
        for token, expected in test_cases:
            with self.subTest(token=repr(token), expected=expected):
                actual = is_special_char_pattern(token)
                self.assertEqual(
                    actual,
                    expected,
                    f"トークン {repr(token)} の is_special_char_pattern 結果が期待値と異なります",
                )


# ----- メイン分析関数のテストクラス -----
# 実際のモデルをロードして分析結果を検証
class AnalysisResultTests(unittest.TestCase):
    tokenizer = None  # クラス変数として tokenizer を保持
    result = None
    stats = {}
    token_ids_by_category = {}
    details = {}

    @classmethod
    def setUpClass(cls):
        # このクラスの全テスト実行前に一度だけ実行
        print(f"\n--- {cls.__name__} セットアップ開始 ---")
        print(f"テスト対象モデル: {TARGET_MODEL_ID}")
        try:
            # 実際のトークナイザーをロード (モック不使用)
            cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
                TARGET_MODEL_ID, trust_remote_code=True
            )
            print(f"トークナイザー ({cls.tokenizer.__class__.__name__}) のロード完了")

            # 実際の分析を実行 (時間がかかる可能性あり)
            print(f"トークン分析を開始します (min_token_id={MIN_TEST_TOKEN_ID})...")
            cls.result = analyze_token_categories(
                TARGET_MODEL_ID, min_token_id=MIN_TEST_TOKEN_ID
            )
            print("トークン分析完了")

            # 結果の取得と基本的なチェック
            if cls.result is None:
                raise RuntimeError("analyze_token_categories が None を返しました")
            cls.stats = cls.result.get("statistics", {})
            cls.token_ids_by_category = cls.result.get("token_ids", {})
            cls.details = cls.result.get("analysis_details", {})
            print(f"分析対象トークン数: {cls.details.get('num_tokens_analyzed', 0):,}")

        except Exception as e:
            # エラー発生時の処理
            print(f"\n****** セットアップ中に致命的なエラーが発生しました ******")
            print(f"エラータイプ: {type(e).__name__}")
            print(f"エラーメッセージ: {e}")
            import traceback

            print(traceback.format_exc())  # 詳細なトレースバックを出力
            print("***************************************************\n")
            cls.result = None  # エラーフラグ
        finally:
            print(f"--- {cls.__name__} セットアップ完了 ---")

    def test_分析結果の基本構造と必須キーの存在確認(self):
        # セットアップが成功したか確認
        if self.result is None:
            self.fail("セットアップ失敗のためテスト強制終了")

        # 主要なキーの存在確認
        self.assertEqual(self.result["model_id"], TARGET_MODEL_ID)
        self.assertIsInstance(self.result["vocab_size"], int)
        self.assertGreaterEqual(self.result["vocab_size"], 0)
        self.assertIsInstance(self.result["num_special_tokens"], int)
        self.assertGreaterEqual(self.result["num_special_tokens"], 0)

        expected_top_keys = [
            "model_id",
            "vocab_size",
            "num_special_tokens",
            "analysis_details",
            "statistics",
            "token_ids",
        ]
        for key in expected_top_keys:
            self.assertIn(
                key, self.result, f"必須キー '{key}' が分析結果に含まれていません"
            )

        # analysis_details のキー確認
        self.assertIsInstance(self.details, dict)
        expected_detail_keys = [
            "min_token_id_analyzed",
            "max_token_id_analyzed",
            "num_tokens_analyzed",
            "num_errors",
            "excluded_special_ids",
        ]
        for key in expected_detail_keys:
            self.assertIn(
                key,
                self.details,
                f"必須キー 'analysis_details.{key}' が含まれていません",
            )
        self.assertIsInstance(self.details["excluded_special_ids"], list)

        # statistics と token_ids のキー確認
        self.assertIsInstance(self.stats, dict)
        self.assertIsInstance(self.token_ids_by_category, dict)
        expected_category_keys = {
            "contains_japanese",
            "pure_japanese_script",
            "pure_english",
            "contains_hiragana",
            "contains_katakana_full",
            "contains_katakana_half",
            "contains_kanji",
            "contains_jp_punct_symbol",
            "contains_fullwidth_ascii",
            "contains_basic_english",
            "contains_digit",
            "special_char_pattern",
            "uncategorized",
        }
        self.assertEqual(
            set(self.stats.keys()),
            expected_category_keys,
            "statistics のキーが期待されるカテゴリと一致しません",
        )
        self.assertEqual(
            set(self.token_ids_by_category.keys()),
            expected_category_keys,
            "token_ids のキーが期待されるカテゴリと一致しません",
        )

        self.assertGreaterEqual(self.details["num_tokens_analyzed"], 0)
        if self.details["num_tokens_analyzed"] == 0:
            logging.warning(
                "分析対象トークン数が0でした。min_token_idやモデルを確認してください。"
            )

    def test_カテゴリ間の部分集合関係の検証(self):
        # セットアップが成功したか確認
        if self.result is None:
            self.fail("セットアップ失敗のためテスト強制終了")

        def check_subset(subset_name, superset_name):
            subset_set = set(self.token_ids_by_category.get(subset_name, []))
            superset_set = set(self.token_ids_by_category.get(superset_name, []))
            if subset_set and superset_set:
                self.assertTrue(
                    subset_set.issubset(superset_set),
                    f"'{subset_name}' は '{superset_name}' の部分集合であるべきです",
                )

        check_subset("pure_japanese_script", "contains_japanese")
        check_subset("contains_hiragana", "contains_japanese")
        check_subset("contains_katakana_full", "contains_japanese")
        check_subset("contains_katakana_half", "contains_japanese")
        check_subset("contains_kanji", "contains_japanese")
        check_subset("contains_jp_punct_symbol", "contains_japanese")
        check_subset("contains_fullwidth_ascii", "contains_japanese")
        check_subset("pure_english", "contains_basic_english")

        pure_jp_set = set(self.token_ids_by_category.get("pure_japanese_script", []))
        jp_ps_set = set(self.token_ids_by_category.get("contains_jp_punct_symbol", []))
        fw_ascii_set = set(
            self.token_ids_by_category.get("contains_fullwidth_ascii", [])
        )

        if pure_jp_set and jp_ps_set:
            self.assertTrue(
                pure_jp_set.isdisjoint(jp_ps_set - pure_jp_set),
                "pure_japanese_script と (日本語句読点記号のみを含むトークン) は排他のはずです",
            )
        if pure_jp_set and fw_ascii_set:
            self.assertTrue(
                pure_jp_set.isdisjoint(fw_ascii_set - pure_jp_set),
                "pure_japanese_script と (全角ASCIIのみを含むトークン) は排他のはずです",
            )

    def test_カテゴリ間の排他関係の検証(self):
        # セットアップが成功したか確認
        if self.result is None:
            self.fail("セットアップ失敗のためテスト強制終了")

        # 排他関係をチェックする関数 (失敗メッセージ改善版)
        def check_disjoint(cat1_name, cat2_name):
            set1 = set(self.token_ids_by_category.get(cat1_name, []))
            set2 = set(self.token_ids_by_category.get(cat2_name, []))
            if set1 and set2:  # 両方に要素がある場合のみチェック
                intersection = set1.intersection(set2)
                if intersection:  # 共通要素が存在する場合のみ失敗させる
                    sample_size = 5
                    intersection_list = sorted(list(intersection))
                    sample_elements = intersection_list[:sample_size]
                    num_common = len(intersection_list)
                    error_msg = (
                        f"カテゴリ '{cat1_name}' と '{cat2_name}' は排他であるべきですが、"
                        f"{num_common} 個の共通要素が見つかりました。\n"
                        f"  共通要素サンプル (最大 {sample_size} 件): {sample_elements}"
                    )
                    # デコード例を追加 (Tokenizerがロード済みの場合)
                    if self.tokenizer:
                        try:
                            decoded_samples = [
                                f"{tid}:{repr(self.tokenizer.decode([tid], clean_up_tokenization_spaces=False))}"
                                for tid in sample_elements
                            ]
                            error_msg += f"\n  デコード例: {decoded_samples}"
                        except Exception as e:
                            error_msg += f"\n  (デコード中にエラー: {e})"
                    self.fail(error_msg)  # カスタムメッセージでテストを失敗させる

        # --- 主要な排他関係のチェック ---
        check_disjoint("pure_japanese_script", "pure_english")
        check_disjoint("pure_japanese_script", "special_char_pattern")
        check_disjoint("pure_japanese_script", "uncategorized")

        check_disjoint("pure_english", "contains_japanese")
        check_disjoint("pure_english", "special_char_pattern")
        check_disjoint("pure_english", "uncategorized")

        check_disjoint("special_char_pattern", "contains_japanese")
        check_disjoint("special_char_pattern", "contains_basic_english")
        check_disjoint("special_char_pattern", "uncategorized")

        # 未分類は他の「明確な」カテゴリとは排他
        uncat_set = set(self.token_ids_by_category.get("uncategorized", []))
        if uncat_set:
            defined_categories = [
                "contains_japanese",
                "pure_japanese_script",
                "pure_english",
                "contains_hiragana",
                "contains_katakana_full",
                "contains_katakana_half",
                "contains_kanji",
                "contains_jp_punct_symbol",
                "contains_fullwidth_ascii",
                "contains_basic_english",
                "contains_digit",
                "special_char_pattern",
            ]
            for cat_name in defined_categories:
                # check_disjointは内部で空集合チェックするのでそのまま呼び出す
                check_disjoint("uncategorized", cat_name)

    def test_特定トークンIDのカテゴリ所属検証(self):
        # セットアップが成功したか確認
        if self.result is None:
            self.fail("セットアップ失敗のためテスト強制終了")
        if self.tokenizer is None:
            self.fail("Tokenizerがロードされていません")  # Tokenizerも確認

        # 分析対象ID範囲と特殊IDを取得
        min_id = self.details.get("min_token_id_analyzed", -1)  # 安全なデフォルト値
        max_id = self.details.get("max_token_id_analyzed", -1)
        special_ids_set = set(self.details.get("excluded_special_ids", []))

        # 事前定義リストを使って検証
        for (
            token_id,
            token_repr,
            expected_cats,
            not_expected_cats,
        ) in EXPECTED_TOKEN_CATEGORIES:
            # 分析対象外・特殊トークンはスキップ
            if not (min_id <= token_id <= max_id) or token_id in special_ids_set:
                continue

            # デコード (エラー発生時は記録)
            try:
                actual_decoded = self.tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False
                )
            except Exception as e:
                actual_decoded = f"DECODE ERROR: {e}"

            # subTestで個別のテストケースとして実行 (不要なprint削除済み)
            with self.subTest(
                token_id=token_id,
                token_repr=repr(token_repr),
                actual_decoded=repr(actual_decoded),
            ):
                # 期待カテゴリへの所属確認
                for cat_name in expected_cats:
                    self.assertIn(
                        cat_name,
                        self.token_ids_by_category,
                        f"テストエラー: カテゴリ '{cat_name}' が結果セットのキーに存在しません",
                    )
                    cat_set = set(self.token_ids_by_category.get(cat_name, []))
                    self.assertIn(
                        token_id,
                        cat_set,
                        f"ID {token_id} ({repr(actual_decoded)}) はカテゴリ '{cat_name}' に【含まれるべき】です",
                    )
                # 非期待カテゴリへの非所属確認
                for cat_name in not_expected_cats:
                    if cat_name in self.token_ids_by_category:
                        cat_set = set(self.token_ids_by_category[cat_name])
                        self.assertNotIn(
                            token_id,
                            cat_set,
                            f"ID {token_id} ({repr(actual_decoded)}) はカテゴリ '{cat_name}' に【含まれるべきではありません】",
                        )

    def test_統計値の整合性検証(self):
        # セットアップが成功したか確認
        if self.result is None:
            self.fail("セットアップ失敗のためテスト強制終了")

        # 各カテゴリの統計数と実際のIDリスト長が一致するか
        for name, count in self.stats.items():
            self.assertEqual(
                count,
                len(self.token_ids_by_category.get(name, [])),
                f"カテゴリ '{name}' の統計数({count})とIDリスト長({len(self.token_ids_by_category.get(name, []))})が一致しません",
            )

        # 分析されたトークン数 = ユニークに分類されたトークン数 + 未分類トークン数
        all_categorized_ids_union = set()
        for name, ids in self.token_ids_by_category.items():
            if name != "uncategorized":
                all_categorized_ids_union.update(ids)

        num_analyzed = self.details.get("num_tokens_analyzed", 0)
        num_uncategorized = self.stats.get("uncategorized", 0)
        num_categorized_unique = len(all_categorized_ids_union)

        # 0件の場合も含めて等式が成り立つか検証
        self.assertEqual(
            num_analyzed,
            num_categorized_unique + num_uncategorized,
            f"分析トークン数({num_analyzed:,})が、分類済みユニーク数({num_categorized_unique:,}) + 未分類数({num_uncategorized:,}) = {num_categorized_unique + num_uncategorized:,} と一致しません",
        )


# --- テスト実行 ---
if __name__ == "__main__":
    # verbosity=2 で各テストメソッド名も表示
    unittest.main(verbosity=2)

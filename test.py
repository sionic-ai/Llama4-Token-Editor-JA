#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import logging
import transformers
import os
import random  # サンプル抽出に使用
from token_analyzer_jp import (
    # 利用するヘルパー関数
    is_japanese_related_char,
    is_pure_japanese_script_char,
    is_special_char_pattern,
    # メイン分析関数
    analyze_token_categories,
    # 定義された文字セット（分類ロジック再現のため）
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
# サンプル検証で抽出するIDの最大数
MAX_SAMPLES_PER_CATEGORY = 5  # 各カテゴリから最大5件
TOTAL_SAMPLES_FOR_LOGIC_TEST = 50  # 全体で最大50件程度


# ----- ヘルパー関数のテストクラス -----
class HelperFunctionTests(unittest.TestCase):
    def test_is_japanese_related_charの拡張ケース(self):
        # 改善された is_japanese_related_char 関数をテスト
        test_cases = [
            # 基本
            ("あ", True),
            ("ア", True),
            ("漢", True),
            ("ー", True),
            # 半角カタカナ
            ("ｶ", True),
            ("ﾟ", True),
            # 句読点・記号
            ("。", True),
            ("　", True),
            ("・", True),
            ("￥", True),
            ("「", True),
            ("､", True),
            # 全角ASCII
            ("Ａ", True),
            ("ｂ", True),
            ("０", True),
            ("！", True),
            ("～", True),
            # 非該当
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
            # 該当
            ("あ", True),
            ("ア", True),
            ("漢", True),
            ("ｶ", True),
            ("ー", True),
            ("･", True),
            # 非該当
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
            # 該当 (英数、空白、定義済み言語文字以外のみ)
            ("!!!", True),
            ("@#$", True),
            ("&&&", True),
            ("+-*/", True),
            ("---", True),
            ("===", True),
            # 非該当 (何かが混ざっている)
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

    # --- 分析結果の基本構造・統計値・集合関係のテスト (変更なし) ---
    def test_分析結果の基本構造と必須キーの存在確認(self):
        if self.result is None:
            self.fail("セットアップ失敗のためテスト強制終了")
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
            self.assertIn(key, self.result)
        self.assertIsInstance(self.details, dict)
        expected_detail_keys = [
            "min_token_id_analyzed",
            "max_token_id_analyzed",
            "num_tokens_analyzed",
            "num_errors",
            "excluded_special_ids",
        ]
        for key in expected_detail_keys:
            self.assertIn(key, self.details)
        self.assertIsInstance(self.details["excluded_special_ids"], list)
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
        self.assertEqual(set(self.stats.keys()), expected_category_keys)
        self.assertEqual(set(self.token_ids_by_category.keys()), expected_category_keys)
        self.assertGreaterEqual(self.details["num_tokens_analyzed"], 0)
        if self.details["num_tokens_analyzed"] == 0:
            logging.warning("分析対象トークン数が0でした。")

    def test_カテゴリ間の部分集合関係の検証(self):
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
        if self.result is None:
            self.fail("セットアップ失敗のためテスト強制終了")

        def check_disjoint(cat1_name, cat2_name):
            set1 = set(self.token_ids_by_category.get(cat1_name, []))
            set2 = set(self.token_ids_by_category.get(cat2_name, []))
            if set1 and set2:
                intersection = set1.intersection(set2)
                if intersection:
                    sample_size = 5
                    intersection_list = sorted(list(intersection))
                    sample_elements = intersection_list[:sample_size]
                    num_common = len(intersection_list)
                    error_msg = (
                        f"カテゴリ '{cat1_name}' と '{cat2_name}' は排他であるべきですが、{num_common} 個の共通要素。\n"
                        f"  サンプル: {sample_elements}"
                    )
                    if self.tokenizer:
                        try:
                            decoded_samples = [
                                f"{tid}:{repr(self.tokenizer.decode([tid], clean_up_tokenization_spaces=False))}"
                                for tid in sample_elements
                            ]
                            error_msg += f"\n  デコード例: {decoded_samples}"
                        except Exception as e:
                            error_msg += f"\n  (デコードエラー: {e})"
                    self.fail(error_msg)

        check_disjoint("pure_japanese_script", "pure_english")
        check_disjoint("pure_japanese_script", "special_char_pattern")
        check_disjoint("pure_japanese_script", "uncategorized")
        check_disjoint("pure_english", "contains_japanese")
        check_disjoint("pure_english", "special_char_pattern")
        check_disjoint("pure_english", "uncategorized")
        check_disjoint("special_char_pattern", "contains_japanese")
        check_disjoint("special_char_pattern", "contains_basic_english")
        check_disjoint("special_char_pattern", "uncategorized")
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
                check_disjoint("uncategorized", cat_name)

    def test_統計値の整合性検証(self):
        if self.result is None:
            self.fail("セットアップ失敗のためテスト強制終了")
        for name, count in self.stats.items():
            self.assertEqual(
                count,
                len(self.token_ids_by_category.get(name, [])),
                f"カテゴリ '{name}' の統計数({count})とIDリスト長({len(self.token_ids_by_category.get(name, []))})が一致しません",
            )
        all_categorized_ids_union = set()
        for name, ids in self.token_ids_by_category.items():
            if name != "uncategorized":
                all_categorized_ids_union.update(ids)
        num_analyzed = self.details.get("num_tokens_analyzed", 0)
        num_uncategorized = self.stats.get("uncategorized", 0)
        num_categorized_unique = len(all_categorized_ids_union)
        self.assertEqual(
            num_analyzed,
            num_categorized_unique + num_uncategorized,
            f"分析トークン数({num_analyzed:,})が、分類済みユニーク数({num_categorized_unique:,}) + 未分類数({num_uncategorized:,}) = {num_categorized_unique + num_uncategorized:,} と一致しません",
        )

    # --- 分類ロジックの一貫性検証テスト ---
    def test_分類ロジックの一貫性検証_サンプル(self):
        """
        実際の分析結果からトークンをサンプリングし、各トークンについて
        分類ロジックを再実行した結果と、実際の分析結果が一致するか検証する。
        """
        if self.result is None:
            self.fail("セットアップ失敗のためテスト強制終了")
        if self.tokenizer is None:
            self.fail("Tokenizerがロードされていません")
        if self.details["num_tokens_analyzed"] == 0:
            self.skipTest("分析対象トークンがないためスキップ")

        sampled_token_ids = set()
        categories_to_sample = [
            "pure_japanese_script",
            "pure_english",
            "special_char_pattern",
            "uncategorized",
            "contains_katakana_half",
            "contains_fullwidth_ascii",
            "contains_jp_punct_symbol",
            "contains_digit",
        ]
        for cat_name in categories_to_sample:
            ids_in_category = self.token_ids_by_category.get(cat_name, [])
            if ids_in_category:
                k = min(len(ids_in_category), MAX_SAMPLES_PER_CATEGORY)
                sampled_token_ids.update(random.sample(ids_in_category, k))

        final_sample_ids = sorted(list(sampled_token_ids))
        if len(final_sample_ids) > TOTAL_SAMPLES_FOR_LOGIC_TEST:
            final_sample_ids = random.sample(
                final_sample_ids, TOTAL_SAMPLES_FOR_LOGIC_TEST
            )
            final_sample_ids.sort()

        if not final_sample_ids:
            self.skipTest("検証対象のサンプルトークンIDが見つかりませんでした。")

        print(
            f"\n--- 分類ロジック一貫性検証 (サンプルID数: {len(final_sample_ids)}) ---"
        )

        for token_id in final_sample_ids:
            try:
                decoded_token = self.tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False
                )
            except Exception as e:
                self.fail(f"ID {token_id} のデコード中にエラー: {e}")
                continue

            with self.subTest(token_id=token_id, decoded_token=repr(decoded_token)):
                expected_cats_for_this_token = self._calculate_expected_categories(
                    decoded_token
                )
                actual_cats_for_this_token = set()
                for cat_name, ids_list in self.token_ids_by_category.items():
                    # IDリストが大きくなる可能性を考慮し、set変換は一度だけ行うか、
                    # もしくは analyze_token_categories 側で結果をsetで返すように変更検討
                    if token_id in set(ids_list):
                        actual_cats_for_this_token.add(cat_name)

                self.assertSetEqual(
                    actual_cats_for_this_token,
                    expected_cats_for_this_token,
                    f"ID {token_id} ({repr(decoded_token)}) の分類結果がロジックと一致しません",
                )

    # --- 新しいテストメソッド: Partial日本語トークンの検証 ---
    def test_partial_japanese_token_samples(self):
        """
        `contains_japanese` に含まれ、かつ `pure_japanese_script` には含まれない
        トークン（部分トークンや混合トークンの候補）が実際に `contains_japanese`
        に正しく分類されているかをサンプル検証する。
        """
        if self.result is None:
            self.fail("セットアップ失敗のためテスト強制終了")
        if self.tokenizer is None:
            self.fail("Tokenizerがロードされていません")

        contains_jp_set = set(self.token_ids_by_category.get("contains_japanese", []))
        pure_jp_set = set(self.token_ids_by_category.get("pure_japanese_script", []))

        partial_or_mixed_jp_ids = list(contains_jp_set - pure_jp_set)

        if not partial_or_mixed_jp_ids:
            self.skipTest("Partial/Mixed日本語トークンの候補が見つかりませんでした。")

        num_samples = min(len(partial_or_mixed_jp_ids), TOTAL_SAMPLES_FOR_LOGIC_TEST)
        sampled_ids = random.sample(partial_or_mixed_jp_ids, num_samples)
        sampled_ids.sort()

        print(
            f"\n--- Partial/Mixed日本語トークン検証 (サンプルID数: {len(sampled_ids)}) ---"
        )

        for token_id in sampled_ids:
            try:
                decoded_token = self.tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False
                )
            except Exception as e:
                logging.warning(
                    f"ID {token_id} のデコード中にエラー (Partial/Mixed検証): {e}"
                )
                continue

            with self.subTest(token_id=token_id, decoded_token=repr(decoded_token)):
                self.assertIn(
                    token_id,
                    contains_jp_set,
                    f"ID {token_id} ({repr(decoded_token)}) は partial/mixed 候補であり、'contains_japanese' に含まれるべきです",
                )
                self.assertNotIn(
                    token_id,
                    pure_jp_set,
                    f"ID {token_id} ({repr(decoded_token)}) は partial/mixed 候補であり、'pure_japanese_script' に含まれるべきではありません",
                )
                has_jp_related_flag = False
                for char in decoded_token:
                    if is_japanese_related_char(char):
                        has_jp_related_flag = True
                        break
                self.assertTrue(
                    has_jp_related_flag,
                    f"ID {token_id} ({repr(decoded_token)}) はデコード結果に日本語関連文字を含むはずです (contains_japanese判定)",
                )

    # --- ヘルパーメソッド (_calculate_expected_categories, _calculate_token_flags) ---
    def _calculate_expected_categories(self, decoded_token):
        """
        与えられたデコード済みトークン文字列に対して、
        analyze_token_categories と同じ分類ロジックを適用し、
        属すると期待されるカテゴリ名のセットを返す。
        """
        if not decoded_token:
            return set()
        flags = self._calculate_token_flags(decoded_token)
        expected_cats = set()
        if flags["is_related_to_jp"]:
            expected_cats.add("contains_japanese")
        if flags["has_hiragana"]:
            expected_cats.add("contains_hiragana")
        if flags["has_katakana_full"]:
            expected_cats.add("contains_katakana_full")
        if flags["has_katakana_half"]:
            expected_cats.add("contains_katakana_half")
        if flags["has_kanji"]:
            expected_cats.add("contains_kanji")
        if flags["has_jp_punct_symbol"]:
            expected_cats.add("contains_jp_punct_symbol")
        if flags["has_fullwidth_ascii"]:
            expected_cats.add("contains_fullwidth_ascii")
        if flags["all_pure_jp_script"] and (
            flags["has_hiragana"]
            or flags["has_katakana_full"]
            or flags["has_katakana_half"]
            or flags["has_kanji"]
        ):
            expected_cats.add("pure_japanese_script")
        if flags["has_basic_english"]:
            expected_cats.add("contains_basic_english")
        if flags["all_basic_english"] and not flags["is_related_to_jp"]:
            expected_cats.add("pure_english")
        if flags["has_digit"]:
            expected_cats.add("contains_digit")
        is_sp_pattern = is_special_char_pattern(decoded_token)
        if (
            not flags["is_related_to_jp"]
            and not flags["has_basic_english"]
            and not flags["has_digit"]
            and is_sp_pattern
        ):
            expected_cats.add("special_char_pattern")
        # is_related_to_jp, has_basic_english, has_digit, is_sp_pattern のいずれもFalseの場合にuncategorized
        if (
            not flags["is_related_to_jp"]
            and not flags["has_basic_english"]
            and not flags["has_digit"]
            and not is_sp_pattern
        ):
            expected_cats.add("uncategorized")
        return expected_cats

    def _calculate_token_flags(self, decoded_token):
        """
        analyze_token_categories 内の文字チェックロジックを模倣し、
        トークンに関するフラグ辞書を返す。
        """
        flags = {
            "has_hiragana": False,
            "has_katakana_full": False,
            "has_katakana_half": False,
            "has_kanji": False,
            "has_jp_punct_symbol": False,
            "has_fullwidth_ascii": False,
            "has_basic_english": False,
            "has_digit": False,
            "has_other_char": False,
            "all_pure_jp_script": True,
            "all_basic_english": True,
            "is_related_to_jp": False,
        }
        if not decoded_token:
            return flags
        for char in decoded_token:
            is_hira = char in HIRAGANA
            is_kata_f = char in KATAKANA
            is_kata_h = char in KATAKANA_HW
            is_kanji = char in KANJI_COMMON or char in KANJI_EXT_A
            is_jp_ps = char in JP_PUNCT or char in JP_SYMBOLS_ETC
            is_fw_ascii = char in JP_FULLWIDTH_ASCII_PRINTABLE
            is_basic_eng = char in ENGLISH_BASIC
            is_digit = "0" <= char <= "9"
            is_space = char.isspace()
            if is_hira:
                flags["has_hiragana"] = True
            if is_kata_f:
                flags["has_katakana_full"] = True
            if is_kata_h:
                flags["has_katakana_half"] = True
            if is_kanji:
                flags["has_kanji"] = True
            if is_jp_ps:
                flags["has_jp_punct_symbol"] = True
            if is_fw_ascii:
                flags["has_fullwidth_ascii"] = True
            if is_basic_eng:
                flags["has_basic_english"] = True
            if is_digit:
                flags["has_digit"] = True
            is_pure_jp = is_hira or is_kata_f or is_kata_h or is_kanji
            if not is_pure_jp:
                flags["all_pure_jp_script"] = False
            if not is_basic_eng:
                flags["all_basic_english"] = False
            is_jp_related_char_flag = is_pure_jp or is_jp_ps or is_fw_ascii
            if is_jp_related_char_flag:
                flags["is_related_to_jp"] = True
            if not (
                is_jp_related_char_flag
                or is_basic_eng
                or is_digit
                or is_space
                or char.isalnum()
            ):
                # 特殊文字パターンに属さないその他の文字があるか
                if not is_special_char_pattern(char):
                    flags["has_other_char"] = True
        return flags


# --- テスト実行 ---
if __name__ == "__main__":
    # verbosity=2 で各テストメソッド名も表示
    unittest.main(verbosity=2)

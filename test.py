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

# テスト実行中のログレベル設定 (必要に応じて変更)
# logging.disable(logging.CRITICAL) # ログを抑制する場合
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)  # INFOレベル以上を表示

# --- 定数 ---
TARGET_MODEL_ID = "unsloth/Llama-4-Scout-17B-16E-Instruct"
MIN_TEST_TOKEN_ID = 102
MAX_SAMPLES_PER_CATEGORY = 5
TOTAL_SAMPLES_FOR_LOGIC_TEST = 50


# --- ユーティリティ関数 (分類ロジック再現用ヘルパー) ---
def _calculate_token_flags_util(decoded_token):
    """トークン文字列から文字種フラグを計算する (ユーティリティ関数版)"""
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
        is_digit_char = "0" <= char <= "9"
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
        if is_digit_char:
            flags["has_digit"] = True
        is_pure_jp = is_hira or is_kata_f or is_kata_h or is_kanji
        if not is_pure_jp:
            flags["all_pure_jp_script"] = False
        if not is_basic_eng:
            flags["all_basic_english"] = False
        is_jp_related_char_flag = is_pure_jp or is_jp_ps or is_fw_ascii
        if is_jp_related_char_flag:
            flags["is_related_to_jp"] = True
        is_potentially_special = not (
            char.isalnum()
            or is_space
            or is_jp_related_char_flag
            or is_basic_eng
            or is_digit_char
        )
        if is_potentially_special:
            if not is_special_char_pattern(char):
                flags["has_other_char"] = True
    return flags


def _calculate_expected_categories_util(decoded_token):
    """分類ロジックに基づき、トークンが属すべきカテゴリのセットを計算する (ユーティリティ関数版)"""
    if not decoded_token:
        return set()
    flags = _calculate_token_flags_util(decoded_token)
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
    if (
        not flags["is_related_to_jp"]
        and not flags["has_basic_english"]
        and not flags["has_digit"]
        and not is_sp_pattern
    ):
        expected_cats.add("uncategorized")
    return expected_cats


# ----- ヘルパー関数の単体テストクラス -----
class HelperFunctionTests(unittest.TestCase):
    def test_is_japanese_related_charの拡張ケース(self):
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
                    f"文字 '{char}' (U+{ord(char):04X}) is_japanese_related_char",
                )

    def test_is_pure_japanese_script_charの拡張ケース(self):
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
                    f"文字 '{char}' (U+{ord(char):04X}) is_pure_japanese_script_char",
                )

    def test_is_special_char_patternの拡張ケース(self):
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
            (" 🔥", False),
        ]
        for token, expected in test_cases:
            with self.subTest(token=repr(token), expected=expected):
                actual = is_special_char_pattern(token)
                self.assertEqual(
                    actual, expected, f"トークン {repr(token)} is_special_char_pattern"
                )


# ----- 分類ロジックの単体テストクラス -----
class LogicVerificationTests(unittest.TestCase):
    def test_characteristic_patterns_logic(self):
        """定義された特徴的なパターンについて、分類ロジックが正しく働くか検証する。"""
        # (説明, ポジティブ例, ネガティブ例, 期待カテゴリ(ポジティブ), 除外カテゴリ(ポジティブ), ネガティブ例で除外すべきサブカテゴリ)
        # 説明部分を日本語に修正
        test_patterns = [
            (
                "半角カタカナ",
                "ﾃｽﾄ",
                "テスト",
                {"contains_japanese", "contains_katakana_half", "pure_japanese_script"},
                {"contains_katakana_full", "pure_english"},
                "contains_katakana_half",
            ),
            (
                "全角英字",
                "ＡＢＣ",
                "ABC",
                {"contains_japanese", "contains_fullwidth_ascii"},
                {"pure_japanese_script", "pure_english"},
                "contains_fullwidth_ascii",
            ),
            (
                "全角数字",
                "１２３",
                "123",
                {"contains_japanese", "contains_fullwidth_ascii"},
                {"pure_japanese_script", "pure_english", "contains_digit"},
                "contains_fullwidth_ascii",
            ),
            (
                "全角記号",
                "％＆！",
                "%&!",
                {
                    "contains_japanese",
                    "contains_fullwidth_ascii",
                    "contains_jp_punct_symbol",
                },
                {"pure_japanese_script", "pure_english"},
                "contains_fullwidth_ascii",
            ),
            (
                "純粋ひらがな",
                "あいうえお",
                "アイウエオ",
                {"contains_japanese", "contains_hiragana", "pure_japanese_script"},
                {"contains_katakana_full", "pure_english"},
                "contains_hiragana",
            ),
            (
                "純粋カタカナ(長音符含)",
                "トークン",
                "token",
                {"contains_japanese", "contains_katakana_full", "pure_japanese_script"},
                {"contains_hiragana", "pure_english"},
                "contains_katakana_full",
            ),
            (
                "純粋漢字",
                "日本語",
                "にほんご",
                {"contains_japanese", "contains_kanji", "pure_japanese_script"},
                {"contains_hiragana", "pure_english"},
                "contains_kanji",
            ),
            (
                "純粋英語",
                "HelloWorld",
                "ハローワールド",
                {"contains_basic_english", "pure_english"},
                {"contains_japanese"},
                "pure_english",
            ),
            (
                "特殊文字パターン",
                "---",
                "-abc-",
                {"special_char_pattern"},
                {"contains_japanese", "contains_basic_english", "contains_digit"},
                "special_char_pattern",
            ),
            (
                "半角カナ+英語",
                "ﾃｽﾄABC",
                "テストABC",
                {
                    "contains_japanese",
                    "contains_katakana_half",
                    "contains_basic_english",
                },
                {"pure_japanese_script", "pure_english"},
                "contains_katakana_half",
            ),
            (
                "漢字+数字(半角)",
                "東京1",
                "Tokyo1",
                {"contains_japanese", "contains_kanji", "contains_digit"},
                {"pure_japanese_script", "pure_english"},
                "contains_kanji",
            ),
            (
                "空白のみ",
                "   ",
                "abc",
                {"uncategorized"},
                {"contains_japanese", "contains_basic_english", "special_char_pattern"},
                "uncategorized",
            ),
        ]
        print("\n--- 特徴パターンの分類ロジック検証 ---")
        for (
            description,
            positive_example,
            negative_example,
            expected_cats_positive,
            excluded_cats_positive,
            neg_excluded_subcat,
        ) in test_patterns:
            with self.subTest(description=description, type="ポジティブ"):
                print(f"  検証: '{description}' - ポジティブ例 '{positive_example}'")
                calculated_cats = _calculate_expected_categories_util(positive_example)
                self.assertSetEqual(
                    calculated_cats,
                    expected_cats_positive,
                    f"'{positive_example}' のカテゴリ不一致",
                )
                for excluded_cat in excluded_cats_positive:
                    self.assertNotIn(
                        excluded_cat,
                        calculated_cats,
                        f"'{positive_example}' は '{excluded_cat}' に含まれるべきでない",
                    )
            if negative_example:
                with self.subTest(description=description, type="ネガティブ"):
                    print(
                        f"  検証: '{description}' - ネガティブ例 '{negative_example}'"
                    )
                    calculated_cats_neg = _calculate_expected_categories_util(
                        negative_example
                    )
                    if neg_excluded_subcat:
                        self.assertNotIn(
                            neg_excluded_subcat,
                            calculated_cats_neg,
                            f"ネガティブ例 '{negative_example}' はカテゴリ '{neg_excluded_subcat}' に含まれるべきでない",
                        )


# ----- メイン分析関数の統合テストクラス -----
class AnalysisIntegrationTests(unittest.TestCase):
    tokenizer = None
    result = None
    stats = {}
    token_ids_by_category = {}
    token_ids_by_category_sets = {}
    details = {}

    @classmethod
    def setUpClass(cls):
        print(f"\n--- {cls.__name__} セットアップ開始 ---")
        print(f"テスト対象モデル: {TARGET_MODEL_ID}")
        try:
            cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
                TARGET_MODEL_ID, trust_remote_code=True
            )
            print(f"トークナイザー ({cls.tokenizer.__class__.__name__}) のロード完了")
            print(f"トークン分析を開始します (min_token_id={MIN_TEST_TOKEN_ID})...")
            cls.result = analyze_token_categories(
                TARGET_MODEL_ID, min_token_id=MIN_TEST_TOKEN_ID
            )
            print("トークン分析完了")
            if cls.result is None:
                raise RuntimeError("analyze_token_categories が None")
                cls.stats = cls.result.get("statistics", {})
                cls.token_ids_by_category = cls.result.get("token_ids", {})
                cls.token_ids_by_category_sets = {
                    name: set(ids) for name, ids in cls.token_ids_by_category.items()
                }
                cls.details = cls.result.get(
                    "analysis_details", {}
                )  # detailsがNoneにならないように修正
            print(
                f"分析対象トークン数: {cls.details.get('num_tokens_analyzed', 0):,}"
            )  # .get()を使用
        except Exception as e:
            print(
                f"\n****** セットアップ中に致命的なエラー ******\nエラータイプ: {type(e).__name__}\nエラーメッセージ: {e}"
            )
            import traceback

            print(traceback.format_exc())
            print("***********************************\n")
            cls.result = None
        finally:
            print(f"--- {cls.__name__} セットアップ完了 ---")

    def test_分析結果の基本構造と必須キーの存在確認(self):
        if self.result is None:
            self.fail("セットアップ失敗")
            return
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
        [self.assertIn(key, self.result) for key in expected_top_keys]
        self.assertIsInstance(self.details, dict)
        expected_detail_keys = [
            "min_token_id_analyzed",
            "max_token_id_analyzed",
            "num_tokens_analyzed",
            "num_errors",
            "excluded_special_ids",
        ]
        # --- detailsが空でないことを確認してからキーをチェック ---
        if self.details:  # detailsがNoneや空でない場合のみチェック
            for key in expected_detail_keys:
                self.assertIn(key, self.details, f"details にキー '{key}' がありません")
            self.assertIsInstance(
                self.details.get("excluded_special_ids", []), list
            )  # .get() を使用
        else:
            self.fail(
                "analysis_details が結果に含まれていません、または空です"
            )  # detailsがなければ失敗
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
        # --- .get() を使用して安全にアクセス ---
        self.assertGreaterEqual(
            self.details.get("num_tokens_analyzed", -1), 0
        )  # -1で初期化し、0以上かをチェック
        if self.details.get("num_tokens_analyzed", 0) == 0:
            logging.warning("分析対象トークン数が0でした。")

    def test_カテゴリ間の部分集合関係の検証(self):
        if self.result is None:
            self.fail("セットアップ失敗")
            return

        def check_subset(sub, super_):
            sub_set = self.token_ids_by_category_sets.get(sub, set())
            super_set = self.token_ids_by_category_sets.get(super_, set())
            self.assertTrue(
                sub_set.issubset(super_set), f"'{sub}' ⊆ '{super_}'"
            ) if sub_set or super_set else None  # 空集合同士もTrueなのでチェック不要、どちらかあればチェック

        check_subset("pure_japanese_script", "contains_japanese")
        check_subset("contains_hiragana", "contains_japanese")
        check_subset("contains_katakana_full", "contains_japanese")
        check_subset("contains_katakana_half", "contains_japanese")
        check_subset("contains_kanji", "contains_japanese")
        check_subset("contains_jp_punct_symbol", "contains_japanese")
        check_subset("contains_fullwidth_ascii", "contains_japanese")
        check_subset("pure_english", "contains_basic_english")
        pure_jp_set = self.token_ids_by_category_sets.get("pure_japanese_script", set())
        jp_ps_set = self.token_ids_by_category_sets.get(
            "contains_jp_punct_symbol", set()
        )
        fw_ascii_set = self.token_ids_by_category_sets.get(
            "contains_fullwidth_ascii", set()
        )
        if pure_jp_set and jp_ps_set:
            self.assertTrue(
                pure_jp_set.isdisjoint(jp_ps_set - pure_jp_set),
                "pure_jp と jp_punct_symbol(のみ) は排他",
            )
        if pure_jp_set and fw_ascii_set:
            self.assertTrue(
                pure_jp_set.isdisjoint(fw_ascii_set - pure_jp_set),
                "pure_jp と fw_ascii(のみ) は排他",
            )

    def test_カテゴリ間の排他関係の検証(self):
        if self.result is None:
            self.fail("セットアップ失敗")
            return

        def check_disjoint(cat1, cat2):
            s1 = self.token_ids_by_category_sets.get(cat1, set())
            s2 = self.token_ids_by_category_sets.get(cat2, set())
            if s1 and s2:
                intersection = s1.intersection(s2)
                if intersection:
                    sample_size = 5
                    int_list = sorted(list(intersection))
                    samples = int_list[:sample_size]
                    num_common = len(int_list)
                    msg = f"排他検証失敗: '{cat1}'∩'{cat2}'={num_common} 個共通\n サンプル:{samples}"
                    if self.tokenizer:
                        try:
                            decoded = [
                                f"{t}:{repr(self.tokenizer.decode([t], clean_up_tokenization_spaces=False))}"
                                for t in samples
                            ]
                            msg += f"\n デコード例:{decoded}"
                        except Exception as e:
                            msg += f"\n (デコードエラー:{e})"
                    self.fail(msg)

        check_disjoint("pure_japanese_script", "pure_english")
        check_disjoint("pure_japanese_script", "special_char_pattern")
        check_disjoint("pure_japanese_script", "uncategorized")
        check_disjoint("pure_english", "contains_japanese")
        check_disjoint("pure_english", "special_char_pattern")
        check_disjoint("pure_english", "uncategorized")
        check_disjoint("special_char_pattern", "contains_japanese")
        check_disjoint("special_char_pattern", "contains_basic_english")
        check_disjoint("special_char_pattern", "uncategorized")
        if self.token_ids_by_category_sets.get("uncategorized", set()):
            defined_cats = [
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
            for cat in defined_cats:
                check_disjoint("uncategorized", cat)

    def test_統計値の整合性検証(self):
        if self.result is None:
            self.fail("セットアップ失敗")
            return
        for name, count in self.stats.items():
            self.assertEqual(
                count,
                len(self.token_ids_by_category.get(name, [])),
                f"統計値不一致:{name}",
            )
        all_cat_ids = set().union(
            *[
                s
                for n, s in self.token_ids_by_category_sets.items()
                if n != "uncategorized"
            ]
        )
        num_analyzed = self.details.get("num_tokens_analyzed", 0)
        num_uncat = self.stats.get("uncategorized", 0)
        num_cat_unique = len(all_cat_ids)
        self.assertEqual(
            num_analyzed,
            num_cat_unique + num_uncat,
            f"合計数不一致:分析={num_analyzed:,},分類済={num_cat_unique:,},未分類={num_uncat:,}",
        )

    def test_分類ロジックの一貫性検証_サンプル(self):
        if self.result is None:
            self.fail("セットアップ失敗")
            return
        if self.tokenizer is None:
            self.fail("Tokenizer未ロード")
            return
        # --- .get() を使用して安全にアクセス ---
        if self.details.get("num_tokens_analyzed", 0) == 0:
            self.skipTest("分析対象なし")
            return
        sampled_ids = set()
        cats_to_sample = [
            "pure_japanese_script",
            "pure_english",
            "special_char_pattern",
            "uncategorized",
            "contains_katakana_half",
            "contains_fullwidth_ascii",
            "contains_jp_punct_symbol",
            "contains_digit",
        ]
        for cat in cats_to_sample:
            ids_cat = self.token_ids_by_category_sets.get(cat, set())
            if ids_cat:
                k = min(len(ids_cat), MAX_SAMPLES_PER_CATEGORY)
                sampled_ids.update(random.sample(list(ids_cat), k))
        final_ids = sorted(list(sampled_ids))
        if len(final_ids) > TOTAL_SAMPLES_FOR_LOGIC_TEST:
            final_ids = sorted(random.sample(final_ids, TOTAL_SAMPLES_FOR_LOGIC_TEST))
        if not final_ids:
            self.skipTest("検証サンプルIDなし")
            return
        print(f"\n--- 分類ロジック一貫性検証 (サンプルID数: {len(final_ids)}) ---")
        for tid in final_ids:
            try:
                decoded = self.tokenizer.decode(
                    [tid], clean_up_tokenization_spaces=False
                )
            except Exception as e:
                self.fail(f"ID {tid} デコードエラー: {e}")
                continue
            with self.subTest(tid=tid, decoded=repr(decoded)):
                expected = _calculate_expected_categories_util(decoded)
                actual = {
                    name
                    for name, ids_set in self.token_ids_by_category_sets.items()
                    if tid in ids_set
                }
                self.assertSetEqual(
                    actual, expected, f"ID {tid} ({repr(decoded)}) 分類ロジック不一致"
                )

    def test_partial_japanese_token_samples(self):
        if self.result is None:
            self.fail("セットアップ失敗")
            return
        if self.tokenizer is None:
            self.fail("Tokenizer未ロード")
            return
        contains_jp = self.token_ids_by_category_sets.get("contains_japanese", set())
        pure_jp = self.token_ids_by_category_sets.get("pure_japanese_script", set())
        partial_mixed_ids = list(contains_jp - pure_jp)
        if not partial_mixed_ids:
            self.skipTest("Partial/Mixed候補なし")
            return
        num_samples = min(len(partial_mixed_ids), TOTAL_SAMPLES_FOR_LOGIC_TEST)
        sampled = sorted(random.sample(partial_mixed_ids, num_samples))
        print(
            f"\n--- Partial/Mixed日本語トークン検証 (サンプルID数: {len(sampled)}) ---"
        )
        for tid in sampled:
            try:
                decoded = self.tokenizer.decode(
                    [tid], clean_up_tokenization_spaces=False
                )
            except Exception as e:
                logging.warning(f"ID {tid} デコードエラー (Partial/Mixed検証): {e}")
                continue
            with self.subTest(tid=tid, decoded=repr(decoded)):
                self.assertIn(
                    tid,
                    contains_jp,
                    f"ID {tid} ({repr(decoded)}) は contains_japanese に含まれるべき",
                )
                self.assertNotIn(
                    tid,
                    pure_jp,
                    f"ID {tid} ({repr(decoded)}) は pure_japanese_script に含まれないべき",
                )
                has_jp_flag = any(is_japanese_related_char(c) for c in decoded)
                self.assertTrue(
                    has_jp_flag,
                    f"ID {tid} ({repr(decoded)}) は日本語関連文字を含むべき",
                )


# --- テスト実行 ---
if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(HelperFunctionTests))
    suite.addTest(unittest.makeSuite(LogicVerificationTests))
    suite.addTest(unittest.makeSuite(AnalysisIntegrationTests))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    # if not result.wasSuccessful(): exit(1) # CI用

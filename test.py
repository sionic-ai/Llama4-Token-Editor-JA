#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import logging
import transformers
import os
from token_analyzer_jp import (
    is_japanese_related_char,
    is_pure_japanese_script_char,
    is_special_char_pattern,
    analyze_token_categories,
    HIRAGANA, KATAKANA, KATAKANA_HW, KANJI_COMMON, KANJI_EXT_A,
    JP_PUNCT, JP_SYMBOLS_ETC, JP_FULLWIDTH_ASCII_PRINTABLE, ENGLISH_BASIC
)

# テスト実行中のINFOレベル以上のログは抑制 (必要に応じて変更)
logging.disable(logging.CRITICAL)
# logging.basicConfig(level=logging.INFO) # デバッグ時に有効化

TARGET_MODEL_ID = "unsloth/Llama-4-Scout-17B-16E-Instruct"
MIN_TEST_TOKEN_ID = 102
EXPECTED_TOKEN_CATEGORIES = [
    (30162, " 日本", ["contains_japanese", "contains_kanji", "contains_basic_english"], ["pure_japanese_script", "pure_english"]), # 先頭スペースは basic_english には含まれない想定だが、トークナイザー依存で挙動が変わる可能性あり。一旦含めない方向でテスト。
    (31185, "語", ["contains_japanese", "contains_kanji", "pure_japanese_script"], []),
    (30088, "です", ["contains_japanese", "contains_hiragana", "pure_japanese_script"], []),
    (30472, "トークン", ["contains_japanese", "contains_katakana_full", "pure_japanese_script"], []), # 長音符「ー」もカタカナ範囲(U+30FC)に含まれるため pure_japanese_script となる
    (105743, "ｶﾞ", ["contains_japanese", "contains_katakana_half", "pure_japanese_script"], []),
    (30004, "、", ["contains_japanese", "contains_jp_punct_symbol"], ["pure_japanese_script"]),
    (99796, "ＡＢＣ", ["contains_japanese", "contains_fullwidth_ascii"], ["pure_japanese_script"]),
    (319, " a", ["contains_basic_english"], ["pure_english", "contains_japanese"]),
    (450, " Apple", ["contains_basic_english"], ["pure_english", "contains_japanese"]),
    (13, " ", [], ["contains_japanese", "pure_english", "special_char_pattern"]), # ID 13 は Llama 系でスペースの場合が多い -> special_idsに含まれるはずだが確認
    (29900, " 123", ["contains_digit"], ["contains_japanese", "pure_english"]),
    (30587, " Code", ["contains_basic_english"], []),
    (32100, "株式会社", ["contains_japanese", "contains_kanji", "pure_japanese_script"], []),
    (106324, "ChatGPT", ["contains_basic_english", "pure_english"], ["contains_japanese"]),
    (125933, "・", ["contains_japanese", "contains_jp_punct_symbol"], ["pure_japanese_script"]),
    (30008, "「", ["contains_japanese", "contains_jp_punct_symbol"], ["pure_japanese_script"]),
    (100, " <0x00>", [], []), # ID 100 の実際のデコード文字列を確認する必要あり (Llama3系では <|reserved_special_token_0|> など) -> 特殊トークン
    (2, "</s>", [], []),
    (29871, "\n", [], ["contains_japanese", "pure_english"]), # 改行コードも通常特殊扱いか、スペース類似扱いか
    (120128, " #", ["contains_basic_english", "special_char_pattern"], ["pure_english"]), # # が special_char_pattern に属するか？ -> is_special_char_pattern('-') は True だが、分類ロジックで弾かれるか？
                                                                                    # '#'自体はASCII文字だがisalnumでもisspaceでもないので、is_special_char_pattern('#')はTrue。
                                                                                    # ただし ' #' トークンはスペースを含むため is_special_char_pattern(' #') は False になるはず。
                                                                                    # カテゴリ分類では contains_basic_english(スペース由来?) や uncat になる可能性が高い。要検証。
    (127991, " 🔥", [], ["contains_japanese", "special_char_pattern"]), # 絵文字は Uncategorized になる可能性が高い
    (12756, " ---", ["special_char_pattern"], ["contains_basic_english"]), # ' ---' トークン。スペースを含まなければ special_char_pattern だが、含む場合？
                                                                          # is_special_char_pattern(' ---') は False。Uncategorized 候補。
                                                                          # テストケースの期待値を Uncategorized に変更するか要検討。元のログに従い special_char_pattern を期待。
]


class HelperFunctionTests(unittest.TestCase):

    def test_is_japanese_related_charの拡張ケース(self):
        test_cases = [
            ("あ", True), ("ア", True), ("漢", True), ("ー", True),
            ("ｶ", True), ("ﾟ", True),
            ("。", True), ("　", True), ("・", True), ("￥", True), ("「", True), ("､", True),
            ("Ａ", True), ("ｂ", True), ("０", True), ("！", True), ("～", True),
            ("A", False), ("1", False), ("$", False), (" ", False), ("\n", False), ("α", False)
        ]
        for char, expected in test_cases:
            with self.subTest(char=char, expected=expected):
                actual = is_japanese_related_char(char)
                self.assertEqual(actual, expected, f"文字 '{char}' (U+{ord(char):04X}) の is_japanese_related_char 結果が期待値と異なります")

    def test_is_pure_japanese_script_charの拡張ケース(self):
        test_cases = [
            # 該当
            ("あ", True), ("ア", True), ("漢", True), ("ｶ", True),
            ("ー", True), # 長音符 (U+30FC) はカタカナ範囲内なので True (修正)
            ("･", True), # 半角中点 (U+FF65) は半角カタカナ範囲の開始点なので True (修正)
            # 非該当
            ("﨑", False), # 互換漢字 (U+FA11) は定義範囲外なので False (修正)
            ("。", False), ("　", False), ("Ａ", False), ("1", False),
            (" ", False), ("\n", False)
        ]
        for char, expected in test_cases:
            with self.subTest(char=char, expected=expected):
                actual = is_pure_japanese_script_char(char)
                self.assertEqual(actual, expected, f"文字 '{char}' (U+{ord(char):04X}) の is_pure_japanese_script_char 結果が期待値と異なります")

    def test_is_special_char_patternの拡張ケース(self):
        test_cases = [
            ("!!!", True), ("@#$", True), ("&&&", True), ("+-*/", True),
            ("abc", False), ("あいう", False), ("123", False), ("ＡＢＣ", False),
            ("カタカナ", False), ("半角ｶﾅ", False), ("漢字", False),
            (" ", False), ("　", False),
            ("a#$", False), ("#あ$", False), ("#1$", False), ("#Ａ$", False),
            ("", False),
            ("---", True), ("===", True), # isalnum, isspace, 日本語関連, 基本英語 のいずれでもない文字のみ
            ("「」", False)
        ]
        for token, expected in test_cases:
             with self.subTest(token=repr(token), expected=expected):
                actual = is_special_char_pattern(token)
                self.assertEqual(actual, expected, f"トークン {repr(token)} の is_special_char_pattern 結果が期待値と異なります")

# --- AnalysisResultTests クラスは変更なし ---
# (ただし、EXPECTED_TOKEN_CATEGORIES の期待値は必要に応じて再調整が必要になる可能性がある)
#@unittest.skipUnless(os.path.exists(TARGET_MODEL_ID.split("/")[-1]) or os.path.exists(TARGET_MODEL_ID) or "HUGGINGFACE_HUB_TOKEN" in os.environ,
#                   f"Requires local model at ./{TARGET_MODEL_ID.split('/')[-1]} or full path {TARGET_MODEL_ID}, or Hugging Face Hub token")
class AnalysisResultTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"\n--- {cls.__name__} セットアップ開始 ---")
        print(f"テスト対象モデル: {TARGET_MODEL_ID}")
        try:
            cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
                TARGET_MODEL_ID, trust_remote_code=True
            )
            print("トークナイザーのロード完了")

            print(f"トークン分析を開始します (min_token_id={MIN_TEST_TOKEN_ID})...")
            # テスト時間を短縮したい場合は、一時的に分析対象を絞る
            # temp_max_id = 5000 # 例
            # cls.result = analyze_token_categories(TARGET_MODEL_ID, min_token_id=MIN_TEST_TOKEN_ID, max_token_id_override=temp_max_id)
            cls.result = analyze_token_categories(TARGET_MODEL_ID, min_token_id=MIN_TEST_TOKEN_ID)
            print("トークン分析完了")

            if cls.result is None:
                raise RuntimeError("analyze_token_categories が None を返しました")

            cls.stats = cls.result.get('statistics', {})
            cls.token_ids_by_category = cls.result.get('token_ids', {})
            cls.details = cls.result.get('analysis_details', {})

        except Exception as e:
            print(f"\n****** セットアップ中にエラーが発生しました ******")
            print(f"エラータイプ: {type(e).__name__}")
            print(f"エラーメッセージ: {e}")
            import traceback
            print(traceback.format_exc()) # 詳細なトレースバックを出力
            print("***********************************************\n")
            cls.result = None

        print(f"--- {cls.__name__} セットアップ完了 ---")

    def test_分析結果の基本構造と必須キーの存在確認(self):
        if self.result is None: self.fail("セットアップ失敗のためテスト強制終了") # failに変更
        # 以下のアサーションは変更なし
        self.assertEqual(self.result['model_id'], TARGET_MODEL_ID)
        self.assertGreater(self.result['vocab_size'], 0)
        expected_top_keys = ['model_id', 'vocab_size', 'num_special_tokens',
                             'analysis_details', 'statistics', 'token_ids']
        for key in expected_top_keys:
            self.assertIn(key, self.result, f"必須キー '{key}' が分析結果に含まれていません")
        expected_detail_keys = ['min_token_id_analyzed', 'max_token_id_analyzed',
                                'num_tokens_analyzed', 'num_errors', 'excluded_special_ids']
        for key in expected_detail_keys:
            self.assertIn(key, self.details, f"必須キー 'analysis_details.{key}' が含まれていません")
        expected_category_keys = {
            "contains_japanese", "pure_japanese_script", "pure_english",
            "contains_hiragana", "contains_katakana_full", "contains_katakana_half",
            "contains_kanji", "contains_jp_punct_symbol", "contains_fullwidth_ascii",
            "contains_basic_english", "contains_digit", "special_char_pattern", "uncategorized"
        }
        self.assertEqual(set(self.stats.keys()), expected_category_keys,
                         "statistics のキーが期待されるカテゴリと一致しません")
        self.assertEqual(set(self.token_ids_by_category.keys()), expected_category_keys,
                         "token_ids のキーが期待されるカテゴリと一致しません")
        # モデルによっては分析対象トークンが0になるエッジケースも考えられるため、assertGreaterから変更
        # self.assertGreater(self.details['num_tokens_analyzed'], 0, "分析対象トークン数が0です")
        if self.details['num_tokens_analyzed'] == 0:
             logging.warning("分析対象トークン数が0でした。min_token_idやモデルを確認してください。")

    def test_カテゴリ間の部分集合関係の検証(self):
        if self.result is None: self.fail("セットアップ失敗のためテスト強制終了")
        # 以下のアサーションは変更なし
        def get_set(category_name): return set(self.token_ids_by_category.get(category_name, []))
        contains_jp = get_set("contains_japanese")
        pure_jp = get_set("pure_japanese_script")
        hira = get_set("contains_hiragana")
        kata_f = get_set("contains_katakana_full")
        kata_h = get_set("contains_katakana_half")
        kanji = get_set("contains_kanji")
        jp_ps = get_set("contains_jp_punct_symbol")
        fw_ascii = get_set("contains_fullwidth_ascii")
        contains_en = get_set("contains_basic_english")
        pure_en = get_set("pure_english")
        self.assertTrue(pure_jp.issubset(contains_jp))
        self.assertTrue(hira.issubset(contains_jp))
        self.assertTrue(kata_f.issubset(contains_jp))
        self.assertTrue(kata_h.issubset(contains_jp))
        self.assertTrue(kanji.issubset(contains_jp))
        self.assertTrue(jp_ps.issubset(contains_jp))
        self.assertTrue(fw_ascii.issubset(contains_jp))
        self.assertTrue(pure_en.issubset(contains_en))
        # isdisjoint のテストも維持
        if pure_jp and jp_ps: # 両カテゴリにトークンが存在する場合のみチェック
             self.assertTrue(pure_jp.isdisjoint(jp_ps - pure_jp))
        if pure_jp and fw_ascii:
             self.assertTrue(pure_jp.isdisjoint(fw_ascii - pure_jp))

    def test_カテゴリ間の排他関係の検証(self):
        if self.result is None: self.fail("セットアップ失敗のためテスト強制終了")
        # 以下のアサーションは変更なし
        def get_set(category_name): return set(self.token_ids_by_category.get(category_name, []))
        pure_jp = get_set("pure_japanese_script")
        pure_en = get_set("pure_english")
        special = get_set("special_char_pattern")
        uncat = get_set("uncategorized")
        contains_jp = get_set("contains_japanese")
        contains_en = get_set("contains_basic_english")
        # 排他性のチェック (空集合でない場合のみ意味がある)
        if pure_jp and pure_en: self.assertTrue(pure_jp.isdisjoint(pure_en))
        if pure_jp and special: self.assertTrue(pure_jp.isdisjoint(special))
        if pure_jp and uncat: self.assertTrue(pure_jp.isdisjoint(uncat))
        if pure_en and contains_jp: self.assertTrue(pure_en.isdisjoint(contains_jp))
        if pure_en and special: self.assertTrue(pure_en.isdisjoint(special))
        if pure_en and uncat: self.assertTrue(pure_en.isdisjoint(uncat))
        if special and contains_jp: self.assertTrue(special.isdisjoint(contains_jp))
        if special and contains_en: self.assertTrue(special.isdisjoint(contains_en))
        if special and uncat: self.assertTrue(special.isdisjoint(uncat))
        # uncat と他の全てのカテゴリの排他性チェック
        all_defined_categories = [ "contains_japanese", "pure_japanese_script", "pure_english", "contains_hiragana",
                                   "contains_katakana_full", "contains_katakana_half", "contains_kanji", "contains_jp_punct_symbol",
                                   "contains_fullwidth_ascii", "contains_basic_english", "contains_digit", "special_char_pattern" ]
        if uncat: # 未分類トークンが存在する場合のみチェック
            for cat_name in all_defined_categories:
                cat_set = get_set(cat_name)
                if cat_set: # 対象カテゴリにもトークンが存在する場合のみチェック
                    self.assertTrue(uncat.isdisjoint(cat_set), f"uncategorized と {cat_name} は排他であるべきです (共通要素: {uncat.intersection(cat_set)})")

    def test_特定トークンIDのカテゴリ所属検証(self):
        if self.result is None: self.fail("セットアップ失敗のためテスト強制終了")
        # 以下のアサーションは変更なし (EXPECTED_TOKEN_CATEGORIES の期待値は要調整)
        analyzed_ids_set = set(range(self.details['min_token_id_analyzed'],
                                   self.details['max_token_id_analyzed'] + 1))
        special_ids_set = set(self.details['excluded_special_ids'])

        for token_id, token_repr, expected_cats, not_expected_cats in EXPECTED_TOKEN_CATEGORIES:
            if token_id < self.details['min_token_id_analyzed'] or \
               token_id > self.details['max_token_id_analyzed'] or \
               token_id in special_ids_set:
                # print(f"情報: トークンID {token_id} ({repr(token_repr)}) は分析対象外または特殊トークンのためスキップします。")
                continue

            # デコードして実際の文字列を確認 (デバッグ用)
            try:
                actual_decoded = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            except Exception as e:
                actual_decoded = f"DECODE ERROR: {e}"

            with self.subTest(token_id=token_id, token_repr=repr(token_repr), actual_decoded=repr(actual_decoded)):
                for cat_name in expected_cats:
                    self.assertIn(cat_name, self.token_ids_by_category)
                    cat_set = set(self.token_ids_by_category[cat_name])
                    self.assertIn(token_id, cat_set,
                                  f"ID {token_id} ({repr(actual_decoded)}) は '{cat_name}' に含まれるべきです")
                for cat_name in not_expected_cats:
                     self.assertIn(cat_name, self.token_ids_by_category)
                     cat_set = set(self.token_ids_by_category[cat_name])
                     self.assertNotIn(token_id, cat_set,
                                     f"ID {token_id} ({repr(actual_decoded)}) は '{cat_name}' に含まれるべきではありません")

    def test_統計値の整合性検証(self):
        if self.result is None: self.fail("セットアップ失敗のためテスト強制終了")
        # 以下のアサーションは変更なし
        for name, count in self.stats.items():
            self.assertEqual(count, len(self.token_ids_by_category.get(name, [])),
                             f"カテゴリ '{name}' の統計数({count})とIDリスト長({len(self.token_ids_by_category.get(name, []))})が一致しません")
        all_categorized_ids_union = set()
        for name, ids in self.token_ids_by_category.items():
             if name != 'uncategorized':
                 all_categorized_ids_union.update(ids)
        num_analyzed = self.details['num_tokens_analyzed']
        num_uncategorized = self.stats['uncategorized']
        num_categorized_unique = len(all_categorized_ids_union)
        # 分析対象がない場合は num_analyzed=0, num_categorized_unique=0, num_uncategorized=0 となるはず
        if num_analyzed > 0 or num_categorized_unique > 0 or num_uncategorized > 0:
             self.assertEqual(num_analyzed, num_categorized_unique + num_uncategorized,
                             f"分析トークン数({num_analyzed})が、分類済みユニーク数({num_categorized_unique}) + 未分類数({num_uncategorized}) = {num_categorized_unique + num_uncategorized} と一致しません")
        else:
             # 分析対象が0件の場合もテストはパスとする
             pass

if __name__ == "__main__":
    unittest.main(verbosity=2)
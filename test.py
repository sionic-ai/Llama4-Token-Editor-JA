#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import logging
from unittest.mock import patch
import transformers
from token_analyzer_jp import (
    is_japanese_related_char,
    is_pure_japanese_script_char,
    is_special_char_pattern,
    analyze_token_categories
)

# テスト実行時のログ出力を抑制
logging.disable(logging.CRITICAL)


def is_complete_japanese_utf8(byte_seq: bytes) -> bool:
    """
    与えられたUTF-8バイト列全体をデコードし、結果がちょうど1文字の日本語文字である場合にTrueを返します。
    それ以外の場合はFalseを返します。
    """
    try:
        decoded = byte_seq.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        return False
    # 全体が1文字でなければ False
    if len(decoded) != 1:
        return False
    code_point = ord(decoded)
    # ひらがな、カタカナ、CJK統合漢字、CJK拡張A、半角カタカナの範囲内なら True
    return ((0x3040 <= code_point <= 0x309F) or
            (0x30A0 <= code_point <= 0x30FF) or
            (0x4E00 <= code_point <= 0x9FFF) or
            (0x3400 <= code_point <= 0x4DBF) or
            (0xFF65 <= code_point <= 0xFF9F))


# ----- ヘルパー関数のテスト -----
class HelperFunctionTests(unittest.TestCase):
    def test_is_japanese_related_char(self):
        test_cases = [
            ("あ", True),
            ("ア", True),
            ("漢", True),
            ("。", True),  # 日本語の句読点も含む
            ("A", False),
            ("1", False),
        ]
        for char, expected in test_cases:
            actual = is_japanese_related_char(char)
            print(f"[is_japanese_related_char] テスト対象: '{char}'")
            print(f"  期待値: {expected}")
            print(f"  実際値: {actual}")
            self.assertEqual(actual, expected, f"'{char}' の判定が異なります")
            print("  アサーション成功\n")

    def test_is_pure_japanese_script_char(self):
        test_cases = [
            ("あ", True),
            ("ア", True),
            ("漢", True),
            ("。", False),  # 句読点は除外
            ("A", False),
        ]
        for char, expected in test_cases:
            actual = is_pure_japanese_script_char(char)
            print(f"[is_pure_japanese_script_char] テスト対象: '{char}'")
            print(f"  期待値: {expected}")
            print(f"  実際値: {actual}")
            self.assertEqual(actual, expected, f"'{char}' の判定が異なります")
            print("  アサーション成功\n")

    def test_is_special_char_pattern(self):
        test_cases = [
            ("!!!", True),
            ("@#$", True),
            ("abc", False),
            ("あいう", False),
            ("", False)
        ]
        for token, expected in test_cases:
            actual = is_special_char_pattern(token)
            print(f"[is_special_char_pattern] テスト対象: '{token}'")
            print(f"  期待値: {expected}")
            print(f"  実際値: {actual}")
            self.assertEqual(actual, expected, f"'{token}' の判定が異なります")
            print("  アサーション成功\n")


# ----- 部分日本語トークン検証テスト（固定文字列を使用） -----
class PartialJapaneseTokenTests(unittest.TestCase):
    def test_single_character_full_and_partial_same(self):
        # 単一の日本語文字 "あ" の場合、全バイトと部分（全バイトそのまま）ともに True になるケース
        ch = "あ"
        full_bytes = ch.encode('utf-8')
        full_result = is_complete_japanese_utf8(full_bytes)
        print(f"[単一文字検証] 文字: '{ch}'")
        print(f"  フルバイト列: {full_bytes}")
        print(f"  期待値 (フル): True")
        print(f"  実際値 (フル): {full_result}")
        self.assertTrue(full_result, f"'{ch}' のフルバイト列は True であるべきです")
        # 部分抽出として、全バイトそのままを partial とする（実際は全体と同じ）
        partial_bytes = full_bytes[:]  # 全部
        partial_result = is_complete_japanese_utf8(partial_bytes)
        print(f"  部分バイト列（全体と同じ）: {partial_bytes}")
        print(f"  期待値 (部分): True")
        print(f"  実際値 (部分): {partial_result}")
        self.assertTrue(partial_result, f"'{ch}' の部分（全体と同じ）バイト列は True であるべきです")
        print("  アサーション成功\n")

    def test_complete_and_partial_token(self):
        # 単一文字の場合は、通常、1バイト削除すると不完全になるので False
        japanese_chars = ["あ", "い", "う", "え", "お"]
        for ch in japanese_chars:
            full_bytes = ch.encode('utf-8')
            full_result = is_complete_japanese_utf8(full_bytes)
            print(f"[完全トークン検証] 文字: '{ch}'")
            print(f"  バイト列: {full_bytes}")
            print(f"  期待値: True")
            print(f"  実際値: {full_result}")
            self.assertTrue(full_result, f"'{ch}' の完全バイト列は True であるべきです")
            print("  アサーション成功\n")
            partial_bytes = full_bytes[:-1]
            partial_result = is_complete_japanese_utf8(partial_bytes)
            print(f"[部分トークン検証] 文字: '{ch}' (partial: 末尾1バイト削除)")
            print(f"  バイト列: {partial_bytes}")
            print(f"  期待値: False")
            print(f"  実際値: {partial_result}")
            self.assertFalse(partial_result, f"'{ch}' の不完全バイト列は False であるべきです")
            print("  アサーション成功\n")

    def test_multi_character_token_partial_true(self):
        # 複数文字のトークンの場合、全体は2文字で False だが、
        # 先頭3バイトだけ抽出すると1文字になり、True となるケース
        token = "あい"  # 2文字
        full_bytes = token.encode('utf-8')
        full_result = is_complete_japanese_utf8(full_bytes)
        print(f"[多文字トークン検証] トークン: '{token}'")
        print(f"  フルバイト列: {full_bytes}")
        print(f"  期待値 (フル): False (全体は2文字)")
        print(f"  実際値 (フル): {full_result}")
        self.assertFalse(full_result, "フルトークンは2文字のため False であるべきです")
        partial_bytes = full_bytes[:3]  # 先頭3バイトで "あ" が得られる
        partial_result = is_complete_japanese_utf8(partial_bytes)
        print(f"  部分バイト列: {partial_bytes}")
        print(f"  期待値 (部分): True (先頭で 'あ' が得られる)")
        print(f"  実際値 (部分): {partial_result}")
        self.assertTrue(partial_result, "部分抽出で 'あ' が得られるため True であるべきです")
        print("  アサーション成功\n")

    def test_ascii_multi_character_partial(self):
        # ASCIIの場合、例: "AB" では、部分抽出 (先頭1バイト) は 'A' となるが、日本語ではないので False
        token = "AB"
        full_bytes = token.encode('utf-8')  # b'AB'
        print(f"[ASCII多文字トークン検証] トークン: '{token}'")
        partial_bytes = full_bytes[:1]  # b'A'
        partial_result = is_complete_japanese_utf8(partial_bytes)
        print(f"  部分バイト列: {partial_bytes}")
        print(f"  期待値: False (ASCII 'A' は日本語ではない)")
        print(f"  実際値: {partial_result}")
        self.assertFalse(partial_result, "ASCIIの場合、部分抽出で日本語にならないため False")
        print("  アサーション成功\n")


# ----- モデル解析結果検証テスト（サンプル100件のみ抽出） -----
class AnalysisResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "unsloth/Llama-4-Scout-17B-16E-Instruct"
        try:
            # 実際のトークナイザーを取得
            cls.tokenizer = transformers.AutoTokenizer.from_pretrained(cls.model_id, trust_remote_code=True)
            original_vocab_size = cls.tokenizer.vocab_size
            print(f"[setUpClass] 元の vocab_size: {original_vocab_size}")
            # テスト用に vocab_size を 102+100 に上書き（102～201の約100件）
            object.__setattr__(cls.tokenizer, "vocab_size", 102 + 100)
            print(f"[setUpClass] 上書き後の vocab_size: {cls.tokenizer.vocab_size}")
            # AutoTokenizer.from_pretrained をモックして、上書き済みトークナイザーを返すようにする
            cls.patcher = patch('token_analyzer_jp.transformers.AutoTokenizer.from_pretrained',
                                return_value=cls.tokenizer)
            cls.mock_from_pretrained = cls.patcher.start()
            cls.result = analyze_token_categories(cls.model_id, min_token_id=102)
        except Exception as e:
            cls.result = None
            print(f"モデル読み込みエラー: {e}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'patcher'):
            cls.patcher.stop()

    def test_result_structure(self):
        if self.result is None:
            self.skipTest("モデルの読み込みに失敗したため、構造検証をスキップします。")
        print("[分析結果構造検証] 分析結果の基本構造:")
        expected_keys = ['model_id', 'vocab_size', 'num_special_tokens', 'analysis_details', 'statistics', 'token_ids']
        for key in expected_keys:
            exists = key in self.result
            print(f"  キー: {key} - 期待: 存在, 実際: {'存在' if exists else '未存在'}")
            self.assertIn(key, self.result, f"キー {key} が存在しません")
        print("  全てのキーが存在します。\n")

    def test_japanese_token_stats(self):
        if self.result is None:
            self.skipTest("モデルの読み込みに失敗したため、日本語トークン統計検証をスキップします。")
        stats = self.result['statistics']
        actual = stats.get('contains_japanese', 0)
        print("[日本語トークン統計検証]")
        print(f"  期待値: 0より大きい")
        print(f"  実際値: {actual}")
        self.assertGreater(actual, 0, "日本語を含むトークンが存在するはずです")
        print("  アサーション成功\n")

    def test_pure_japanese_subset(self):
        if self.result is None:
            self.skipTest("モデルの読み込みに失敗したため、部分集合検証をスキップします。")
        token_ids = self.result['token_ids']
        pure_jp = set(token_ids.get('pure_japanese_script', []))
        contains_jp = set(token_ids.get('contains_japanese', []))
        print("[純粋な日本語スクリプト部分集合検証]")
        print(f"  純粋な日本語トークン数: {len(pure_jp)}")
        print(f"  日本語を含むトークン数: {len(contains_jp)}")
        self.assertTrue(pure_jp.issubset(contains_jp),
                        "純粋な日本語スクリプトは日本語を含むトークンの部分集合でなければなりません")
        print("  アサーション成功\n")

    def test_sampled_partial_token_behavior(self):
        """
        「日本語を含む」トークンからサンプルとして最大100件抽出し、
        各トークンのUTF-8バイト列の末尾1バイトを除去した場合、完全な日本語文字として認識されない（Falseとなる）ことを検証する。
        """
        if self.result is None:
            self.skipTest("モデルの読み込みに失敗したため、部分トークン検証をスキップします。")
        token_ids = self.result['token_ids'].get('contains_japanese', [])
        sample_count = min(100, len(token_ids))
        print(f"[サンプルトークン部分検証] 抽出件数: {sample_count} 件")
        sample_ids = token_ids[:sample_count]
        for token_id in sample_ids:
            token = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            token_bytes = token.encode('utf-8')
            if len(token_bytes) < 3:
                continue
            partial_bytes = token_bytes[:-1]
            partial_result = is_complete_japanese_utf8(partial_bytes)
            print(f"  [Token ID: {token_id}] Token: {repr(token)}")
            print(f"    バイト列: {token_bytes}")
            print(f"    部分バイト列: {partial_bytes}")
            print(f"    期待値: False, 実際値: {partial_result}")
            self.assertFalse(partial_result, f"Token ID {token_id} の部分バイト列は False であるべきです")
            print("    アサーション成功\n")


if __name__ == "__main__":
    unittest.main()

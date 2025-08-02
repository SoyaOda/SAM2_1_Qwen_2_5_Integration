# プロジェクト概要

## プロジェクト名
SAM2_1_Qwen_2_5_Integration

## 目的
LISA（Language Instructed Segmentation Assistant）の成功を受けて、最新のVLM（Vision Language Model）と最新のSAM（Segment Anything Model）を統合し、画像と言語を深い次元で理解する基盤モデル（LISA改）を作成する。

### 最終目標
- LISA改の実装
- FoodLMMの学習方法を参考にファインチューニング
- FoodLMM改を作成し、写真内の料理や食材の量の推定を精度高く行う

## 技術スタック
- **VLM**: Qwen2.5-VL (3B/72B)
- **セグメンテーション**: SAM 2.1 (Segment Anything Model)
- **フレームワーク**: PyTorch + Hugging Face Transformers
- **言語**: Python

## 主要コンポーネント
1. **視覚エンコーダ** - Qwen2.5-VLのViTベース（896×896画像、1280次元）
2. **アダプタ層** - ViT出力をSAMデコーダ用に変換（1280→512次元）
3. **SAMマスクデコーダ** - セグメンテーションマスク生成
4. **深度推定ヘッド** - オプションで深度マップ推定
5. **LLM統合** - 特殊トークン（<SEG>）を介したマスク出力

## 現在の状態
- 実装はまだ開始されていない
- 仕様書（o3_spec1.md）が準備されている
- Pythonコードファイルはまだ存在しない
# コーディング規約とスタイル

## 言語
- Python 3.8+

## 命名規則
- クラス名: PascalCase (例: `VisionEncoder`, `MaskDecoder`)
- 関数・変数名: snake_case (例: `image_embeddings`, `forward_pass`)
- 定数: UPPER_SNAKE_CASE (例: `MAX_SEQ_LENGTH`, `PATCH_SIZE`)

## コードスタイル
- PEP 8準拠
- 最大行長: 100文字
- インデント: スペース4つ
- 文字列: ダブルクォート優先

## 型ヒント
- 全ての関数に型ヒントを使用
- `from typing import` で必要な型をインポート
- 例:
```python
def process_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    pass
```

## ドキュメント
- 全てのクラス・関数にdocstring記載
- Google スタイルのdocstring推奨
- 例:
```python
def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """損失を計算する。
    
    Args:
        pred: 予測値のテンソル
        target: 正解値のテンソル
        
    Returns:
        計算された損失値
    """
```

## PyTorch特有の規約
- デバイス指定は明示的に行う
- `nn.Module`を継承したクラスでモデル定義
- `forward()`メソッドで順伝播を定義

## フォーマッター・リンター
- Black（コードフォーマッター）
- Ruff（高速リンター）
- mypy（型チェッカー）
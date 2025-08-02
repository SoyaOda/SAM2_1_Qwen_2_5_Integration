# 推奨コマンド

## 開発環境セットアップ（予定）
```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 依存関係インストール（requirements.txt作成後）
pip install -r requirements.txt
```

## 実行コマンド（予定）
```bash
# モデル訓練
python train.py --config configs/train_config.yaml

# 推論実行
python inference.py --image path/to/image.jpg --prompt "セグメント対象を指定"

# 評価実行
python evaluate.py --dataset coco --checkpoint path/to/checkpoint.pt
```

## テスト・品質管理（予定）
```bash
# ユニットテスト
pytest tests/

# コード品質チェック
ruff check .
black --check .

# 型チェック
mypy .
```

## Git操作
```bash
# 変更確認
git status
git diff

# コミット
git add .
git commit -m "コミットメッセージ"
```

## システムコマンド
- `ls` - ファイル一覧表示
- `cd` - ディレクトリ移動
- `mkdir` - ディレクトリ作成
- `rm` - ファイル削除

注: 現在はコードがまだ実装されていないため、上記のコマンドは将来的に使用予定のものです。
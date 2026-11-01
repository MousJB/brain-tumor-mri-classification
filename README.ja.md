# MRI 画像からの脳腫瘍分類（PyTorch）

[English](README.md)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

脳 MRI のスライスを **4 クラス**（神経膠腫、髄膜腫、下垂体腫瘍、腫瘍なし）に分類し、次の 3 つの手法を
比較するプロジェクトです。

1. ゼロから学習した**独自の CNN**
2. 同じ CNN を**劣化を模したデータ拡張**（ぼかし、ノイズ、モーションブラー）で学習し、ぼかし・ノイズ・
   モーションアーチファクトを加えた画像で評価
3. **ResNet18 による転移学習**（ImageNet で学習済みのバックボーンを固定し、分類層のみ学習）

> [!WARNING]
> **このデータセットの Training/Testing の分割にはリークがあります。** 公開後の監査で、
> **Testing の画像 1,311 枚のうち 216 枚（16.5%）が Training の画像とバイト単位で同一のコピー**
> であることがわかりました。そのため、以下の精度はおそらく楽観的であり、未知の患者に対する性能として
> 読むべきではありません。詳細: [docs/data_audit.ja.md](docs/data_audit.ja.md)

> 学習目的のプロジェクトであり、医療機器ではありません。

---

## データセット

[MRI Brain Tumor Dataset: 4-Class (7,023 Images)](https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images)（Kaggle）

| 分割 | 枚数 |
|---|---|
| Training | 5,712（腫瘍なし 1,595 · 下垂体腫瘍 1,457 · 髄膜腫 1,339 · 神経膠腫 1,321） |
| Testing | 1,311 + 難易度の高い 3 セット: Blurred、Noisy、Patient Motion Artifact |

画像は 224×224 で、グレースケールとして読み込みます。難易度の高いセットは同じ画像を合成的に劣化
させたものであり、実際の撮像条件での性能ではなく、劣化に対する感度を測るものです。

## 手法

**前処理**（[src/dataset.py](src/dataset.py)）: グレースケール化 → 1〜99 パーセンタイルでのクリッピング →
非ゼロ（脳）画素での z スコア正規化

**データ拡張**（任意、それぞれ確率 0.5）: 5×5 のガウシアンぼかし、ガウスノイズ（σ = 5）、
水平方向のモーションブラー（カーネル 3、5、7）

**モデル**（[src/models.py](src/models.py)）:

| モデル | 構成 | 学習するパラメータ |
|---|---|---|
| SimpleCNN | 3 ×（Conv 3×3 → BatchNorm → ReLU → MaxPool）、FC 256、Dropout 0.5 | すべて |
| ResNet18 | ImageNet で事前学習、グレースケールを 3 チャンネルに複製 | 最終の全結合層のみ |

## 結果

元の Google Colab ノートブックで測定した値です。

| モデル | 検証精度 | Blurred | Noisy | Motion |
|---|---|---|---|---|
| SimpleCNN（12 エポック） | 94.58% | – | – | – |
| SimpleCNN + データ拡張 | **98.09%** | 98.33% | 80.38% | 93.92% |
| ResNet18 転移学習（50 エポック） | 記録なし | – | – | – |

これらの数値の限界:

- **リーク**: Testing は Training と 216 枚の同一画像を共有しています（上の警告を参照）。
- **そのままでは再現できない**: 元のノートブックと重みは保存されておらず、検証用データの正確な出所も
  記録されていません。このリポジトリのスクリプトは保存されていたモジュールをもとにパイプラインを
  再実装したものですが、上の数値を確かめるための再実行はまだ行っていません。
- **画像単位の分割**: データセットに患者 ID がないため、同じ患者のスライスが分割の両側に入り得ます。

劣化させたセットの中では Noisy が明らかに最も難しい結果でした（80.38%、他は 93〜98%）。

## プロジェクトの構成

```
brain-tumor-mri-classification/
├── src/
│   ├── dataset.py        # Dataset クラス、前処理、データ拡張
│   ├── models.py         # SimpleCNN、ResNet18（転移学習）
│   └── utils.py          # 評価、混同行列、シード固定
├── train.py              # 学習、検証用分割、チェックポイント保存
├── evaluate.py           # テスト用フォルダーでの精度と混同行列
├── data/                 # データセットの置き場所（バージョン管理外、data/README.md を参照）
├── docs/
│   └── data_audit.md     # Training/Testing のリークの監査
├── requirements.txt
├── LICENSE
├── README.md
└── README.ja.md
```

`models/`（重み）と `results/`（指標、図）はスクリプトが作成し、git では無視されます。

## 使い方

```bash
git clone https://github.com/MousJB/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
pip install -r requirements.txt
```

データセットを `data/` に置いてから（配置は [data/README.md](data/README.md) を参照）、次を実行します。

```bash
# 独自の CNN（データ拡張あり）
python train.py --model simplecnn --epochs 12 --augment

# ResNet18 による転移学習
python train.py --model resnet18 --epochs 50 --augment

# テストセットと劣化させたセットでの評価
python evaluate.py --weights models/simplecnn.pth \
    --test-dirs data/Testing data/Blurred data/Noisy data/Motion
```

`train.py` は Training の 15% を層化抽出で検証用に取り分け（`--val-split`）、検証精度が最も高い
チェックポイントを保存します。すべてのオプションは `python train.py -h` で確認できます。

## 今後の改善

- [ ] 重複を取り除いた、患者単位の分割での再評価（現在の分割にはリークがあります）
- [ ] 説明可能性のための Grad-CAM
- [ ] EfficientNet / Vision Transformer
- [ ] k 分割交差検証
- [ ] Gradio または Streamlit によるデモ

## ライセンス

[MIT](LICENSE)

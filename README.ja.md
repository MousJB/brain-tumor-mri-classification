# 深層学習による MRI 画像からの脳腫瘍分類

[English](README.md)

Python と PyTorch を使い、MRI 画像から**脳腫瘍を分類**するためのさまざまな深層学習の手法を
試したプロジェクトです。

MRI 画像を次の **4 クラス**に分類することを目的としています。
- Glioma（神経膠腫）
- Meningioma（髄膜腫）
- Pituitary（下垂体腫瘍）
- No-tumor（腫瘍なし）

> [!WARNING]
> **公開後に見つかったデータリーク（2026 年 10 月の監査）。**
> **Testing の画像 1,311 枚のうち 216 枚（16.5%）が、Training の画像とバイト単位で同一のコピーです**
> （髄膜腫 106、腫瘍なし 105、下垂体腫瘍 5）。そのため、このデータセットの Training/Testing の
> 分割は独立しておらず、**以下に示す精度はおそらく楽観的です**。この偏りの大きさは測定していません。
> [データ監査](#-データ監査2026-年-10-月)を参照してください。

---

## 📊 データセット

**MRI Brain Tumor Dataset: 4-Class (7,023 Images)**
リンク: [Kaggle Dataset](https://www.kaggle.com/datasets/mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images)

- **学習用（Training）**: 5,712 枚
- **テスト用（Testing）**: 標準のテスト画像 + 3 つの難易度の高いデータセット（Blurred、Noisy、
  Patient Motion Artifact）

**クラスの分布（Training）:**
- No-tumor: 1,595 枚
- Pituitary: 1,457 枚
- Meningioma: 1,339 枚
- Glioma: 1,321 枚

画像はすべて **224×224 ピクセル**です。

---

## 🛠️ 使用技術

- Python
- PyTorch + Torchvision
- OpenCV
- scikit-learn
- Matplotlib & Seaborn
- Google Colab

---

## 🚀 実装したモデル

### 1. SimpleCNN（ゼロから構築した CNN）
- Batch Normalization と Dropout を備えた畳み込みブロック 3 つ
- 12 エポック学習
- 最良の検証精度: **約 94.58%**

### 2. データ拡張を伴うファインチューニング
- 頑健なデータ拡張（ガウシアンぼかし、ガウスノイズ）を追加
- 劣化させたデータセットで頑健性を評価:
  - **Blurred データセット**: 精度 **98.33%**
  - **Noisy データセット**: 精度 **80.38%**
  - **Motion Artifact データセット**: 精度 **93.92%**

### 3. 転移学習 - ResNet18
- ImageNet で事前学習した ResNet18
- 最終の全結合層のみを学習（特徴抽出）
- **ランダムなモーションブラー**を含む、強化したデータ拡張
- 50 エポック学習

---

## 📈 結果のまとめ

| モデル | 検証精度 | Blurred | Noisy | Motion |
|---|---|---|---|---|
| SimpleCNN | 94.58% | - | - | - |
| ファインチューニングした SimpleCNN | 98.09%（最良） | 98.33% | 80.38% | 93.92% |
| ResNet18（転移学習） | 安定して収束 | - | - | - |

**これらの数値の読み方。** これらは、216 枚の同一画像を共有している（下の監査を参照）データセット
の Training/Testing の分割で測定したものです。そのため、未知の患者に対する性能の推定値として扱う
べきではありません。また、**このリポジトリからは再現できません**。数値を出したノートブックは保存
されておらず（コミットされているノートブックは空）、モデルの重みもバージョン管理されておらず、
「検証用」データの出所も記録されていません。Blurred、Noisy、Motion の各セットは同じ画像を合成的に
劣化させたものであり、実際の撮像条件ではなく感度試験です。

---

## 🔍 データ監査（2026 年 10 月）

データセットは [MedVision QA](https://github.com/MousJB/medvision-qa) の重複検出モジュールで監査
しました（スクリプト `experiments/brain_mri/audit_mri_bt.py`、結果は `experiments/brain_mri/mri_bt_audit/`）。
対象は元の Training（5,712 枚）と Testing（1,311 枚）で、「Challenging」セットは除外しています。

| 所見 | 値 |
|---|---|
| Training にバイト単位で同一のコピーがある Testing の画像 | **1,311 枚中 216 枚（16.5%）** |
| ラベルが異なる完全重複 | 0 |
| ラベルが異なる準重複（知覚ハッシュ、距離 ≤ 4） | 50。確認した 24 ペアはすべて異なる画像（誤検出） |
| 公開データセット figshare（Cheng ら）の画像に近い画像 | 1,704 |

**わかること。** テストセットの 6 分の 1 は学習中にすでに使われています。したがって、この分割での
評価は、一部が学習データでの評価になっています。

**わからないこと。** ファイルが同一であることは画像が同一であることを示しますが、患者が同一である
ことは示しません。データセットに患者 ID がないため、同じ患者の別のスライスが Training と Testing に
分かれて入っているかどうかは確認できず、実際の重なりはさらに大きい可能性があります。figshare との
一致は、このデータセットに figshare の画像が含まれていることと矛盾しませんが、それを証明するもの
ではありません。

**公正な評価に必要なこと。** 患者単位の分割です。MedVision QA では、患者 ID を持つ figshare データ
セットでこれを行っています（独立したテスト用患者 64 人、学習前に定めた受入基準、患者単位の
ブートストラップ信頼区間）。そこで再学習した ResNet18 のマクロ F1 は 0.923 [0.867; 0.964] でした。
これは別のモデル・別のタスク（3 クラス、「腫瘍なし」なし）であるため、上の表の数値とは直接比較
できません。

---

## 📁 プロジェクトの構成

```
brain-tumor-mri-classification/
├── notebooks/
│   └── Brain_Tumor_Classification.ipynb     # メインのノートブック
├── src/
│   ├── dataset.py
│   ├── models.py
│   └── utils.py
├── requirements.txt
├── .gitignore
├── LICENSE
├── README.md
└── README.ja.md
```

---

## 🧪 再現の手順

1. リポジトリをクローンします。
   ```bash
   git clone https://github.com/MousJB/brain-tumor-mri-classification.git
   cd brain-tumor-mri-classification
   ```
2. 依存パッケージをインストールします。
   ```bash
   pip install -r requirements.txt
   ```
3. Kaggle からデータセットをダウンロードし、`Training/` と `Testing/` フォルダーを `data/` ディレクトリに置きます。
4. Google Colab または Jupyter Notebook で `Brain_Tumor_Classification.ipynb` を開きます。

---

## 🎯 プロジェクトの目的

- 深層学習による医用画像分類を試す
- データ拡張の効果を調べる
- 劣化させた画像（ぼかし、ノイズ、モーションアーチファクト）でモデルの頑健性を確かめる
- 独自の CNN と転移学習（ResNet18）を比較する

---

## 📌 今後の改善

- EfficientNet や Vision Transformer の実装
- 説明可能性のための Grad-CAM の追加
- k 分割交差検証
- Gradio または Streamlit によるモデルの公開
- 重複を取り除いた、患者単位の分割での再評価（現在の Training/Testing の分割にはリークがあります。データ監査を参照）

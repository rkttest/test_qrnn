# test_qrnn

## 改良予定

### algorithm
- attention mask 導入
- beam search 導入

### 学習法
- ValidationLossが少なくなったらlearningrateを減らす
- early stopping の導入
- 少ないワードを多めにする。短い返信を少なめにする

### モデル
- ワードとタイプを分けて入力として与える
	- 現状GPUサイズが足りなそうなのでちょいネットワークを小さくする

- モデルのアンサンブルを行う

### データ
- コミックデータを導入
- jiman2++を導入

## モデルサマリ
1. 通常のモデル
2. drop 0.4, learning_rate途中変更
3. drop 0.05, learning_rate途中変更
# Fukuwaraii!
- サポーターズ主催ハッカソン(2020年12月26,27日)開発アプリ

## アプリ概要
-  ユーザは自分の顔画像をアップロードすると, サービス内のフォルダで事前に持つ顔画像情報と合わせて,ランダムで顔のパーツを切り出し, 組合せ表示するアプリケーション.

## 目的
- ハッカソンの課題「コロナ禍の年末年始を楽しくするアプリ開発」

## 担当
- 私(熊田)は, Backend(サービス内のフォルダで事前に持つ顔画像情報と合わせて,ランダムで顔のパーツを切り出す)のアルゴリズムを実装した.
- 具体的には, src以下のfind.pyを実装した. 
- その後,frontendと結合し, app.pyを作成した. app.pyに関しては,メンバーと相談しながら行った.

## アプリの売り
-  顔のパーツを表示するので, 匿名性が高い(SNSに投稿しやすい).
-  データが蓄積されていくため, ランダム性が高く飽きない.


## プレゼン資料
- [リンク](https://docs.google.com/presentation/d/1VgksbunSQY3jBK0-RsF1lz1n6cbTx9p08euMEajI49Y/edit#slide=id.gb2b64906ad_1_24)

## アプリ起動方法
1. pip install -r requirement
2. python app.py
* python3系(動作確認3.6.8)で実行してください.

## 使用技術
- Frontend: python, Flask
- Backend: python, opencv

## 今後の課題
- Frontend: 画像生成, SNSとの連携機能の実装
- Backeend: パーツ画像の貼り付け, 事前の顔画像の各パーツの座標データをデータベースに保持する
- serverにdeploy. 現状プロジェクトが約400MBであり, Heerokuの無料ヴァージョンではデプロイがうまくいかない模様.
## 参考
- [PythonでdlibとOpenCVを用いてHelen datasetを学習して顔器官検出](https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac)

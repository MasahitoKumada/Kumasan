#  ファイル説明
- haarcascades: Helen datasetはイリノイ大学のVuong Leさんのサイトで公開されているデータセット. 顔写真及びそれに対応する各顔パーツの座標を含む.
- input: 入力画像を入れるフォルダ.
- output: 出力画像を入れるフォルダ.
- find.py: 複数画像から顔パーツを抽出し, 顔パーツをランダムに選択し組合せた顔画像の座標を標準出力するプログラム. (Backendの)原型.
- test.txt: find.pyの実行結果の一例.

# プログラム実行方法
- pip install -r requirement.txt
- python find.py: input内の元画像について, 顔パーツを切り出し元画像と合わせてoutputに出力するプログラム.

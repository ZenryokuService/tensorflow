import tensorflow as tf

session = tf.InteractiveSession()

# TensorBord出力ディレクトリ
outDir = './dst'

if tf.gfile.Exists(outDir):
    tf.gfile.DeleteRecursively(outDir)

tf.gfile.MakeDirs(outDir)

# 定数で1 + 2
x = tf.constant(1, name='x')
y = tf.constant(2, name='y')
z = x + y

# このコマンドでzをグラフ上に出力
_ = tf.summary.scalar('z', z)

# SummaryWriterでグラフを書く(これより後のコマンドはグラフに出力されない)
summary_writer = tf.summary.FileWriter(outDir , session.graph)

# 実行
print(session.run(z))

# SummaryWriterクローズ
summary_writer.close()

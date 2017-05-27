gcloud ml-engine local train \
--package-path=youtube-8m --module-name=youtube-8m.train -- \
--train_data_pattern=/home/jihaozhe/tfrecord/train*.tfrecord \
--train_dir=/tmp/yt8m_train --model=DenseModel --start_new_model \
--feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128'

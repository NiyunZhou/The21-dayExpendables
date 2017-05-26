gcloud ml-engine local train \
--package-path=youtube-8m --module-name=youtube-8m.train -- \
--train_data_pattern='gs://youtube8m-ml-us-east1/1/frame_level/train/train*.tfrecord' \
--frame_features=True --feature_names="rgb" \
--feature_sizes="1024" --batch_size=1 \
--train_dir=/tmp/yt8m_train --model=LstmModel --start_new_model

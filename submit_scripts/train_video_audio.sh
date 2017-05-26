BUCKET_NAME=gs://yt8m_train
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/train/train*.tfrecord' \
--model=LogisticModel \
--feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128' \
--train_dir=$BUCKET_NAME/yt8m_video_audio_3NN_skip_modi1_l2drpout


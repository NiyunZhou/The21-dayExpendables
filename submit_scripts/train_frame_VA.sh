BUCKET_NAME=gs://yt8m_train
# 提交任务
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/1/frame_level/train/train*.tfrecord' \
--frame_features=True --model=Frame2VideoModel \
--feature_names='rgb,audio' --feature_sizes='1024,128' \
--train_dir=$BUCKET_NAME/yt8m_train_frame_multi_video_level_model \
--batch_size=256

BUCKET_NAME=gs://yt8m_train
JOB_TO_EVAL=yt8m_video_audio_3NN_skip_modi1_l2drpout
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.inference \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --input_data_pattern='gs://youtube8m-ml/1/video_level/test/test*.tfrecord' \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128' \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions0526.csv

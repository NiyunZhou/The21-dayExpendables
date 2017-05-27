BUCKET_NAME=gs://jhz-thu
JOB_TO_EVAL=dense
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/validate/validate*.tfrecord' \
--model=DenseModel \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --run_once=True \
--feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128'

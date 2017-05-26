BUCKET_NAME=gs://yt8m_train
JOB_TO_EVAL=yt8m_train_frame_TS-LSTM_c60s_1epoch
JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.inference \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --input_data_pattern='gs://youtube8m-ml/1/frame_level/test/test*.tfrecord' \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv \
--frame_features=True \
--feature_names="rgb" \
--batch_size=2048


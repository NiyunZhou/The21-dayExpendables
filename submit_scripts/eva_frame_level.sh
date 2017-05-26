BUCKET_NAME=gs://yt8m_train
JOB_TO_EVAL=yt8m_train_frame_TS-LSTM_c60s_1epoch
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/1/frame_level/validate/validate*.tfrecord' \
--model=LstmModel \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --run_once=True \
--output_dir=$BUCKET_NAME/${JOB_TO_EVAL}/ \
--batch_size=512 \
--frame_features=True \
--feature_names="rgb"




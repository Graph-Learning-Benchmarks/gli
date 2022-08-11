max_train_cfg=[];
max_model_cfg=[];
max_acc=0;
for dir in $(find ./grid -mindepth 1 -type d); do 
	echo ${dir};
	train_cfg_dir=$(ls "${dir}" | grep "train");
	model_cfg_dir=$(ls "${dir}" | grep "model");
	echo $train_cfg_dir 
	echo $model_cfg_dir
	acc=$(python3 train.py --model-cfg "${dir}/${model_cfg_dir}" --train-cfg "${dir}/${train_cfg_dir}" --dataset cora | grep "Test Accuracy" | sed -r 's/Test Accuracy (.*)/\1/');
	echo ${acc};
	if (( $(echo "$acc > $max_acc" |bc -l) )); then
		max_acc=${acc}
		max_train_cfg=${train_cfg_dir}
		max_model_cfg=${model_cfg_dir}
	fi
	rm -r ${dir};
done
echo "Max accuracy and configuration:"
echo $max_acc
echo $max_train_cfg
echo $max_model_cfg

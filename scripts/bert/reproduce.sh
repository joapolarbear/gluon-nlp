#!/bin/bash -e
export BYTEPS_TRACE_ON=1
export BYTEPS_TRACE_END_STEP=20
export BYTEPS_TRACE_START_STEP=10
export BYTEPS_TRACE_DIR='./traces'
# export BYTEPS_TRACE_DEBUG=1

export USE_CUDA_PATH=/usr/local/cuda:/usr/local/cudnn/lib64 \
	PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH} \
	LD_LIBRARY_PATH=/usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nccl/lib:$LD_LIBRARY_PATH \
	LIBRARY_PATH=/usr/local/lib:/usr/local/cudnn/lib64:/usr/local/cuda/lib64:$LIBRARY_PATH \
	LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
# export https_proxy=http://10.8.77.222:8118 http_proxy=http://10.8.77.222:8118

export MXNET_GPU_WORKER_NTHREADS=1

# ----------------- re-install byteps -----------------
export DMLC_ROLE="${DMLC_ROLE:-worker}"
if [ "$1" = "yes" ]; then
	if [ $DMLC_ROLE = "worker" ]; then
	update-alternatives --install /usr/bin/gcc gcc $(readlink -f $(which gcc)) 100 
	update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc $(readlink -f $(which gcc)) 100 
	update-alternatives --install /usr/bin/g++ g++ $(readlink -f $(which g++)) 100 
	update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ $(readlink -f $(which g++)) 100
	update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 200 
	update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 200 
	update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 200 
	update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 200

	# -- uninstall first
	cd /usr/local/byteps 
	pip3 uninstall -y byteps
	python3 setup.py clean --all

	# # -- pull and install
	# git pull

	cd /usr/local 
	rm -rf byteps
	git clone --single-branch --branch byteprofile_latency_dev --recurse-submodules https://github.com/joapolarbear/byteps.git 
	cd /usr/local/byteps 

	BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 python3 setup.py install 
	BYTEPS_WITHOUT_PYTORCH=1 BYTEPS_WITHOUT_TENSORFLOW=1 python3 setup.py bdist_wheel

	update-alternatives --remove gcc /usr/bin/gcc-4.9 
	update-alternatives --remove x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 
	update-alternatives --remove g++ /usr/bin/g++-4.9 
	update-alternatives --remove x86_64-linux-gnu-g++ /usr/bin/g++-4.9
	else
		echo "No need to re-install byteps."
	fi
fi


# ---------------------- start to run ----------------------
cd /root

if [ "$1" = "yes" ]; then
	if [ $DMLC_ROLE = "worker" ]; then
	# Set yes to reinstall gluon-nlp and zip, and download dataset.
	# rm -rf gluon-nlp
	# git clone -b test-byteprofile_latency https://github.com/joapolarbear/gluon-nlp.git
	cd gluon-nlp
	python3 setup.py install
	apt-get update && apt-get install -y zip
	mkdir -p /root/.mxnet/models
	cd /root/.mxnet/models 
	wget https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/vocab/book_corpus_wiki_en_uncased-a6607397.zip
	unzip -o *.zip
	else
		echo "No need to re-install gluon-nlp."
	fi
fi

DATA="/tmp/wiki_en_uncased_data/wiki_en_uncased_0*"
OPTIMIZER="bertadam"
cd /root

# optimizer parameters
export LR=0.00354;   
export OPTIONS=--synthetic_data\ --eval_use_npz; 
export WARMUP_RATIO=0.1;          
export NUMSTEPS=281250;   
export CKPTDIR=ckpt_stage1_lamb_16k-682a361-c5fd6fc-0412-cu90; 
export ACC=1;         
export GPUS=0,1

# start
export TRUNCATE_NORM="${TRUNCATE_NORM:-1}"
export LAMB_BULK="${LAMB_BULK:-30}"
export EPS_AFTER_SQRT="${EPS_AFTER_SQRT:-1}"
export NUMSTEPS="${NUMSTEPS:-900000}"
export DTYPE="${DTYPE:-float16}"
export ACC="${ACC:-1}"
export MODEL="${MODEL:-bert_24_1024_16}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
export MAX_PREDICTIONS_PER_SEQ="${MAX_PREDICTIONS_PER_SEQ:-20}"
export LR="${LR:-0.000625}"
export LOGINTERVAL="${LOGINTERVAL:-10}"
export CKPTDIR="${CKPTDIR:-ckpt_stage1_lamb}"
export CKPTINTERVAL="${CKPTINTERVAL:-300000000}"
export OPTIMIZER="${OPTIMIZER:-lamb}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.003125}"
export NVIDIA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
export DMLC_WORKER_ID="${DMLC_WORKER_ID:-0}"
export DMLC_NUM_WORKER="${DMLC_NUM_WORKER:-1}"
export NCCL_MIN_NRINGS="${NCCL_MIN_NRINGS:-16}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD:-120}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD:-120}"
export MXNET_SAFE_ACCUMULATION="${MXNET_SAFE_ACCUMULATION:-1}"
export OPTIONS="${OPTIONS:- }"
export DATA="${DATA:-/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train}"
export DATAEVAL="${DATAEVAL:-/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test}"

echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"

TOTAL_BATCH_SIZE=10
echo "total batch size is $TOTAL_BATCH_SIZE"

python3 /usr/local/byteps/launcher/launch.py \
	python3 /root/gluon-nlp/scripts/bert/run_pretraining.py \
		--data=$DATA \
		--data_eval=$DATAEVAL \
		--optimizer $OPTIMIZER \
		--warmup_ratio $WARMUP_RATIO \
		--num_steps $NUMSTEPS \
		--ckpt_interval $CKPTINTERVAL \
		--dtype $DTYPE \
		--ckpt_dir $CKPTDIR \
		--lr $LR \
		--accumulate $ACC \
		--model $MODEL \
		--max_seq_length $MAX_SEQ_LENGTH \
		--max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
		--num_data_workers 4 \
		--no_compute_acc \
		--log_interval $LOGINTERVAL \
		--total_batch_size $TOTAL_BATCH_SIZE \
		--total_batch_size_eval $TOTAL_BATCH_SIZE \
		--gpus $NVIDIA_VISIBLE_DEVICES \
		--synthetic_data \
		--comm_backend byteps

		# --profile test


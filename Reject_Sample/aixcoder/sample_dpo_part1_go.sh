# 第一步：定义变量
model_path="models/aixcoder/sft-models"
model_name=aixcoder-0124-go


# 定义模式
mode="dpo_sample"  
# 根据模式选择不同的 prompt 文件
if [ "$mode" == "dpo_sample" ]; then
  prompt_files=(
    "dpo_data/go_api_output.jsonl"
    "dpo_data/go_line_output.jsonl"
    "dpo_data/go_structured_span_output.jsonl"
  )
  work_dir="Reject_Sample"
  seed_data="po_data/go_api.jsonl"
  temperature=1.5
  num_samples=10
  BLEU_threshold=0.6
elif [ "$mode" == "evaluate" ]; then
  prompt_files=(
    "Reject_Sampling/datasets/prompt_api_seed_test.jsonl"
    "Reject_Sampling/datasets/prompt_line_seed_test.jsonl"
    "Reject_Sampling/datasets/prompt_structured_span_seed_test.jsonl"
  )
  work_dir="Reject_Sampling/Result_Evaluate"
  seed_data="Reject_Sampling/datasets/api_seed_test.jsonl"
  temperature=0
  num_samples=1
  BLEU_threshold=0
else
  echo "无效的模式：$mode"
  exit 1
fi


echo "当前模式：$mode"
echo "使用的 prompt 文件："
for file in "${prompt_files[@]}"; do
  echo "$file"
done


tensor_parallel_size=8
port=8000 # 主机端口可固定，容器映射 8000

docker_name="vllm_model_${model_name}"

docker kill $docker_name || true
docker rm $docker_name || true

docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --name ${docker_name} \
  -v /nfs100:/nfs100 \
  -p ${port}:8000 \
  --shm-size=8g \
  vllm: \
  --model="${model_path}" \
  --served-model-name=${model_name} \
  --tensor-parallel-size=${tensor_parallel_size} \
  --max-model-len=16384


sleep 180
echo "Docker 容器 ${docker_name} 已启动"


# 第三步：获取容器的 IP 地址和构造 vllm_url
container_ip=$(docker inspect -f '{{.NetworkSettings.Networks.bridge.IPAddress}}' ${docker_name})
if [[ -z "${container_ip}" ]]; then
  echo "未能获取容器 IP 地址，脚本终止。"
  exit 1
fi
vllm_url="http://${container_ip}:8000/v1/completions"
echo "构造的 vllm_url 为: ${vllm_url}"

# 第四步：循环遍历每个 prompt 文件并运行推理
for prompt_file in "${prompt_files[@]}"; do
  echo "Running inference for prompt file: ${prompt_file}"
  if [[ "$prompt_file" == *"api"* ]]; then
    inf_work_dir="${work_dir}/${model_name}/api"
  elif [[ "$prompt_file" == *"line"* ]]; then
    inf_work_dir="${work_dir}/${model_name}/line"
  elif [[ "$prompt_file" == *"span"* ]]; then
    inf_work_dir="${work_dir}/${model_name}/span"
  fi

  docker run --rm -it --name inference_ \
    -v /nfs100:/nfs100 \
    -v /dev/shm:/dev/shm \
    trl_env:0910 python Reject_Sampling/inference.py \
    --work_dir="${inf_work_dir}" \
    --model_name="${model_name}" \
    --model_path="${model_path}" \
    --prompt_file="${prompt_file}" \
    --vllm_url="${vllm_url}" \
    --temperature="${temperature}" \
    --num_samples="${num_samples}"
  
  docker wait inference_
  # 打印分隔符，区分每次推理
  echo "Finished inference for prompt file: ${prompt_file}"
  echo "----------------------------------------"
done

# 第五步：执行评估脚本
scripts=(
  "/nfs100/zhuhao/dpo/DPO_dataset/eval_api.py"
  "/nfs100/zhuhao/dpo/DPO_dataset/eval_line.py"
  "/nfs100/zhuhao/dpo/DPO_dataset/eval_span.py"
)

# 构造运行命令的字符串
commands=""
for script in "${scripts[@]}"; do
  if [[ $script == *eval_api.py ]]; then
    commands+="python $script --work_dir=${work_dir} --model_name=${model_name} \
    --model_path=${model_path} --bleu_threshold=${BLEU_threshold} --seed_data=${seed_data} && "
  else
    commands+="python $script --work_dir=${work_dir} --model_name=${model_name} \
    --model_path=${model_path} --bleu_threshold=${BLEU_threshold} && "
  fi
done

# 移除最后的 "&&"
commands=${commands%&& }

# 在 Docker 容器中运行
docker run --rm -it --name eval__test \
  -v /nfs100:/nfs100 \
  -v /dev/shm:/dev/shm \
  _deveval \
  bash -c "$commands"

docker stop ${docker_name}
docker rm ${docker_name}

echo "所有脚本在容器中运行完成。"
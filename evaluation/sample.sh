# 第一步：定义变量
model_path="aixcoder/sft"
model_name=aixcoder_test
lora_name=$model_name

# if not doing DPO training
lora_path=""

prompt_model_name="aixcoder_our_style"

prompt_dir="prompt"  # Directory for generating or storing prompts
seed_data_dir="test_data"  # Directory for benchmark data downloaded as seed data

languages=("python" "java" "cplus" "go")
code_types=("api" "line" "structured_span")
mode="evaluate"
prompt_files=()
for lang in "${languages[@]}"; do
  for code_type in "${code_types[@]}"; do
    prompt_files+=("${prompt_dir}/${prompt_model_name}/${lang}_${code_type}.jsonl")
  done
done

work_dir="Result_Evaluate/${prompt_model_name}"
temperature=0
num_samples=1
BLEU_threshold=0


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

# Container with pre-installed vllm dependencies for model serving

docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --name ${docker_name} \
  -v /nfs100:/nfs100 \
  -p ${port}:8000 \
  --shm-size=8g \
  vllm \
  --model=${model_path} \
  --served-model-name=${model_name} \
  --tensor-parallel-size=${tensor_parallel_size} \
  --max-model-len=16384 \


sleep 180
echo "Docker 容器 ${docker_name} 已启动"

container_ip=$(docker inspect -f '{{.NetworkSettings.Networks.bridge.IPAddress}}' ${docker_name})
if [[ -z "${container_ip}" ]]; then
  echo "未能获取容器 IP 地址，脚本终止。"
  exit 1
fi
vllm_url="http://${container_ip}:8000/v1/completions"
echo "构造的 vllm_url 为: ${vllm_url}"

for prompt_file in "${prompt_files[@]}"; do
  echo "Running inference for prompt file: ${prompt_file}"
  # 从 prompt 文件名中提取语言和代码类型
  if [[ "$prompt_file" == *"api"* ]]; then
    typ="api"
  elif [[ "$prompt_file" == *"line"* ]]; then
    typ="line"
  elif [[ "$prompt_file" == *"span"* ]]; then
    typ="span"
  fi

  if [[ "$prompt_file" == *"python"* ]]; then
    lang="python"
  elif [[ "$prompt_file" == *"java"* ]]; then
    lang="java"
  elif [[ "$prompt_file" == *"cplus"* ]]; then
    lang="cplus"
  elif [[ "$prompt_file" == *"go"* ]]; then
    lang="go"
  fi
  inf_work_dir="${work_dir}/${lang}/${lora_name}/${typ}"
  docker run --rm -it --name inference_lhy \
    -v /nfs100:/nfs100 \
    -v /dev/shm:/dev/shm \
    trl_env:0910 python evaluation/inference.py \
    --work_dir="${inf_work_dir}" \
    --model_name="${lora_name}" \
    --model_path="${model_path}" \
    --prompt_file="${prompt_file}" \
    --vllm_url="${vllm_url}" \
    --temperature="${temperature}" \
    --num_samples="${num_samples}"
  
  docker wait inference_lhy
  echo "Finished inference for prompt file: ${prompt_file}"
  echo "----------------------------------------"
done

# sample done! evaluation
scripts=(
  "evaluation/eval_api.py"
  "evaluation/eval_line.py"
  "evaluation/eval_span.py"
)


for lang in "${languages[@]}"; do
  eval_work_dir="${work_dir}/${lang}"
  commands=""
  for script in "${scripts[@]}"; do
    if [[ $script == *eval_api.py ]]; then
      seed_data="${seed_data_dir}/${lang}_api.jsonl"
      commands+="python $script --work_dir=${eval_work_dir} --model_name=${lora_name} \
      --model_path=${model_path} --bleu_threshold=${BLEU_threshold} --seed_data=${seed_data} && "
    else
      commands+="python $script --work_dir=${eval_work_dir} --model_name=${lora_name} \
      --model_path=${model_path} --bleu_threshold=${BLEU_threshold} && "
    fi
  done

  commands=${commands%&& }

  docker run --rm -it --name eval_ \
    -v /nfs100:/nfs100 \
    -v /dev/shm:/dev/shm \
    _deveval \
    bash -c "$commands"
done

docker stop ${docker_name}
docker rm ${docker_name}

echo "所有脚本在容器中运行完成"



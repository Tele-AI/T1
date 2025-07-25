<div align="center">
<h1>
  T1
</h1>
</div>


<p align="center">
	🦉 <a href="https://github.com/Tele-AI/T1" target="_blank">github</a> • 🤗 <a href="https://huggingface.co/Tele-AI" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/organization/TeleAI" target="_blank">ModelScope</a> • 🐾 <a href="https://gitee.com/Tele-AI/T1" target="_blank">gitee</a> • 💬 <a href="https://github.com/Tele-AI/Telechat/blob/master/images/wechat.jpg" target="_blank">WeChat</a>
</p>

# 目录

- [模型介绍](#模型介绍)
- [效果评测](#效果评测)
- [模型推理](#模型推理)
- [国产化适配](#国产化适配)
- [声明、协议、引用](#声明协议引用)

# 模型介绍

**T1** 模型是 **TeleChat** 系列专注于复杂推理的模型，由中国电信人工智能研究院基于国产算力研发训练。该系列模型借助先进的思维推理和批判纠错能力，在下游复杂任务中有很好的表现。本次我们开源了 **T1-35B** 和 **T1-115B** 两款不同尺寸的模型，与同尺寸模型相比都具有较好的效果表现。

### 训练策略
采用课程学习贯穿全流程的后训练方案，循序渐进提升模型效果。

- 微调阶段：将多任务数据集进行难度划分（根据模型推理正误比率判断），首先使用中低难度冷启动微调，然后使用RFT方式筛选中高难度数据进行持续微调进行效果提升；
- 强化学习阶段：首先对数理逻辑、代码能力进行提升，采用难度渐进式课程学习方案进行能力强化；然后，基于指令遵循、安全、幻觉、Function Call等10多种混合通用任务进行持续强化，全面提升模型效果；

### 模型下载
| 模型版本    | 下载链接                                                      |
|---------|-----------------------------------------------------------|
| T1-35B  | [modelscope](https://modelscope.cn/models/TeleAI/T1-35B)  |
| T1-115B | [modelscope](https://modelscope.cn/models/TeleAI/T1-115B) |

# 效果评测
| 模型                           | MATH-500 | AlignBench | BFCL(avg v1&v2) |
|------------------------------|----------|------------|-----------------|
| OpenAI o1-mini               | 90       | 7.91       | -               |
| DeepSeek-R1-Distill-Qwen-32B | 94.3     | 7.42       | 76.14           |
| QWQ-32B                      | 96       | 7.97       | 83.1            |
| Qwen3-32B（长推理）               | 93       | 8.27       | 86.82           |
| **T1-35B**                       | 90       | 7.93       | 80.11           |
| **T1-115B**                      | 94       | 8.22       | 83.39           |


# 模型推理

### Transformers
T1 系列模型支持使用 `transformers` 库进行推理，示例如下：


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("T1/T1-35B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "T1/T1-35B",
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
prompt = "生抽和酱油的区别是什么？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages,
	tokenize=False,
   	add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```
<details>
<summary>推理结果</summary>

```markdown

嗯，用户问生抽和酱油的区别，这看起来是一个关于调味品的问题。首先，我需要确定用户的基本需求。可能他们在做饭时不确定该用哪种，或者想了解它们的用途。接下来，我得回忆一下生抽和酱油的定义和特点。

生抽通常是指以大豆和面粉为原料，经过发酵制成的液体调味品，颜色较浅，味道偏咸鲜。而酱油一般指的是老抽，颜色深，味道更浓郁，常用于上色。不过，可能用户对“酱油”的理解有偏差，因为有些地方可能把酱油泛指所有酱油类调味品，包括生抽和老抽。

然后，我需要考虑用户的背景。可能是家庭烹饪者，尤其是新手，对调料不太熟悉。他们可能想知道在炒菜或凉拌时该用哪种，或者如何正确使用以达到最佳效果。此外，用户可能没有明确区分生抽、老抽和其他类型的酱油，比如味极鲜、蒸鱼豉油等。

接下来，要分析用户可能的深层需求。除了基本的区别，他们可能还关心使用场景、健康因素（比如盐分含量）或者如何替代使用。例如，生抽适合凉拌和炒菜，而老抽适合红烧和卤味。如果用户有健康顾虑，可能需要提到低盐产品。

还要考虑常见的误区。比如，有些人可能认为酱油就是生抽，或者混淆了生抽和老抽的功能。需要明确指出两者的不同用途，避免用户在烹饪时出错。

另外，用户可能对“酱油”这个词的广义理解有误，认为所有酱油都是同一种东西。这时候需要解释清楚，酱油是一个大类，包含多种类型，而生抽是其中的一种，主要用于提鲜，老抽用于上色。

最后，总结回答时要结构清晰，分点说明区别，包括原料、颜色、味道、用途和健康提示。确保语言通俗易懂，避免专业术语过多，让用户容易理解。同时，提醒用户根据烹饪需求选择合适的调料，可能还需要举例说明常见菜品中使用的情况，帮助用户更好地应用知识。
</think>

生抽和酱油都是常用的调味品，但它们在原料、制作工艺、风味和用法上有明显区别。以下是详细的区分：

---

### **1. 原料与工艺**
- **生抽**  
  - **原料**：以大豆（黄豆或黑豆）、小麦粉为主，辅以盐、水、曲种（米曲霉）。  
  - **工艺**：通过发酵（固态或液态）制成，颜色较浅，呈红褐色或琥珀色，味道偏咸鲜，含较多氨基酸。  

- **酱油**（传统广式老酱油或老抽）  
  - **原料**：同样以大豆、小麦为基础，但可能添加焦糖色、糖类或其他增色成分。  
  - **工艺**：发酵后需长时间晾晒或烘烤，颜色更深（深褐色），咸味较重，鲜味略逊于生抽。

---

### **2. 风味差异**
- **生抽**：鲜味突出，咸中带甜，适合直接调味。  
- **酱油**：咸味浓郁，部分产品带有焦糖香气，常用于上色或提味。

---

### **3. 用途区别**
- **生抽**：  
  - 凉拌菜（如拌黄瓜、海带）增鲜提味。  
  - 炒菜（如小炒肉、青菜）代替盐，保持食材原味。  
  - 蘸料（如白切鸡、刺身）提鲜。  

- **酱油**：  
  - 红烧、炖煮（如红烧肉、卤味）上色和增香。  
  - 腌制或蘸食（如蘸饺子、拌面）增加咸香。

---

### **4. 健康注意事项**
- **盐分**：两者含盐量较高，生抽钠含量约为15%（每百毫升），酱油约18%，需注意控制用量。  
- **添加剂**：部分生抽含谷氨酸钠（味精），增加鲜味；酱油可能添加焦糖色素或防腐剂，选择时可参考配料表。

---

### **常见误区澄清**
- **“酱油=老抽”**：传统广式酱油（如豉油）与老抽（北方称法）类似，但部分地区将“酱油”泛指点缀所有酱油类（包括生抽、老抽、味极鲜等）。购买时需看清包装标注。  
- **替代使用**：生抽不可完全替代酱油，因上色能力不足；老抽咸味过重，不宜直接用于凉拌。

---

### **总结**
| **对比项**   | **生抽**                | **酱油（老抽）**          |
|--------------|-------------------------|--------------------------|
| **颜色**     | 淡红褐色                | 深褐色                  |
| **鲜味**     | 突出                    | 较弱                    |
| **咸味**     | 中等                    | 较浓                    |
| **主要用途** | 凉拌、炒菜、蘸料        | 上色、炖煮、腌制        |

根据烹饪需求选择：追求鲜味用生抽，需要上色或浓郁咸香用酱油。
```
</details>

#### ModelScope
T1 系列模型支持使用 ModelScope 推理，示例如下：
```python
import os
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
tokenizer = AutoTokenizer.from_pretrained('T1/T1-35BB', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('T1/T1-35B', trust_remote_code=True, device_map="auto",
                                                  torch_dtype=torch.bfloat16)
prompt = "生抽与老抽的区别？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### vLLM 推理

T1 支持使用 [vLLM](https://github.com/vllm-project/vllm) 进行部署与推理加速，示例如下:
##### 离线推理
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("T1/T1-35B", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.6, repetition_penalty=1.05, max_tokens=8192)
llm = LLM(model="T1/T1-35B", trust_remote_code=True, tensor_parallel_size=4, dtype="bfloat16")

prompt = "生抽和酱油的区别是什么？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

outputs = llm.generate([text], sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

##### OpenAI 兼容的 API 服务
您可以借助 vLLM，构建一个与 OpenAI API 兼容的 API 服务。请按照以下所示运行命令：
```
vllm serve T1/T1-35B \
    --trust-remote-code \
    --dtype bfloat16 \
    --disable-custom-all-reduce
```

然后，您可以与 T1 进行对话：
```python
from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
chat_response = client.chat.completions.create(
    model="T1/T1-35B",
    messages=[
        {"role": "user", "content": "生抽和酱油的区别是什么？"},
    ],
    temperature=0.6,
    max_tokens=8192,
    extra_body={
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    },
)
print("Chat response:", chat_response)
```


#### 推理注意事项

1. T1 系列模型在 chat template 中加入了一些适配复杂推理模型的特性：
    - T1 系列模型在 chat template 中加入了`<think>\n`符号以确保推理时能够生成 reason 过程。如果借助 `transformers` 库推理，并采用`apply_chat_template`方法，且 `add_generation_prompt` 设为`True`，则将会在推理时自动拼接`<think>\n`符号；如果使用 vLLM 库推理，也会自动在推理起始拼接`<think>\n`符号。此时输出结果会缺少开头的`<think>\n`符号。
    - T1 系列模型在进行多轮推理时不应传入之前轮次回答中的`<think>..</think>`过程，在chat template 中已经实现了对多轮历史信息的自动处理。

2. T1 系列模型推理参数选择
    - 在推理数学、代码任务时，建议使用`repetition_penalty=1.0, temperature=0.6, top_p=0.95`的推理设置。
    - 在推理通用任务时，建议使用`repetition_penalty=1.05, temperature=0.6, top_p=0.95`的推理设置，可以有效减少重复生成现象。


# 国产化适配

T1系列模型均进行了**国产化算力适配**，具体信息可见
1. <a href="https://modelers.cn/models/MindSpore-Lab/T1-35B" target="_blank">MindSpore-Lab/T1-35B</a>
2. <a href="https://modelers.cn/models/MindSpore-Lab/T1-115B" target="_blank">MindSpore-Lab/T1-115B</a>


# 声明、引用

### 声明

我们在此声明，不要使用 T1 系列模型及其衍生模型进行任何危害国家社会安全或违法的活动。同时，我们也要求使用者不要将 T1 系列模型用于没有安全审查和备案的互联网服务。我们希望所有使用者遵守上述原则，确保科技发展在合法合规的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用 T1 系列开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

### 引用

如需引用我们的工作，请使用如下 reference：

```
@misc{wang2025technicalreporttelechat2telechat25,
      title={Technical Report of TeleChat2, TeleChat2.5 and T1}, 
      author={Zihan Wang and Xinzhang Liu and Yitong Yao and Chao Wang and Yu Zhao and Zhihao Yang and Wenmin Deng and Kaipeng Jia and Jiaxin Peng and Yuyao Huang and Sishi Xiong and Zhuo Jiang and Kaidong Yu and Xiaohui Hu and Fubei Yao and Ruiyu Fang and Zhuoru Jiang and Ruiting Song and Qiyi Xie and Rui Xue and Xuewei He and Yanlei Xue and Zhu Yuan and Zhaoxi Zhang and Zilu Huang and Shiquan Wang and Xin Wang and Hanming Wu and Mingyuan Wang and Xufeng Zhan and Yuhan Sun and Zhaohu Xing and Yuhao Jiang and Bingkai Yang and Shuangyong Song and Yongxiang Li and Zhongjiang He and Xuelong Li},
      year={2025},
      eprint={2507.18013},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.18013}, 
}
```


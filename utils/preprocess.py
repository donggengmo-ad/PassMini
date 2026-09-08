from scripts import data, experiment
from scripts.tokenizer import CharTokenizer
from pathlib import Path
import json

# 数据配置
config = experiment.DataConfig()
source_path = Path("../data/raw/rockyou.txt")
tot_size = data.count_lines(source_path) * 0.9
train_size = int(tot_size * 0.6)
val_size = int(tot_size * 0.1)
test_size = int(tot_size * 0.3)
seed = 42

# 准备数据
datastat = data.prepare_dataset(source_path=source_path,
                                output_dir=config.dataset_path,
                                train_size=train_size,
                                val_size=val_size,
                                test_size=test_size,
                                seed=seed)
with open(config.dataset_path / "datastat.json", "w", encoding="utf-8") as f:
    json.dump(datastat, f, indent=4, ensure_ascii=False)

# 训练分词器
train_dataset = data.read_dataset(config.dataset_path)["train"]
tokenizer = CharTokenizer.from_text(train_dataset)
tokenizer.dump(config.tokenizer_path)
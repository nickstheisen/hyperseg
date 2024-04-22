import hydra
from omegaconf import DictConfig, OmegaConf
from hyperseg.datasets import get_datamodule
import time
from tqdm import tqdm

@hydra.main(version_base=None, config_path="conf", config_name="train_conf")
def benchmark(cfg):
    print(OmegaConf.to_yaml(cfg.dataset))
        
    t1 = time.time()
    datamodule = get_datamodule(cfg.dataset) 
    datamodule.setup()

    for data in tqdm(datamodule.train_dataloader()):
        image, labels = data
    t2 = time.time()
    print(f"time {t2-t1}")

if __name__ == '__main__':
    benchmark()

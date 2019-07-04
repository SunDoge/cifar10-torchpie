from torchpie.config import config
import torchpie.parallel as tpp

if __name__ == "__main__":
    print(config)
    print(config.get_string('arch'))
    print(tpp.distributed)
from utils.model_presets import imagenet_clust_pretrained


config = imagenet_clust_pretrained['ResNet18']
model = config.generate()

for name, param in model.state_dict().items():
    if 'weight' not in name:
        continue

    try:
        param.int_repr()
        print(f"integer representaion valid: {name} {'weight' in name}")
    except:
        print(f"integer representaion invalid: {name}")
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

traits = {
    "face" : {
        "shape": ["oval-shaped", "round", "rectangular","heart-shaped"],
    },
    "neck": {
        "shape": ["long", "short", "thin", "thick", "fat", "slender"],
    },
    "skin" : {
        "tone": ["rosy", "ruddy", "olive", "brown", "light", "tanned", "creamy", "pink", "pale"],
    },
    "hair" : {
        "color": ["black", "brown", "blond", "white", "red", "blue", "gray"],
        "style": ["long", "short", "curly", "wavy", "spiky", "crew-cut", "bald", "dreadlocks", "ponytail", "side-parting", "centre parting"]
    },
    "eyes" : {
        "color": ["black", "brown", "blue", "green", "gray","red"],
        "shape": ["almond", "round", "down-turned", "upturned", "deep-set", "wide-set", "close-set"],
    },
    "eyebrows" : {
        "shape": ["bushy", "thin", "normal"],
        "color": ["black", "brown", "blond", "white", "red", "blue", "gray"],
    },
    "eyelashes" : {
        "length": ["long", "medium", "short"],
    },
    "lips" : {
        "shape": ["thin", "full","heart-shaped"],
        "color" : ["pink", "red"],
    },
    "nose" : {
        "shape": ["high", "long", "winged", "round", "upturned", "bumpy"],
        "size": ["small", "big"],
    },
    "ears" : {
        "shape": ["round", "long", "pointed"],
        "size": ["small", "big", "large"],
        "type": ["attached", "detached"],
    },
    "chin" : {
        "shape": ["pointed", "round", "double", "square", "chiselled", "strong", "narrow", "wide"],
    },
    "jaw" : {
        "shape": ["strong", "angular", "sharp", "chiselled", "narrow", "wide"],
    },
    "others" : {
        "sex": ["girl", "woman", "man", "boy"],
        "age": ["baby", "child", "very young", "young", "middle-aged", "old", "very old"],
        "glasses": ["glasses", "no glasses", "sun-glasses", "round glasses", "oval glasses", "square glasses"],
        "beard": ["beard", "no beard", "mustache", "goatee", "full beard", "unshaven", "sideburns", "3 days beard"],
    }
}


#    "glasses" : {
#        "style": ["no", "rimmed", "rimless", "aviator", "cat-eye", "round", "square", "browline", "butterfly", "oval", "wayfarer"],
#    },


# Initialize processor and model

#processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
#model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(mps_device)



def fitImage(path,f):
    try:    
        images = [Image.open(f"{path}/{f}")]
        fit = []
        for trait, features in traits.items():
            instances = []
            for feature, values in features.items():
                phrases = [f"({value} {trait})" for value in values]
                inputs = processor(text=phrases, images = images, padding=True, return_tensors="pt").to(mps_device)
                outputs = model(**inputs)
                image_logits = outputs.logits_per_image.to(mps_device)
                image_logits = image_logits.cpu().detach().numpy()
                maxIdx = image_logits.argmax()
                instances.append(values[maxIdx])
            if (trait == "others"):
                fit.append(', '.join(instances)) 
            else:
                fit.append(f"{' '.join(instances)} {trait}")
        return ", ".join(fit)
    except:
        return None


for i in range(1,201):
    fit = fitImage("./dataset/noBg",f"Bild{i}.png")
    if (fit):
        print(f"{i},{fit}")

from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import argparse


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


def fitImage(path,f, console):
    try:    
        filePath = os.path.join(path, f)
        images = [Image.open(filePath)]
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
    

def writeTagFile(path, f, fit, prefix):
    root, _ = os.path.splitext(f)
    # Create a new filename with the new extension
    tagFile = f"{root}.txt"
    filePath = os.path.join(path, tagFile)
    with open(filePath, 'w') as file:
        # Write the string to the file
        file.write(f"{prefix}, {fit}" if prefix else fit)

def printTagsAsCsv(f, fit):
    print(f"{f},{fit}")

def process_files(directory_path, file, console, prefix):
    # Check if the path exists and is a directory
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # List all files in the directory
        for filename in [file] if file else tqdm(os.listdir(directory_path)):
            fit = fitImage(directory_path, filename, console)
            if fit:
                if (console):
                    printTagsAsCsv(filename, fit)
                else:
                    writeTagFile(directory_path, filename, fit, prefix)    
    else:
        print("Invalid directory path.")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Fits portraits from files in a directory.")

    # Add directory_path argument
    parser.add_argument("--directory_path", type=str, help="Path to the directory containing files.")
    parser.add_argument("--file", type=str, help="Filename to fit.")
    parser.add_argument("--prefix", type=str, default="", help="Constant prefix to print.")
    parser.add_argument("--console", action='store_true', help="Do not create files but print to console (csv)")

    # Parse command-line arguments
    args = parser.parse_args()

    # Process files in the directory
    process_files(args.directory_path, args.file, args.console, args.prefix)




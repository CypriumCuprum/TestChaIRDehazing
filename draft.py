import os
from PIL import Image
root = "D:\CUPRUM\PTIT\IEC\Test\datasynthesize"
filelist = os.listdir(os.path.join(root, "ref_imgs"))
length = len(filelist)
length_train = (length*2)//3
length_test = length//6

train = filelist[:length_train]
valid = filelist[length_train:(length_train+length_test)]
test = filelist[(length_train+length_test):]

namehazedir  = ["outputs10", "outputs15", "outputs20", "outputs25", "outputs30"]
def updatedir(filelist, phase="train"):
    for filename in filelist:
        img = Image.open(os.path.join(root, "ref_imgs", filename)).convert("RGB")
        img.save(os.path.join(root,phase, "ref_imgs", filename))

        haze_name = filename.split(".")[0]+"_synt.jpg"

        for dir in namehazedir:
            haze_img = Image.open(os.path.join(root, dir, haze_name)).convert("RGB")
            haze_img.save(os.path.join(root, phase, dir, haze_name))

if __name__ == '__main__':
    updatedir(train, phase="train")
    updatedir(valid, phase = "valid")
    updatedir(test, phase="test")

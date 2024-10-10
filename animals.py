from roboflow import Roboflow
rf = Roboflow(api_key="f7jPzAPiQv5SoDs56FpL")
project = rf.workspace("mydataset-lg9gk").project("animalsdataset")
version = project.version(1)
dataset = version.download("coco-mmdetection")
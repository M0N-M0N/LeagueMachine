import os
import ultralytics
from roboflow import Roboflow


# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="fI2JUojEYxbJofSC2lzX")

# Retrieve your current workspace and project name
print(rf.workspace())

# Specify the project for upload
# let's you have a project at https://app.roboflow.com/my-workspace/my-project
workspaceId = 'leaguemachine'
projectId = 'leaguemachine'
project = rf.workspace(workspaceId).project(projectId)


if __name__ == '__main__':
    version = project.version(5)
    # version.deploy("yolov8", "training1")
    version.deploy("yolov8", "runs/detect/train6/weights/", "last.pt")
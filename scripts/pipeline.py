from detector import CarDetector
from moviepy.editor import VideoFileClip
import sys

video_to_process = sys.argv[1]
detector = CarDetector.load()

def pipeline(frame):
	result = detector.detect_full_pipeline(frame)
	return result


clip = VideoFileClip(video_to_process)#.subclip(35, 40)
video_output = "out_" + video_to_process
frame = clip.fl_image(pipeline)

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

with PyCallGraph(output=GraphvizOutput()):
	frame.write_videofile(video_output, audio=False)